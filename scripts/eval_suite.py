from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.eval_suite import evaluate_side_swapped, wilson_interval
from trainer.policies import SearchConfig, policy_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical side-swapped paired-seed evaluation and report "
            "win rate, Wilson CI, and side-gap."
        )
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=100,
        help="Games with candidate on each seat (total games = 2x).",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="eval-suite",
        help="Shared seed prefix used for both side-swapped legs.",
    )
    parser.add_argument(
        "--candidate-policy",
        type=str,
        default="search",
        help="Candidate policy (random|heuristic|search).",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy (random|heuristic|search).",
    )
    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/evals/<utc>-<seed-prefix>-suite-<candidate>-vs-<opponent>.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Each worker runs an independent bridge process.",
    )
    parser.add_argument(
        "--seed-start-index",
        type=int,
        default=0,
        help="Seed index offset for deterministic sharding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.seed_start_index < 0:
        raise SystemExit("--seed-start-index must be >= 0.")
    if args.games_per_side <= 0:
        raise SystemExit("--games-per-side must be > 0.")

    results = _evaluate_results(args)

    output_path = args.out or _default_output_path(
        seed_prefix=args.seed_prefix,
        candidate_policy=args.candidate_policy,
        opponent_policy=args.opponent_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "gamesPerSide": args.games_per_side,
            "seedPrefix": args.seed_prefix,
            "seedStartIndex": args.seed_start_index,
            "workers": args.workers,
            "candidatePolicy": args.candidate_policy,
            "opponentPolicy": args.opponent_policy,
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "candidateWinRate": payload["results"]["candidateWinRate"],
                "candidateWinRateCi95": payload["results"]["candidateWinRateCi95"],
                "sideGap": payload["results"]["sideGap"],
                "candidateWins": payload["results"]["candidateWins"],
                "opponentWins": payload["results"]["opponentWins"],
                "draws": payload["results"]["draws"],
                "totalGames": payload["results"]["totalGames"],
            },
            indent=2,
        )
    )
    return 0


def _evaluate_results(args: argparse.Namespace) -> Dict[str, object]:
    if args.workers == 1:
        return _run_eval_shard(
            games_per_side=args.games_per_side,
            seed_prefix=args.seed_prefix,
            seed_start_index=args.seed_start_index,
            candidate_policy=args.candidate_policy,
            opponent_policy=args.opponent_policy,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
        )

    worker_count = min(args.workers, args.games_per_side)
    if worker_count == 1:
        return _run_eval_shard(
            games_per_side=args.games_per_side,
            seed_prefix=args.seed_prefix,
            seed_start_index=args.seed_start_index,
            candidate_policy=args.candidate_policy,
            opponent_policy=args.opponent_policy,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
        )

    shard_sizes = _split_games(args.games_per_side, worker_count)

    payloads: List[Dict[str, object]] = []
    seed_index = args.seed_start_index
    for shard_games in shard_sizes:
        payloads.append(
            {
                "gamesPerSide": shard_games,
                "seedPrefix": args.seed_prefix,
                "seedStartIndex": seed_index,
                "candidatePolicy": args.candidate_policy,
                "opponentPolicy": args.opponent_policy,
                "searchWorlds": args.search_worlds,
                "searchRollouts": args.search_rollouts,
                "searchDepth": args.search_depth,
                "searchMaxRootActions": args.search_max_root_actions,
                "searchRolloutEpsilon": args.search_rollout_epsilon,
            }
        )
        seed_index += shard_games

    shard_results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_evaluate_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            shard_results.append(future.result())

    return _merge_shard_results(shard_results)


def _run_eval_shard(
    *,
    games_per_side: int,
    seed_prefix: str,
    seed_start_index: int,
    candidate_policy: str,
    opponent_policy: str,
    search_worlds: int,
    search_rollouts: int,
    search_depth: int,
    search_max_root_actions: int,
    search_rollout_epsilon: float,
) -> Dict[str, object]:
    search_config = SearchConfig(
        worlds=search_worlds,
        rollouts=search_rollouts,
        depth=search_depth,
        max_root_actions=search_max_root_actions,
        rollout_epsilon=search_rollout_epsilon,
    )

    candidate = policy_from_name(
        candidate_policy,
        search_config=search_config,
    )
    opponent = policy_from_name(
        opponent_policy,
        search_config=search_config,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)
            summary = evaluate_side_swapped(
                env=env,
                candidate_policy=candidate,
                opponent_policy=opponent,
                games_per_side=games_per_side,
                seed_prefix=seed_prefix,
                seed_start_index=seed_start_index,
            )
    finally:
        candidate.close()
        opponent.close()

    return summary.to_json()


def _evaluate_shard(payload: Dict[str, object]) -> Dict[str, object]:
    return _run_eval_shard(
        games_per_side=int(payload["gamesPerSide"]),
        seed_prefix=str(payload["seedPrefix"]),
        seed_start_index=int(payload["seedStartIndex"]),
        candidate_policy=str(payload["candidatePolicy"]),
        opponent_policy=str(payload["opponentPolicy"]),
        search_worlds=int(payload["searchWorlds"]),
        search_rollouts=int(payload["searchRollouts"]),
        search_depth=int(payload["searchDepth"]),
        search_max_root_actions=int(payload["searchMaxRootActions"]),
        search_rollout_epsilon=float(payload["searchRolloutEpsilon"]),
    )


def _split_games(total_games_per_side: int, workers: int) -> List[int]:
    base = total_games_per_side // workers
    remainder = total_games_per_side % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _merge_shard_results(shard_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not shard_results:
        raise RuntimeError("No shard results to merge.")

    candidate_name = str(shard_results[0]["candidate"])
    opponent_name = str(shard_results[0]["opponent"])
    for result in shard_results[1:]:
        if str(result["candidate"]) != candidate_name or str(result["opponent"]) != opponent_name:
            raise RuntimeError("Shard results use different candidate/opponent policies.")

    games_per_side = sum(int(result["gamesPerSide"]) for result in shard_results)
    total_games = sum(int(result["totalGames"]) for result in shard_results)
    candidate_wins = sum(int(result["candidateWins"]) for result in shard_results)
    opponent_wins = sum(int(result["opponentWins"]) for result in shard_results)
    draws = sum(int(result["draws"]) for result in shard_results)

    winners = {
        "PlayerA": sum(int(result["winners"]["PlayerA"]) for result in shard_results),
        "PlayerB": sum(int(result["winners"]["PlayerB"]) for result in shard_results),
        "Draw": draws,
    }

    candidate_wins_as_a = 0
    candidate_wins_as_b = 0
    leg_a_games = 0
    leg_b_games = 0
    leg_a_turn_total = 0.0
    leg_b_turn_total = 0.0

    wins_by_policy_a: Dict[str, int] = defaultdict(int)
    wins_by_policy_b: Dict[str, int] = defaultdict(int)
    leg_a_winners: Dict[str, int] = defaultdict(int)
    leg_b_winners: Dict[str, int] = defaultdict(int)

    for result in shard_results:
        leg_a = result["legs"]["candidateAsPlayerA"]
        leg_b = result["legs"]["candidateAsPlayerB"]

        leg_a_games += int(leg_a["games"])
        leg_b_games += int(leg_b["games"])

        leg_a_turn_total += float(leg_a["averageTurn"]) * int(leg_a["games"])
        leg_b_turn_total += float(leg_b["averageTurn"]) * int(leg_b["games"])

        candidate_wins_as_a += int(leg_a["winners"]["PlayerA"])
        candidate_wins_as_b += int(leg_b["winners"]["PlayerB"])

        for key, value in leg_a["winsByPolicy"].items():
            wins_by_policy_a[str(key)] += int(value)
        for key, value in leg_b["winsByPolicy"].items():
            wins_by_policy_b[str(key)] += int(value)

        for key, value in leg_a["winners"].items():
            leg_a_winners[str(key)] += int(value)
        for key, value in leg_b["winners"].items():
            leg_b_winners[str(key)] += int(value)

    candidate_win_rate = candidate_wins / float(total_games)
    ci_low, ci_high = wilson_interval(candidate_wins, total_games)

    candidate_win_rate_as_a = candidate_wins_as_a / float(max(1, leg_a_games))
    candidate_win_rate_as_b = candidate_wins_as_b / float(max(1, leg_b_games))
    side_gap = abs(candidate_win_rate_as_a - candidate_win_rate_as_b)

    average_turn = (
        (leg_a_turn_total + leg_b_turn_total) / float(max(1, leg_a_games + leg_b_games))
    )

    return {
        "gamesPerSide": games_per_side,
        "totalGames": total_games,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "winners": winners,
        "candidateWins": candidate_wins,
        "opponentWins": opponent_wins,
        "draws": draws,
        "candidateWinRate": candidate_win_rate,
        "candidateWinRateCi95": {
            "low": ci_low,
            "high": ci_high,
        },
        "candidateWinRateAsPlayerA": candidate_win_rate_as_a,
        "candidateWinRateAsPlayerB": candidate_win_rate_as_b,
        "sideGap": side_gap,
        "averageTurn": average_turn,
        "legs": {
            "candidateAsPlayerA": {
                "games": leg_a_games,
                "winners": dict(leg_a_winners),
                "winsByPolicy": dict(wins_by_policy_a),
                "averageTurn": (leg_a_turn_total / float(max(1, leg_a_games))),
            },
            "candidateAsPlayerB": {
                "games": leg_b_games,
                "winners": dict(leg_b_winners),
                "winsByPolicy": dict(wins_by_policy_b),
                "averageTurn": (leg_b_turn_total / float(max(1, leg_b_games))),
            },
        },
    }


def _default_output_path(seed_prefix: str, candidate_policy: str, opponent_policy: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed = _slug(seed_prefix)
    safe_candidate = _slug(candidate_policy)
    safe_opponent = _slug(opponent_policy)
    return Path("artifacts/evals") / f"{stamp}-{safe_seed}-suite-{safe_candidate}-vs-{safe_opponent}.json"


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
