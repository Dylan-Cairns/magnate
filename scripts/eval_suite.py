from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.eval_suite import evaluate_side_swapped, wilson_interval
from trainer.policies import MctsConfig, SearchConfig, policy_from_name


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
        help="Candidate policy (random|heuristic|search|mcts|ppo).",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy (random|heuristic|search|mcts|ppo).",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --candidate-policy=ppo.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --opponent-policy=ppo.",
    )
    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--mcts-worlds", type=int, default=6)
    parser.add_argument("--mcts-simulations", type=int, default=96)
    parser.add_argument("--mcts-depth", type=int, default=24)
    parser.add_argument("--mcts-max-root-actions", type=int, default=8)
    parser.add_argument("--mcts-c-puct", type=float, default=1.25)
    parser.add_argument(
        "--search-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for search.",
    )
    parser.add_argument(
        "--mcts-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for MCTS.",
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for guidance checkpoints.",
    )
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
            "candidateCheckpoint": (
                str(args.candidate_checkpoint) if args.candidate_checkpoint else None
            ),
            "opponentCheckpoint": (
                str(args.opponent_checkpoint) if args.opponent_checkpoint else None
            ),
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
            "mcts": {
                "worlds": args.mcts_worlds,
                "simulations": args.mcts_simulations,
                "depth": args.mcts_depth,
                "maxRootActions": args.mcts_max_root_actions,
                "cPuct": args.mcts_c_puct,
            },
            "guidance": {
                "searchCheckpoint": (
                    str(args.search_guidance_checkpoint)
                    if args.search_guidance_checkpoint
                    else None
                ),
                "mctsCheckpoint": (
                    str(args.mcts_guidance_checkpoint)
                    if args.mcts_guidance_checkpoint
                    else None
                ),
                "temperature": args.guidance_temperature,
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
            candidate_checkpoint=args.candidate_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
            mcts_worlds=args.mcts_worlds,
            mcts_simulations=args.mcts_simulations,
            mcts_depth=args.mcts_depth,
            mcts_max_root_actions=args.mcts_max_root_actions,
            mcts_c_puct=args.mcts_c_puct,
            search_guidance_checkpoint=args.search_guidance_checkpoint,
            mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
            guidance_temperature=args.guidance_temperature,
        )

    worker_count = min(args.workers, args.games_per_side)
    if worker_count == 1:
        return _run_eval_shard(
            games_per_side=args.games_per_side,
            seed_prefix=args.seed_prefix,
            seed_start_index=args.seed_start_index,
            candidate_policy=args.candidate_policy,
            opponent_policy=args.opponent_policy,
            candidate_checkpoint=args.candidate_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
            mcts_worlds=args.mcts_worlds,
            mcts_simulations=args.mcts_simulations,
            mcts_depth=args.mcts_depth,
            mcts_max_root_actions=args.mcts_max_root_actions,
            mcts_c_puct=args.mcts_c_puct,
            search_guidance_checkpoint=args.search_guidance_checkpoint,
            mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
            guidance_temperature=args.guidance_temperature,
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
                "candidateCheckpoint": (
                    str(args.candidate_checkpoint) if args.candidate_checkpoint else None
                ),
                "opponentCheckpoint": (
                    str(args.opponent_checkpoint) if args.opponent_checkpoint else None
                ),
                "searchWorlds": args.search_worlds,
                "searchRollouts": args.search_rollouts,
                "searchDepth": args.search_depth,
                "searchMaxRootActions": args.search_max_root_actions,
                "searchRolloutEpsilon": args.search_rollout_epsilon,
                "mctsWorlds": args.mcts_worlds,
                "mctsSimulations": args.mcts_simulations,
                "mctsDepth": args.mcts_depth,
                "mctsMaxRootActions": args.mcts_max_root_actions,
                "mctsCPuct": args.mcts_c_puct,
                "searchGuidanceCheckpoint": (
                    str(args.search_guidance_checkpoint)
                    if args.search_guidance_checkpoint
                    else None
                ),
                "mctsGuidanceCheckpoint": (
                    str(args.mcts_guidance_checkpoint) if args.mcts_guidance_checkpoint else None
                ),
                "guidanceTemperature": args.guidance_temperature,
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
    candidate_checkpoint: Path | None,
    opponent_checkpoint: Path | None,
    search_worlds: int,
    search_rollouts: int,
    search_depth: int,
    search_max_root_actions: int,
    search_rollout_epsilon: float,
    mcts_worlds: int,
    mcts_simulations: int,
    mcts_depth: int,
    mcts_max_root_actions: int,
    mcts_c_puct: float,
    search_guidance_checkpoint: Path | None,
    mcts_guidance_checkpoint: Path | None,
    guidance_temperature: float,
) -> Dict[str, object]:
    search_config = SearchConfig(
        worlds=search_worlds,
        rollouts=search_rollouts,
        depth=search_depth,
        max_root_actions=search_max_root_actions,
        rollout_epsilon=search_rollout_epsilon,
    )
    mcts_config = MctsConfig(
        worlds=mcts_worlds,
        simulations=mcts_simulations,
        depth=mcts_depth,
        max_root_actions=mcts_max_root_actions,
        c_puct=mcts_c_puct,
    )

    candidate = policy_from_name(
        candidate_policy,
        checkpoint_path=candidate_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=search_guidance_checkpoint,
        mcts_guidance_checkpoint=mcts_guidance_checkpoint,
        guidance_temperature=guidance_temperature,
    )
    opponent = policy_from_name(
        opponent_policy,
        checkpoint_path=opponent_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=search_guidance_checkpoint,
        mcts_guidance_checkpoint=mcts_guidance_checkpoint,
        guidance_temperature=guidance_temperature,
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
    candidate_checkpoint = payload.get("candidateCheckpoint")
    opponent_checkpoint = payload.get("opponentCheckpoint")
    search_guidance_checkpoint = payload.get("searchGuidanceCheckpoint")
    mcts_guidance_checkpoint = payload.get("mctsGuidanceCheckpoint")

    return _run_eval_shard(
        games_per_side=int(payload["gamesPerSide"]),
        seed_prefix=str(payload["seedPrefix"]),
        seed_start_index=int(payload["seedStartIndex"]),
        candidate_policy=str(payload["candidatePolicy"]),
        opponent_policy=str(payload["opponentPolicy"]),
        candidate_checkpoint=Path(str(candidate_checkpoint)) if candidate_checkpoint else None,
        opponent_checkpoint=Path(str(opponent_checkpoint)) if opponent_checkpoint else None,
        search_worlds=int(payload["searchWorlds"]),
        search_rollouts=int(payload["searchRollouts"]),
        search_depth=int(payload["searchDepth"]),
        search_max_root_actions=int(payload["searchMaxRootActions"]),
        search_rollout_epsilon=float(payload["searchRolloutEpsilon"]),
        mcts_worlds=int(payload["mctsWorlds"]),
        mcts_simulations=int(payload["mctsSimulations"]),
        mcts_depth=int(payload["mctsDepth"]),
        mcts_max_root_actions=int(payload["mctsMaxRootActions"]),
        mcts_c_puct=float(payload["mctsCPuct"]),
        search_guidance_checkpoint=(
            Path(str(search_guidance_checkpoint)) if search_guidance_checkpoint else None
        ),
        mcts_guidance_checkpoint=(
            Path(str(mcts_guidance_checkpoint)) if mcts_guidance_checkpoint else None
        ),
        guidance_temperature=float(payload["guidanceTemperature"]),
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
    draws = sum(int(result["draws"]) for result in shard_results)
    opponent_wins = total_games - candidate_wins - draws

    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    average_turn_weighted = 0.0
    for result in shard_results:
        shard_winners = result["winners"]
        if not isinstance(shard_winners, dict):
            raise RuntimeError("Shard result is missing winners map.")
        winners["PlayerA"] += int(shard_winners.get("PlayerA", 0))
        winners["PlayerB"] += int(shard_winners.get("PlayerB", 0))
        winners["Draw"] += int(shard_winners.get("Draw", 0))
        average_turn_weighted += float(result["averageTurn"]) * int(result["totalGames"])

    candidate_wins_as_player_a = 0
    candidate_wins_as_player_b = 0
    for result in shard_results:
        legs = result.get("legs")
        if not isinstance(legs, dict):
            raise RuntimeError("Shard result is missing legs map.")
        leg_a = legs.get("candidateAsPlayerA")
        leg_b = legs.get("candidateAsPlayerB")
        if not isinstance(leg_a, dict) or not isinstance(leg_b, dict):
            raise RuntimeError("Shard result has invalid leg payloads.")
        leg_a_winners = leg_a.get("winners")
        leg_b_winners = leg_b.get("winners")
        if not isinstance(leg_a_winners, dict) or not isinstance(leg_b_winners, dict):
            raise RuntimeError("Shard leg payload is missing winners map.")
        candidate_wins_as_player_a += int(leg_a_winners.get("PlayerA", 0))
        candidate_wins_as_player_b += int(leg_b_winners.get("PlayerB", 0))

    candidate_win_rate = candidate_wins / float(total_games)
    ci_low, ci_high = wilson_interval(candidate_wins, total_games)
    candidate_win_rate_as_player_a = candidate_wins_as_player_a / float(games_per_side)
    candidate_win_rate_as_player_b = candidate_wins_as_player_b / float(games_per_side)

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
        "candidateWinRateCi95": {"low": ci_low, "high": ci_high},
        "candidateWinRateAsPlayerA": candidate_win_rate_as_player_a,
        "candidateWinRateAsPlayerB": candidate_win_rate_as_player_b,
        "sideGap": abs(candidate_win_rate_as_player_a - candidate_win_rate_as_player_b),
        "averageTurn": average_turn_weighted / float(total_games),
        "legs": {
            "candidateAsPlayerA": _merge_leg(shard_results, "candidateAsPlayerA"),
            "candidateAsPlayerB": _merge_leg(shard_results, "candidateAsPlayerB"),
        },
    }


def _merge_leg(
    shard_results: List[Dict[str, object]],
    leg_key: str,
) -> Dict[str, object]:
    games = 0
    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    wins_by_policy: Dict[str, int] = defaultdict(int)
    average_turn_weighted = 0.0

    for result in shard_results:
        legs = result.get("legs")
        if not isinstance(legs, dict):
            raise RuntimeError("Shard result is missing legs map.")
        leg = legs.get(leg_key)
        if not isinstance(leg, dict):
            raise RuntimeError(f"Shard result is missing leg '{leg_key}'.")

        leg_games = int(leg.get("games", 0))
        games += leg_games
        average_turn_weighted += float(leg.get("averageTurn", 0.0)) * leg_games

        leg_winners = leg.get("winners")
        if not isinstance(leg_winners, dict):
            raise RuntimeError(f"Leg '{leg_key}' is missing winners map.")
        winners["PlayerA"] += int(leg_winners.get("PlayerA", 0))
        winners["PlayerB"] += int(leg_winners.get("PlayerB", 0))
        winners["Draw"] += int(leg_winners.get("Draw", 0))

        leg_wins_by_policy = leg.get("winsByPolicy")
        if not isinstance(leg_wins_by_policy, dict):
            raise RuntimeError(f"Leg '{leg_key}' is missing winsByPolicy map.")
        for policy_name, count in leg_wins_by_policy.items():
            wins_by_policy[str(policy_name)] += int(count)

    average_turn = (average_turn_weighted / float(games)) if games > 0 else 0.0
    return {
        "games": games,
        "winners": winners,
        "winsByPolicy": dict(wins_by_policy),
        "averageTurn": average_turn,
    }


def _default_output_path(
    *,
    seed_prefix: str,
    candidate_policy: str,
    opponent_policy: str,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed_prefix = _slug(seed_prefix)
    safe_candidate = _slug(candidate_policy)
    safe_opponent = _slug(opponent_policy)
    return (
        Path("artifacts/evals")
        / f"{stamp}-{safe_seed_prefix}-suite-{safe_candidate}-vs-{safe_opponent}.json"
    )


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
