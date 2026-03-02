from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import evaluate_matchup
from trainer.policies import (
    SearchConfig,
    TDSearchPolicyConfig,
    TDValuePolicyConfig,
    policy_from_name,
)

DEFAULT_PROGRESS_EVERY_GAMES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Magnate policies.")
    parser.add_argument("--games", type=int, default=50, help="Number of games to run.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="eval",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--player-a-policy",
        type=str,
        required=True,
        help="Policy name for PlayerA (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        required=True,
        help="Policy name for PlayerB (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument(
        "--search-worlds",
        type=int,
        default=6,
        help="Search policy world samples per decision.",
    )
    parser.add_argument(
        "--search-rollouts",
        type=int,
        default=1,
        help="Search policy rollouts per action per sampled world.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=14,
        help="Search policy rollout depth limit (decision steps after root action).",
    )
    parser.add_argument(
        "--search-max-root-actions",
        type=int,
        default=6,
        help="Search policy candidate root actions kept after heuristic pre-ranking.",
    )
    parser.add_argument(
        "--search-rollout-epsilon",
        type=float,
        default=0.04,
        help="Search rollout epsilon for random exploratory moves.",
    )
    parser.add_argument(
        "--td-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when a policy is td-value.",
    )
    parser.add_argument(
        "--td-worlds",
        type=int,
        default=8,
        help="Determinization world samples per decision for td-value policy.",
    )
    parser.add_argument(
        "--td-search-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when a policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-checkpoint",
        type=Path,
        default=None,
        help="TD opponent checkpoint used when a policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-temperature",
        type=float,
        default=1.0,
        help="Opponent policy temperature for td-search (lower is greedier).",
    )
    parser.add_argument(
        "--td-search-sample-opponent-actions",
        action="store_true",
        help="Sample opponent rollout actions from opponent model distribution in td-search.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/evals/<utc-timestamp>-<seed-prefix>-<player-a>-vs-<player-b>.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime()
    _validate_policy_args(args)
    policies = {args.player_a_policy.strip().lower(), args.player_b_policy.strip().lower()}
    started_at = time.perf_counter()
    search_config = SearchConfig(
        worlds=args.search_worlds,
        rollouts=args.search_rollouts,
        depth=args.search_depth,
        max_root_actions=args.search_max_root_actions,
        rollout_epsilon=args.search_rollout_epsilon,
    )
    td_value_config = (
        TDValuePolicyConfig(
            checkpoint_path=args.td_value_checkpoint,
            worlds=args.td_worlds,
        )
        if args.td_value_checkpoint is not None
        else None
    )
    td_search_config = (
        TDSearchPolicyConfig(
            value_checkpoint_path=args.td_search_value_checkpoint,
            opponent_checkpoint_path=args.td_search_opponent_checkpoint,
            worlds=args.search_worlds,
            rollouts=args.search_rollouts,
            depth=args.search_depth,
            max_root_actions=args.search_max_root_actions,
            rollout_epsilon=args.search_rollout_epsilon,
            opponent_temperature=args.td_search_opponent_temperature,
            sample_opponent_actions=args.td_search_sample_opponent_actions,
        )
        if "td-search" in policies
        else None
    )

    policy_a = policy_from_name(
        args.player_a_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config,
    )
    policy_b = policy_from_name(
        args.player_b_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)

            def on_progress(
                completed_games: int,
                total_games: int,
                winners: dict,
                _wins_by_policy: dict,
            ) -> None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                player_a_wins = int(winners.get("PlayerA", 0))
                win_rate = (player_a_wins / completed_games) if completed_games > 0 else 0.0
                pct = (completed_games / total_games * 100.0) if total_games > 0 else 100.0
                print(
                    "[eval] progress "
                    f"games={completed_games}/{total_games} "
                    f"pct={pct:.1f}% "
                    f"playerAWinRate={win_rate:.3f} "
                    f"elapsedMin={elapsed_minutes:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            summary = evaluate_matchup(
                env=env,
                policy_player_a=policy_a,
                policy_player_b=policy_b,
                games=args.games,
                seed_prefix=args.seed_prefix,
                progress_every_games=DEFAULT_PROGRESS_EVERY_GAMES,
                on_progress=on_progress,
            )
    finally:
        policy_a.close()
        policy_b.close()

    output_path = args.out or _default_output_path(
        seed_prefix=args.seed_prefix,
        player_a_policy=args.player_a_policy,
        player_b_policy=args.player_b_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.games,
            "seedPrefix": args.seed_prefix,
            "playerAPolicy": args.player_a_policy,
            "playerBPolicy": args.player_b_policy,
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
            "tdValue": {
                "checkpoint": str(args.td_value_checkpoint) if args.td_value_checkpoint else None,
                "worlds": args.td_worlds,
            },
            "tdSearch": {
                "valueCheckpoint": (
                    str(args.td_search_value_checkpoint)
                    if args.td_search_value_checkpoint
                    else None
                ),
                "opponentCheckpoint": (
                    str(args.td_search_opponent_checkpoint)
                    if args.td_search_opponent_checkpoint
                    else None
                ),
                "opponentTemperature": args.td_search_opponent_temperature,
                "sampleOpponentActions": args.td_search_sample_opponent_actions,
            },
        },
        "results": {
            "games": summary.games,
            "winners": summary.winners,
            "winsByPolicy": summary.wins_by_policy,
            "averageTurn": summary.average_turn,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "games": payload["results"]["games"],
                "winners": payload["results"]["winners"],
                "winsByPolicy": payload["results"]["winsByPolicy"],
                "averageTurn": payload["results"]["averageTurn"],
            },
            indent=2,
        )
    )
    return 0


def _default_output_path(seed_prefix: str, player_a_policy: str, player_b_policy: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed_prefix = _slug(seed_prefix)
    safe_player_a = _slug(player_a_policy)
    safe_player_b = _slug(player_b_policy)
    return Path("artifacts/evals") / f"{stamp}-{safe_seed_prefix}-{safe_player_a}-vs-{safe_player_b}.json"


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


def _validate_policy_args(args: argparse.Namespace) -> None:
    policies = {args.player_a_policy.strip().lower(), args.player_b_policy.strip().lower()}
    if "td-value" in policies and args.td_value_checkpoint is None:
        raise SystemExit("--td-value-checkpoint is required when using td-value policy.")
    if "td-search" in policies:
        if args.td_search_value_checkpoint is None:
            raise SystemExit("--td-search-value-checkpoint is required when using td-search policy.")
        if args.td_search_opponent_checkpoint is None:
            raise SystemExit("--td-search-opponent-checkpoint is required when using td-search policy.")


if __name__ == "__main__":
    raise SystemExit(main())
