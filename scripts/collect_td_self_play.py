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
from trainer.policies import (
    SearchConfig,
    TDSearchPolicyConfig,
    TDValuePolicyConfig,
    policy_from_name,
)
from trainer.td import (
    collect_self_play_games,
    flatten_opponent_samples,
    flatten_value_transitions,
    write_opponent_samples_jsonl,
    write_value_transitions_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TD self-play replay artifacts (value transitions + opponent samples)."
    )
    parser.add_argument("--games", type=int, default=200, help="Number of self-play games.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="td-self-play",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--player-a-policy",
        type=str,
        required=True,
        help="Policy for PlayerA (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        required=True,
        help="Policy for PlayerB (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
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
        "--out-dir",
        type=Path,
        default=Path("artifacts/td_replay"),
        help="Output directory for replay artifacts.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="td-self-play",
        help="Run label used in default artifact names.",
    )
    parser.add_argument(
        "--value-out",
        type=Path,
        default=None,
        help="Optional explicit value-transition JSONL path.",
    )
    parser.add_argument(
        "--opponent-out",
        type=Path,
        default=None,
        help="Optional explicit opponent-sample JSONL path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit summary JSON path.",
    )
    parser.add_argument(
        "--progress-every-games",
        type=int,
        default=25,
        help="Print progress every N completed games (0 disables).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime()
    _validate_policy_args(args)
    policies = {args.player_a_policy.strip().lower(), args.player_b_policy.strip().lower()}
    if args.games <= 0:
        raise SystemExit("--games must be > 0.")

    started_at = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_label = _slug(args.run_label)

    value_out = args.value_out or (args.out_dir / f"{stamp}-{safe_label}.value.jsonl")
    opponent_out = args.opponent_out or (args.out_dir / f"{stamp}-{safe_label}.opponent.jsonl")
    summary_out = args.summary_out or (args.out_dir / f"{stamp}-{safe_label}.summary.json")

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

            def on_progress(completed_games: int, total_games: int, winners: dict) -> None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                pct = ((completed_games / total_games) * 100.0) if total_games > 0 else 100.0
                print(
                    "[td-self-play] progress "
                    f"games={completed_games}/{total_games} "
                    f"pct={pct:.1f}% "
                    f"winsA={int(winners.get('PlayerA', 0))} "
                    f"winsB={int(winners.get('PlayerB', 0))} "
                    f"draw={int(winners.get('Draw', 0))} "
                    f"elapsedMin={elapsed_minutes:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            episodes = collect_self_play_games(
                env=env,
                policy_player_a=policy_a,
                policy_player_b=policy_b,
                games=args.games,
                seed_prefix=args.seed_prefix,
                progress_every_games=args.progress_every_games,
                on_progress=on_progress if args.progress_every_games > 0 else None,
            )
    finally:
        _close_unique(policy_a, policy_b)

    value_transitions = flatten_value_transitions(episodes)
    opponent_samples = flatten_opponent_samples(episodes)
    write_value_transitions_jsonl(value_transitions, value_out)
    write_opponent_samples_jsonl(opponent_samples, opponent_out)

    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    turn_total = 0
    for episode in episodes:
        winners[episode.winner] += 1
        turn_total += int(episode.turns)

    summary_payload = {
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
            "games": len(episodes),
            "winners": winners,
            "averageTurn": (turn_total / float(len(episodes))) if episodes else 0.0,
            "valueTransitions": len(value_transitions),
            "opponentSamples": len(opponent_samples),
        },
        "artifacts": {
            "valueTransitions": str(value_out),
            "opponentSamples": str(opponent_out),
            "summary": str(summary_out),
        },
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "valueTransitionsArtifact": str(value_out),
                "opponentSamplesArtifact": str(opponent_out),
                "summaryArtifact": str(summary_out),
                "games": summary_payload["results"]["games"],
                "winners": summary_payload["results"]["winners"],
                "averageTurn": summary_payload["results"]["averageTurn"],
                "valueTransitions": summary_payload["results"]["valueTransitions"],
                "opponentSamples": summary_payload["results"]["opponentSamples"],
            },
            indent=2,
        )
    )
    return 0


def _close_unique(*policies) -> None:
    seen: set[int] = set()
    for policy in policies:
        if policy is None:
            continue
        key = id(policy)
        if key in seen:
            continue
        seen.add(key)
        policy.close()


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
