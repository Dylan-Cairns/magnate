from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import SearchConfig, policy_from_name
from trainer.teacher_data import collect_teacher_samples
from trainer.training import write_samples_jsonl
from trainer.types import PlayerId


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate teacher-labeled decision samples for policy distillation "
            "using bridge-driven self-play."
        )
    )
    parser.add_argument("--games", type=int, default=200, help="Number of games.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="teacher-data",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--teacher-policy",
        type=str,
        default="search",
        help="Teacher policy (random|heuristic|search|bc|ppo).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --teacher-policy=bc or ppo.",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy for non-teacher turns (random|heuristic|search|bc|ppo).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --opponent-policy=bc or ppo.",
    )
    parser.add_argument(
        "--teacher-players",
        type=str,
        default="both",
        choices=("both", "player-a", "player-b"),
        help="Which players use the teacher policy for label generation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSONL samples path. "
            "Default: artifacts/teacher_data/<utc-timestamp>-<seed-prefix>.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON summary path (defaults alongside --out).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="RNG seed for policy sampling where policies are stochastic.",
    )
    parser.add_argument(
        "--progress-every-games",
        type=int,
        default=25,
        help="Print progress every N completed games (0 disables).",
    )
    parser.add_argument(
        "--search-worlds",
        type=int,
        default=8,
        help="Search world samples per decision.",
    )
    parser.add_argument(
        "--search-rollouts",
        type=int,
        default=1,
        help="Search rollouts per action per sampled world.",
    )
    parser.add_argument(
        "--search-depth",
        type=int,
        default=16,
        help="Search rollout depth limit.",
    )
    parser.add_argument(
        "--search-max-root-actions",
        type=int,
        default=6,
        help="Search candidate root actions kept after heuristic pre-ranking.",
    )
    parser.add_argument(
        "--search-rollout-epsilon",
        type=float,
        default=0.12,
        help="Search rollout epsilon for random exploratory moves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    teacher_player_ids = _parse_teacher_players(args.teacher_players)
    search_config = SearchConfig(
        worlds=args.search_worlds,
        rollouts=args.search_rollouts,
        depth=args.search_depth,
        max_root_actions=args.search_max_root_actions,
        rollout_epsilon=args.search_rollout_epsilon,
    )

    teacher_policy = policy_from_name(
        args.teacher_policy,
        checkpoint_path=args.teacher_checkpoint,
        search_config=search_config,
    )
    opponent_policy = None
    if teacher_player_ids != {"PlayerA", "PlayerB"}:
        opponent_policy = policy_from_name(
            args.opponent_policy,
            checkpoint_path=args.opponent_checkpoint,
            search_config=search_config,
        )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)

            def on_progress(
                completed_games: int,
                total_games: int,
                sample_count: int,
                winners: dict,
                decisions_by_player: dict,
            ) -> None:
                elapsed_minutes = (time.perf_counter() - started_at) / 60.0
                pct = (
                    (completed_games / total_games) * 100.0 if total_games > 0 else 100.0
                )
                print(
                    "[teacher-data] progress "
                    f"games={completed_games}/{total_games} "
                    f"pct={pct:.1f}% "
                    f"samples={sample_count} "
                    f"winsA={int(winners.get('PlayerA', 0))} "
                    f"winsB={int(winners.get('PlayerB', 0))} "
                    f"draw={int(winners.get('Draw', 0))} "
                    f"decisionsA={int(decisions_by_player.get('PlayerA', 0))} "
                    f"decisionsB={int(decisions_by_player.get('PlayerB', 0))} "
                    f"elapsedMin={elapsed_minutes:.1f}",
                    file=sys.stderr,
                    flush=True,
                )

            samples, summary = collect_teacher_samples(
                env=env,
                teacher_policy=teacher_policy,
                opponent_policy=opponent_policy,
                games=args.games,
                seed_prefix=args.seed_prefix,
                teacher_player_ids=teacher_player_ids,
                rng_seed=args.rng_seed,
                progress_every_games=args.progress_every_games,
                on_progress=on_progress if args.progress_every_games > 0 else None,
            )
    finally:
        _close_unique(teacher_policy, opponent_policy)

    output_path = args.out or _default_samples_path(args.seed_prefix)
    write_samples_jsonl(samples, output_path)

    summary_path = args.summary_out or _default_summary_path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.games,
            "seedPrefix": args.seed_prefix,
            "teacherPolicy": args.teacher_policy,
            "teacherCheckpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            "teacherPlayers": sorted(teacher_player_ids),
            "opponentPolicy": args.opponent_policy if opponent_policy is not None else None,
            "opponentCheckpoint": (
                str(args.opponent_checkpoint) if args.opponent_checkpoint else None
            ),
            "rngSeed": args.rng_seed,
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
        },
        "results": summary.as_json(),
        "artifacts": {
            "samples": str(output_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "samplesArtifact": str(output_path),
                "summaryArtifact": str(summary_path),
                "games": summary.games,
                "samples": summary.samples,
                "winners": summary.winners,
                "decisionsByPlayer": summary.decisions_by_player,
                "averageTurn": summary.average_turn,
            },
            indent=2,
        )
    )
    return 0


def _parse_teacher_players(value: str) -> Set[PlayerId]:
    normalized = value.strip().lower()
    if normalized == "both":
        return {"PlayerA", "PlayerB"}
    if normalized == "player-a":
        return {"PlayerA"}
    if normalized == "player-b":
        return {"PlayerB"}
    raise ValueError(f"Unsupported teacher player mode: {value!r}")


def _close_unique(*policies) -> None:
    seen: set[int] = set()
    for policy in policies:
        if policy is None:
            continue
        policy_id = id(policy)
        if policy_id in seen:
            continue
        seen.add(policy_id)
        policy.close()


def _default_samples_path(seed_prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed_prefix = _slug(seed_prefix)
    return Path("artifacts/teacher_data") / f"{stamp}-{safe_seed_prefix}.jsonl"


def _default_summary_path(samples_path: Path) -> Path:
    base = samples_path.with_suffix("")
    return base.parent / f"{base.name}.summary.json"


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
