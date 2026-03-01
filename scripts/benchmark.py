from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from trainer.benchmarking import (
    BENCHMARK_GAMES,
    benchmark_spec_json,
    run_canonical_benchmark,
)
from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import MctsConfig, SearchConfig, policy_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical fixed-holdout Magnate benchmark "
            "(candidate as PlayerA vs random and heuristic)."
        )
    )
    parser.add_argument(
        "--candidate-policy",
        type=str,
        default="bc",
        help="Candidate policy name for PlayerA (random|heuristic|search|mcts|bc|ppo).",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path required when --candidate-policy=bc or ppo.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/benchmarks/<utc-timestamp>-<candidate-policy>.json"
        ),
    )
    parser.add_argument(
        "--search-worlds",
        type=int,
        default=8,
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
        default=16,
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
        default=0.12,
        help="Search rollout epsilon for random exploratory moves.",
    )
    parser.add_argument(
        "--mcts-worlds",
        type=int,
        default=6,
        help="MCTS determinized world samples per decision.",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=96,
        help="MCTS simulations per sampled world.",
    )
    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=24,
        help="MCTS simulation depth limit (decision steps).",
    )
    parser.add_argument(
        "--mcts-max-root-actions",
        type=int,
        default=8,
        help="MCTS root actions kept after heuristic pre-ranking.",
    )
    parser.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.25,
        help="MCTS PUCT exploration coefficient.",
    )
    parser.add_argument(
        "--search-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for search priors/value/opponent model.",
    )
    parser.add_argument(
        "--mcts-guidance-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO-format guidance checkpoint for MCTS priors/value.",
    )
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used by guidance checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    search_config = SearchConfig(
        worlds=args.search_worlds,
        rollouts=args.search_rollouts,
        depth=args.search_depth,
        max_root_actions=args.search_max_root_actions,
        rollout_epsilon=args.search_rollout_epsilon,
    )
    mcts_config = MctsConfig(
        worlds=args.mcts_worlds,
        simulations=args.mcts_simulations,
        depth=args.mcts_depth,
        max_root_actions=args.mcts_max_root_actions,
        c_puct=args.mcts_c_puct,
    )
    candidate = policy_from_name(
        args.candidate_policy,
        checkpoint_path=args.candidate_checkpoint,
        search_config=search_config,
        mcts_config=mcts_config,
        search_guidance_checkpoint=args.search_guidance_checkpoint,
        mcts_guidance_checkpoint=args.mcts_guidance_checkpoint,
        guidance_temperature=args.guidance_temperature,
    )

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)
            summary = run_canonical_benchmark(
                env=env,
                candidate_policy=candidate,
                games_per_matchup=BENCHMARK_GAMES,
            )
    finally:
        candidate.close()

    payload = {
        "candidate": {
            "policy": args.candidate_policy,
            "checkpoint": str(args.candidate_checkpoint) if args.candidate_checkpoint else None,
            "name": candidate.name,
        },
        "benchmarkSpec": benchmark_spec_json(),
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
        "results": summary.to_json(),
    }

    output_path = args.out or _default_output_path(args.candidate_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"artifact": str(output_path), **payload}, indent=2))
    return 0


def _default_output_path(candidate_policy: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_policy = candidate_policy.strip().lower().replace(" ", "-")
    return Path("artifacts/benchmarks") / f"{stamp}-{safe_policy}.json"


if __name__ == "__main__":
    raise SystemExit(main())
