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
from trainer.policies import policy_from_name


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
        help="Candidate policy name for PlayerA (random|heuristic|bc|ppo).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = policy_from_name(
        args.candidate_policy,
        checkpoint_path=args.candidate_checkpoint,
    )

    with BridgeClient() as client:
        env = MagnateBridgeEnv(client=client)
        summary = run_canonical_benchmark(
            env=env,
            candidate_policy=candidate,
            games_per_matchup=BENCHMARK_GAMES,
        )

    payload = {
        "candidate": {
            "policy": args.candidate_policy,
            "checkpoint": str(args.candidate_checkpoint) if args.candidate_checkpoint else None,
            "name": candidate.name,
        },
        "benchmarkSpec": benchmark_spec_json(),
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
