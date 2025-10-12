from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.bridge_client import BridgeClient
from trainer.behavior_cloning import (
    BehaviorCloningConfig,
    save_behavior_cloning_checkpoint,
    train_behavior_cloning,
)
from trainer.env import MagnateBridgeEnv
from trainer.policies import policy_from_name
from trainer.training import collect_training_samples, read_samples_jsonl, write_samples_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Magnate training samples through the TS bridge."
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to sample.")
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="train",
        help="Seed prefix used to derive deterministic per-game seeds.",
    )
    parser.add_argument(
        "--player-a-policy",
        type=str,
        default="heuristic",
        help="Policy name for PlayerA (random|heuristic).",
    )
    parser.add_argument(
        "--player-b-policy",
        type=str,
        default="random",
        help="Policy name for PlayerB (random|heuristic).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/training_samples.jsonl"),
        help="Output JSONL path for decision samples.",
    )
    parser.add_argument(
        "--samples-in",
        type=Path,
        default=None,
        help="Use an existing JSONL sample file instead of collecting new games.",
    )
    parser.add_argument(
        "--skip-bc",
        action="store_true",
        help="Skip behavior-cloning optimization (collect/load samples only).",
    )
    parser.add_argument(
        "--bc-checkpoint-out",
        type=Path,
        default=Path("artifacts/bc_checkpoint.json"),
        help="Output path for the behavior-cloned policy checkpoint.",
    )
    parser.add_argument("--bc-epochs", type=int, default=8, help="Behavior-cloning SGD epochs.")
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=0.05,
        help="Behavior-cloning SGD learning rate.",
    )
    parser.add_argument("--bc-l2", type=float, default=1e-4, help="Behavior-cloning L2 regularization.")
    parser.add_argument("--bc-seed", type=int, default=0, help="Behavior-cloning RNG seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_in is not None:
        samples = read_samples_jsonl(args.samples_in)
        print(f"Loaded {len(samples)} decision samples from {args.samples_in}.")
    else:
        policy_a = policy_from_name(args.player_a_policy)
        policy_b = policy_from_name(args.player_b_policy)
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)
            samples = collect_training_samples(
                env=env,
                policy_player_a=policy_a,
                policy_player_b=policy_b,
                games=args.games,
                seed_prefix=args.seed_prefix,
            )
        write_samples_jsonl(samples, args.out)
        print(f"Collected {len(samples)} decision samples into {args.out}.")

    if args.skip_bc:
        return 0

    config = BehaviorCloningConfig(
        epochs=args.bc_epochs,
        learning_rate=args.bc_learning_rate,
        l2=args.bc_l2,
        seed=args.bc_seed,
    )
    model, summary = train_behavior_cloning(samples, config=config)
    save_behavior_cloning_checkpoint(
        model=model,
        output_path=args.bc_checkpoint_out,
        metadata={
            "seedPrefix": args.seed_prefix,
            "games": args.games,
            "playerAPolicy": args.player_a_policy,
            "playerBPolicy": args.player_b_policy,
            "epochs": args.bc_epochs,
            "learningRate": args.bc_learning_rate,
            "l2": args.bc_l2,
            "seed": args.bc_seed,
            "sampleCount": len(samples),
            "samplesSource": str(args.samples_in) if args.samples_in is not None else str(args.out),
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.bc_checkpoint_out),
                "initialLoss": summary.initial.loss,
                "initialAccuracy": summary.initial.accuracy,
                "finalLoss": summary.final.loss,
                "finalAccuracy": summary.final.accuracy,
                "epochs": len(summary.history),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
