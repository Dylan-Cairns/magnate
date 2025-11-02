from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.guidance_training import GuidanceConfig, train_guidance_model
from trainer.ppo_model import save_ppo_checkpoint
from trainer.training import read_samples_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a guidance checkpoint (policy prior + value head) from "
            "teacher decision samples."
        )
    )
    parser.add_argument(
        "--samples-in",
        type=Path,
        required=True,
        help="Input JSONL teacher sample file.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("artifacts/search_guidance_checkpoint.pt"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="Weight for value-head MSE loss.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient applied to policy loss.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Model hidden width.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_samples_jsonl(args.samples_in)
    config = GuidanceConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    model, summary = train_guidance_model(samples=samples, config=config)
    save_ppo_checkpoint(
        model=model,
        output_path=args.checkpoint_out,
        metadata={
            "mode": "search-guidance",
            "samplesIn": str(args.samples_in),
            "samples": summary.sample_count,
            "epochs": args.epochs,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "weightDecay": args.weight_decay,
            "valueLossCoef": args.value_loss_coef,
            "entropyCoef": args.entropy_coef,
            "maxGradNorm": args.max_grad_norm,
            "hiddenDim": args.hidden_dim,
            "seed": args.seed,
        },
    )

    final = summary.final
    print(
        json.dumps(
            {
                "samplesIn": str(args.samples_in),
                "checkpointOut": str(args.checkpoint_out),
                "samples": summary.sample_count,
                "observationDim": summary.observation_dim,
                "actionFeatureDim": summary.action_feature_dim,
                "epochs": args.epochs,
                "final": (
                    {
                        "epoch": final.epoch,
                        "policyLoss": final.policy_loss,
                        "valueLoss": final.value_loss,
                        "entropy": final.entropy,
                        "accuracy": final.accuracy,
                    }
                    if final is not None
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
