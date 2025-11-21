from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td import (
    OpponentModel,
    OpponentReplayBuffer,
    OpponentTrainConfig,
    TDOpponentTrainer,
    TDTrainConfig,
    TDValueTrainer,
    ValueNet,
    ValueReplayBuffer,
    load_opponent_checkpoint,
    load_value_checkpoint,
    read_opponent_samples_jsonl,
    read_value_transitions_jsonl,
    save_opponent_checkpoint,
    save_value_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD value/opponent models from replay JSONL artifacts."
    )
    parser.add_argument(
        "--value-replay",
        type=Path,
        default=None,
        help="Value-transition replay JSONL path.",
    )
    parser.add_argument(
        "--opponent-replay",
        type=Path,
        default=None,
        help="Opponent-sample replay JSONL path.",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of train steps.")
    parser.add_argument("--value-batch-size", type=int, default=128)
    parser.add_argument("--opponent-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight-decay", type=float, default=1e-5)
    parser.add_argument("--opponent-learning-rate", type=float, default=3e-4)
    parser.add_argument("--opponent-weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-sync-interval", type=int, default=200)
    parser.add_argument(
        "--use-mse-loss",
        action="store_true",
        help="Use MSE for value loss (default is Huber).",
    )
    parser.add_argument(
        "--disable-value",
        action="store_true",
        help="Disable value-model training.",
    )
    parser.add_argument(
        "--disable-opponent",
        action="store_true",
        help="Disable opponent-model training.",
    )
    parser.add_argument(
        "--warm-start-value-checkpoint",
        type=Path,
        default=None,
        help="Optional value checkpoint to initialize the value model.",
    )
    parser.add_argument(
        "--warm-start-opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional opponent checkpoint to initialize the opponent model.",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=200,
        help="Checkpoint cadence in train steps.",
    )
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=20,
        help="Print progress every N steps (0 disables).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/td_checkpoints"),
        help="Base output directory.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="td-train",
        help="Run label used in output directory naming.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional explicit summary path (defaults inside run directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)

    rng = torch.Generator().manual_seed(args.seed)
    del rng  # Reserved for future direct torch sampling paths.

    started_at = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = args.out_dir / f"{stamp}-{_slug(args.run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_out = args.summary_out or (run_dir / "summary.json")

    train_value = not args.disable_value
    train_opponent = not args.disable_opponent

    value_replay = ValueReplayBuffer(capacity=max(1, 10_000_000))
    opponent_replay = OpponentReplayBuffer(capacity=max(1, 10_000_000))

    if train_value:
        transitions = read_value_transitions_jsonl(args.value_replay)
        value_replay.extend(transitions)
        if len(value_replay) == 0:
            raise SystemExit(f"value replay is empty: {args.value_replay}")
    if train_opponent:
        samples = read_opponent_samples_jsonl(args.opponent_replay)
        opponent_replay.extend(samples)
        if len(opponent_replay) == 0:
            raise SystemExit(f"opponent replay is empty: {args.opponent_replay}")

    value_model = None
    value_target_model = None
    value_optimizer = None
    value_trainer = None
    if train_value:
        if args.warm_start_value_checkpoint is not None:
            value_model, _ = load_value_checkpoint(path=args.warm_start_value_checkpoint)
        else:
            value_model = ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=args.hidden_dim)
        value_target_model = ValueNet(
            observation_dim=value_model.observation_dim,
            hidden_dim=value_model.hidden_dim,
        )
        value_optimizer = torch.optim.Adam(
            value_model.parameters(),
            lr=args.value_learning_rate,
            weight_decay=args.value_weight_decay,
        )
        value_trainer = TDValueTrainer(
            model=value_model,
            target_model=value_target_model,
            optimizer=value_optimizer,
            config=TDTrainConfig(
                gamma=args.gamma,
                learning_rate=args.value_learning_rate,
                weight_decay=args.value_weight_decay,
                max_grad_norm=args.max_grad_norm,
                target_sync_interval=args.target_sync_interval,
                use_huber_loss=(not args.use_mse_loss),
            ),
        )

    opponent_model = None
    opponent_optimizer = None
    opponent_trainer = None
    if train_opponent:
        if args.warm_start_opponent_checkpoint is not None:
            opponent_model, _ = load_opponent_checkpoint(path=args.warm_start_opponent_checkpoint)
        else:
            opponent_model = OpponentModel(
                observation_dim=OBSERVATION_DIM,
                action_feature_dim=ACTION_FEATURE_DIM,
                hidden_dim=args.hidden_dim,
            )
        opponent_optimizer = torch.optim.Adam(
            opponent_model.parameters(),
            lr=args.opponent_learning_rate,
            weight_decay=args.opponent_weight_decay,
        )
        opponent_trainer = TDOpponentTrainer(
            model=opponent_model,
            optimizer=opponent_optimizer,
            config=OpponentTrainConfig(
                learning_rate=args.opponent_learning_rate,
                weight_decay=args.opponent_weight_decay,
                max_grad_norm=args.max_grad_norm,
            ),
        )

    checkpoint_paths: List[Dict[str, Any]] = []
    latest_value_summary: Dict[str, Any] | None = None
    latest_opponent_summary: Dict[str, Any] | None = None

    import random

    py_rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        if train_value and value_trainer is not None:
            transitions = value_replay.sample(
                batch_size=args.value_batch_size,
                rng=py_rng,
            )
            value_summary = value_trainer.train_batch(transitions=transitions)
            latest_value_summary = {
                "step": value_summary.step,
                "loss": value_summary.loss,
                "predictionMean": value_summary.prediction_mean,
                "targetMean": value_summary.target_mean,
                "targetSynced": value_summary.target_synced,
            }
        else:
            value_summary = None

        if train_opponent and opponent_trainer is not None:
            batch = opponent_replay.sample(
                batch_size=args.opponent_batch_size,
                rng=py_rng,
            )
            opponent_summary = opponent_trainer.train_batch(samples=batch)
            latest_opponent_summary = {
                "step": opponent_summary.step,
                "loss": opponent_summary.loss,
                "accuracy": opponent_summary.accuracy,
            }
        else:
            opponent_summary = None

        if (
            args.progress_every_steps > 0
            and (step % args.progress_every_steps == 0 or step == args.steps)
        ):
            elapsed_minutes = (time.perf_counter() - started_at) / 60.0
            print(
                "[td-train] progress "
                f"step={step}/{args.steps} "
                f"valueLoss={_metric_or_na(value_summary.loss if value_summary else None)} "
                f"opponentLoss={_metric_or_na(opponent_summary.loss if opponent_summary else None)} "
                f"opponentAcc={_metric_or_na(opponent_summary.accuracy if opponent_summary else None)} "
                f"elapsedMin={elapsed_minutes:.1f}",
                file=sys.stderr,
                flush=True,
            )

        should_save = step == args.steps
        if args.save_every_steps > 0 and step % args.save_every_steps == 0:
            should_save = True
        if should_save:
            saved = _save_step_checkpoints(
                step=step,
                run_dir=run_dir,
                value_model=value_model,
                value_optimizer=value_optimizer,
                opponent_model=opponent_model,
                opponent_optimizer=opponent_optimizer,
            )
            checkpoint_paths.append(saved)

    summary_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "steps": args.steps,
            "seed": args.seed,
            "trainValue": train_value,
            "trainOpponent": train_opponent,
            "valueReplay": str(args.value_replay) if args.value_replay else None,
            "opponentReplay": str(args.opponent_replay) if args.opponent_replay else None,
            "valueBatchSize": args.value_batch_size,
            "opponentBatchSize": args.opponent_batch_size,
            "hiddenDim": args.hidden_dim,
            "gamma": args.gamma,
            "valueLearningRate": args.value_learning_rate,
            "valueWeightDecay": args.value_weight_decay,
            "opponentLearningRate": args.opponent_learning_rate,
            "opponentWeightDecay": args.opponent_weight_decay,
            "maxGradNorm": args.max_grad_norm,
            "targetSyncInterval": args.target_sync_interval,
            "valueLoss": "mse" if args.use_mse_loss else "huber",
            "saveEverySteps": args.save_every_steps,
            "progressEverySteps": args.progress_every_steps,
            "warmStartValueCheckpoint": (
                str(args.warm_start_value_checkpoint)
                if args.warm_start_value_checkpoint
                else None
            ),
            "warmStartOpponentCheckpoint": (
                str(args.warm_start_opponent_checkpoint)
                if args.warm_start_opponent_checkpoint
                else None
            ),
        },
        "results": {
            "valueReplaySize": len(value_replay),
            "opponentReplaySize": len(opponent_replay),
            "latestValue": latest_value_summary,
            "latestOpponent": latest_opponent_summary,
            "checkpoints": checkpoint_paths,
            "elapsedSeconds": time.perf_counter() - started_at,
        },
        "artifacts": {
            "runDir": str(run_dir),
            "summary": str(summary_out),
        },
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "runDir": str(run_dir),
                "summaryArtifact": str(summary_out),
                "valueReplaySize": len(value_replay),
                "opponentReplaySize": len(opponent_replay),
                "latestValue": latest_value_summary,
                "latestOpponent": latest_opponent_summary,
                "checkpointsSaved": len(checkpoint_paths),
            },
            indent=2,
        )
    )
    return 0


def _save_step_checkpoints(
    *,
    step: int,
    run_dir: Path,
    value_model: ValueNet | None,
    value_optimizer: torch.optim.Optimizer | None,
    opponent_model: OpponentModel | None,
    opponent_optimizer: torch.optim.Optimizer | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"step": step}
    step_label = f"{step:07d}"
    if value_model is not None:
        value_path = run_dir / f"value-step-{step_label}.pt"
        save_value_checkpoint(
            model=value_model,
            output_path=value_path,
            metadata={"step": step},
            optimizer_state_dict=value_optimizer.state_dict() if value_optimizer else None,
        )
        payload["value"] = str(value_path)
    if opponent_model is not None:
        opponent_path = run_dir / f"opponent-step-{step_label}.pt"
        save_opponent_checkpoint(
            model=opponent_model,
            output_path=opponent_path,
            metadata={"step": step},
            optimizer_state_dict=(
                opponent_optimizer.state_dict() if opponent_optimizer else None
            ),
        )
        payload["opponent"] = str(opponent_path)
    return payload


def _validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0.")
    if args.value_batch_size <= 0:
        raise SystemExit("--value-batch-size must be > 0.")
    if args.opponent_batch_size <= 0:
        raise SystemExit("--opponent-batch-size must be > 0.")
    if args.hidden_dim <= 0:
        raise SystemExit("--hidden-dim must be > 0.")
    if args.max_grad_norm <= 0.0:
        raise SystemExit("--max-grad-norm must be > 0.")
    if args.target_sync_interval <= 0:
        raise SystemExit("--target-sync-interval must be > 0.")
    if args.save_every_steps < 0:
        raise SystemExit("--save-every-steps must be >= 0.")
    if args.progress_every_steps < 0:
        raise SystemExit("--progress-every-steps must be >= 0.")

    train_value = not args.disable_value
    train_opponent = not args.disable_opponent
    if not train_value and not train_opponent:
        raise SystemExit("At least one of value/opponent training must be enabled.")

    if train_value:
        if args.value_replay is None:
            raise SystemExit("--value-replay is required unless --disable-value is set.")
        if not args.value_replay.exists():
            raise SystemExit(f"value replay not found: {args.value_replay}")
    if train_opponent:
        if args.opponent_replay is None:
            raise SystemExit("--opponent-replay is required unless --disable-opponent is set.")
        if not args.opponent_replay.exists():
            raise SystemExit(f"opponent replay not found: {args.opponent_replay}")

    if args.warm_start_value_checkpoint is not None and not args.warm_start_value_checkpoint.exists():
        raise SystemExit(f"warm-start value checkpoint not found: {args.warm_start_value_checkpoint}")
    if args.warm_start_opponent_checkpoint is not None and not args.warm_start_opponent_checkpoint.exists():
        raise SystemExit(
            f"warm-start opponent checkpoint not found: {args.warm_start_opponent_checkpoint}"
        )


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


def _metric_or_na(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
