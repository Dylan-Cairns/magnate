from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td import (
    DISTRICT_AUGMENTATION_NONE,
    DISTRICT_AUGMENTATION_S4,
    TD_VALUE_TARGET_MODE,
    TD_VALUE_TARGET_MODE_TD_LAMBDA,
    TD_VALUE_TARGET_MODES,
    OpponentModel,
    OpponentReplayBuffer,
    OpponentTrainConfig,
    TDOpponentTrainer,
    TDTrainConfig,
    TDValueTrainer,
    ValueNet,
    ValueReplayBuffer,
    augment_opponent_training_batch,
    augment_value_training_batch,
    build_value_sequence_index,
    derive_augmentation_stream_seed,
    load_opponent_checkpoint,
    load_value_checkpoint,
    named_files_content_sha256,
    read_opponent_samples_jsonl_many,
    read_value_transitions_jsonl_many,
    replay_content_sha256,
    save_opponent_checkpoint,
    save_value_checkpoint,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TD value/opponent models from replay JSONL artifacts."
    )
    parser.add_argument(
        "--value-replay",
        type=Path,
        nargs="+",
        default=None,
        help="Value-transition replay JSONL path(s).",
    )
    parser.add_argument(
        "--value-replay-list",
        type=Path,
        default=None,
        help="Optional UTF-8 text file containing one value replay path per line.",
    )
    parser.add_argument(
        "--opponent-replay",
        type=Path,
        nargs="+",
        default=None,
        help="Opponent-sample replay JSONL path(s).",
    )
    parser.add_argument(
        "--opponent-replay-list",
        type=Path,
        default=None,
        help="Optional UTF-8 text file containing one opponent replay path per line.",
    )
    parser.add_argument(
        "--expected-value-replay-content-sha256",
        type=str,
        default=None,
        help="Optional frozen content fingerprint for the ordered value replay files.",
    )
    parser.add_argument(
        "--expected-opponent-replay-content-sha256",
        type=str,
        default=None,
        help="Optional frozen content fingerprint for the ordered opponent replay files.",
    )
    parser.add_argument(
        "--value-replay-max-lines",
        type=int,
        default=0,
        help="Optional tail cap across value replay files; 0 reads every line.",
    )
    parser.add_argument(
        "--opponent-replay-max-lines",
        type=int,
        default=0,
        help="Optional tail cap across opponent replay files; 0 reads every line.",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Number of train steps.")
    parser.add_argument("--value-batch-size", type=int, default=128)
    parser.add_argument("--opponent-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--district-augmentation",
        choices=("none", "s4"),
        default="none",
        help="Optional exact D1/D2/D4/D5 permutation augmentation; D3 remains fixed.",
    )
    parser.add_argument(
        "--district-augmentation-seed",
        type=int,
        default=None,
        help="Required independent RNG seed when --district-augmentation=s4.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight-decay", type=float, default=1e-5)
    parser.add_argument("--opponent-learning-rate", type=float, default=3e-4)
    parser.add_argument("--opponent-weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--target-sync-interval", type=int, default=200)
    parser.add_argument(
        "--value-target-mode",
        type=str,
        choices=sorted(TD_VALUE_TARGET_MODES),
        default=TD_VALUE_TARGET_MODE,
        help="Value target mode: td0 or td-lambda.",
    )
    parser.add_argument(
        "--td-lambda",
        type=float,
        default=0.7,
        help="Lambda used when --value-target-mode=td-lambda.",
    )
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
        "--expected-warm-start-value-sha256",
        type=str,
        default=None,
        help="Optional frozen SHA-256 for --warm-start-value-checkpoint.",
    )
    parser.add_argument(
        "--expected-warm-start-opponent-sha256",
        type=str,
        default=None,
        help="Optional frozen SHA-256 for --warm-start-opponent-checkpoint.",
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=None,
        help="Optional frozen experiment manifest used for provenance verification.",
    )
    parser.add_argument(
        "--expected-experiment-manifest-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 of --experiment-manifest.",
    )
    parser.add_argument(
        "--provenance-repo-root",
        type=Path,
        default=None,
        help="Repository root used to resolve implementation files from the manifest.",
    )
    parser.add_argument(
        "--expected-implementation-sha256",
        type=str,
        default=None,
        help="Expected aggregate content fingerprint of manifest implementation files.",
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Optional torch intra-op CPU thread count.",
    )
    parser.add_argument(
        "--num-interop-threads",
        type=int,
        default=None,
        help="Optional torch inter-op CPU thread count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime()
    args.value_replay = _merge_replay_paths(
        inline_paths=args.value_replay,
        list_path=args.value_replay_list,
        label="value",
    )
    args.opponent_replay = _merge_replay_paths(
        inline_paths=args.opponent_replay,
        list_path=args.opponent_replay_list,
        label="opponent",
    )
    _validate_args(args)
    provenance = _resolve_training_provenance(args)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if args.num_interop_threads is not None:
        torch.set_num_interop_threads(args.num_interop_threads)

    rng = torch.Generator().manual_seed(args.seed)
    del rng  # Reserved for future direct torch sampling paths.

    district_augmentation_mode = (
        DISTRICT_AUGMENTATION_S4
        if args.district_augmentation == "s4"
        else DISTRICT_AUGMENTATION_NONE
    )
    value_augmentation_stream_seed = (
        derive_augmentation_stream_seed(
            base_seed=args.district_augmentation_seed,
            stream="value",
        )
        if args.district_augmentation_seed is not None
        else None
    )
    opponent_augmentation_stream_seed = (
        derive_augmentation_stream_seed(
            base_seed=args.district_augmentation_seed,
            stream="opponent",
        )
        if args.district_augmentation_seed is not None
        else None
    )
    value_augmentation_rng = (
        random.Random(value_augmentation_stream_seed)
        if district_augmentation_mode == DISTRICT_AUGMENTATION_S4
        else None
    )
    opponent_augmentation_rng = (
        random.Random(opponent_augmentation_stream_seed)
        if district_augmentation_mode == DISTRICT_AUGMENTATION_S4
        else None
    )

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
        print(
            "[td-train] loading value replay "
            f"files={len(args.value_replay)} maxLines={args.value_replay_max_lines}",
            file=sys.stderr,
            flush=True,
        )
        transitions = read_value_transitions_jsonl_many(
            args.value_replay,
            max_transitions=args.value_replay_max_lines,
        )
        value_replay.extend(transitions)
        if len(value_replay) == 0:
            raise SystemExit(f"value replay is empty: {args.value_replay}")
        print(
            f"[td-train] loaded value replay rows={len(value_replay)}",
            file=sys.stderr,
            flush=True,
        )
    if train_opponent:
        print(
            "[td-train] loading opponent replay "
            f"files={len(args.opponent_replay)} maxLines={args.opponent_replay_max_lines}",
            file=sys.stderr,
            flush=True,
        )
        samples = read_opponent_samples_jsonl_many(
            args.opponent_replay,
            max_samples=args.opponent_replay_max_lines,
        )
        opponent_replay.extend(samples)
        if len(opponent_replay) == 0:
            raise SystemExit(f"opponent replay is empty: {args.opponent_replay}")
        print(
            f"[td-train] loaded opponent replay rows={len(opponent_replay)}",
            file=sys.stderr,
            flush=True,
        )

    value_model = None
    value_target_model = None
    value_optimizer = None
    value_trainer = None
    value_sequence_index = None
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
                value_target_mode=args.value_target_mode,
                td_lambda=args.td_lambda,
            ),
        )
        if args.value_target_mode == TD_VALUE_TARGET_MODE_TD_LAMBDA:
            print(
                "[td-train] building value sequence index for td-lambda",
                file=sys.stderr,
                flush=True,
            )
            value_sequence_index = build_value_sequence_index(
                transitions=value_replay.as_list(),
            )
            print(
                f"[td-train] built value sequences={len(value_sequence_index)}",
                file=sys.stderr,
                flush=True,
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

    py_rng = random.Random(args.seed)
    value_sampling_trace = hashlib.sha256()
    opponent_sampling_trace = hashlib.sha256()
    for step in range(1, args.steps + 1):
        if train_value and value_trainer is not None:
            value_indices, transitions = value_replay.sample_with_indices(
                batch_size=args.value_batch_size,
                rng=py_rng,
            )
            _update_sampling_trace(
                digest=value_sampling_trace,
                step=step,
                indices=value_indices,
            )
            augmented_value_batch = augment_value_training_batch(
                mode=district_augmentation_mode,
                transitions=transitions,
                sequence_index=(
                    value_sequence_index
                    if args.value_target_mode == TD_VALUE_TARGET_MODE_TD_LAMBDA
                    else None
                ),
                rng=value_augmentation_rng,
            )
            if args.value_target_mode == TD_VALUE_TARGET_MODE_TD_LAMBDA:
                value_summary = value_trainer.train_batch(
                    transitions=augmented_value_batch.transitions,
                    sequence_index=augmented_value_batch.sequence_index,
                )
            else:
                value_summary = value_trainer.train_batch(
                    transitions=augmented_value_batch.transitions
                )
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
            opponent_indices, batch = opponent_replay.sample_with_indices(
                batch_size=args.opponent_batch_size,
                rng=py_rng,
            )
            _update_sampling_trace(
                digest=opponent_sampling_trace,
                step=step,
                indices=opponent_indices,
            )
            augmented_opponent_batch = augment_opponent_training_batch(
                mode=district_augmentation_mode,
                samples=batch,
                rng=opponent_augmentation_rng,
            )
            opponent_summary = opponent_trainer.train_batch(samples=augmented_opponent_batch)
            latest_opponent_summary = {
                "step": opponent_summary.step,
                "loss": opponent_summary.loss,
                "accuracy": opponent_summary.accuracy,
            }
        else:
            opponent_summary = None

        if args.progress_every_steps > 0 and (
            step % args.progress_every_steps == 0 or step == args.steps
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
                training_metadata={
                    "runLabel": args.run_label,
                    "trainingSeed": args.seed,
                    "districtAugmentation": district_augmentation_mode,
                    "districtAugmentationSeed": args.district_augmentation_seed,
                    "valueAugmentationStreamSeed": value_augmentation_stream_seed,
                    "opponentAugmentationStreamSeed": opponent_augmentation_stream_seed,
                    "valueSamplingTraceSha256": (
                        value_sampling_trace.hexdigest() if train_value else None
                    ),
                    "opponentSamplingTraceSha256": (
                        opponent_sampling_trace.hexdigest() if train_opponent else None
                    ),
                    "provenance": provenance,
                },
            )
            checkpoint_paths.append(saved)

    summary_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "runtime": {
            "pythonVersion": sys.version,
            "torchVersion": torch.__version__,
        },
        "config": {
            "steps": args.steps,
            "seed": args.seed,
            "districtAugmentation": district_augmentation_mode,
            "districtAugmentationSeed": args.district_augmentation_seed,
            "valueAugmentationStreamSeed": value_augmentation_stream_seed,
            "opponentAugmentationStreamSeed": opponent_augmentation_stream_seed,
            "trainValue": train_value,
            "trainOpponent": train_opponent,
            "valueReplay": (
                str(args.value_replay[0])
                if args.value_replay is not None and len(args.value_replay) == 1
                else None
            ),
            "opponentReplay": (
                str(args.opponent_replay[0])
                if args.opponent_replay is not None and len(args.opponent_replay) == 1
                else None
            ),
            "valueReplayFiles": (
                [str(path) for path in args.value_replay] if args.value_replay is not None else None
            ),
            "opponentReplayFiles": (
                [str(path) for path in args.opponent_replay]
                if args.opponent_replay is not None
                else None
            ),
            "valueReplayMaxLines": args.value_replay_max_lines,
            "opponentReplayMaxLines": args.opponent_replay_max_lines,
            "valueReplayList": (
                str(args.value_replay_list) if args.value_replay_list is not None else None
            ),
            "opponentReplayList": (
                str(args.opponent_replay_list) if args.opponent_replay_list is not None else None
            ),
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
            "valueTargetMode": args.value_target_mode,
            "tdLambda": args.td_lambda,
            "valueLoss": "mse" if args.use_mse_loss else "huber",
            "saveEverySteps": args.save_every_steps,
            "progressEverySteps": args.progress_every_steps,
            "warmStartValueCheckpoint": (
                str(args.warm_start_value_checkpoint) if args.warm_start_value_checkpoint else None
            ),
            "warmStartOpponentCheckpoint": (
                str(args.warm_start_opponent_checkpoint)
                if args.warm_start_opponent_checkpoint
                else None
            ),
            "numThreads": args.num_threads,
            "numInteropThreads": args.num_interop_threads,
        },
        "results": {
            "valueReplaySize": len(value_replay),
            "opponentReplaySize": len(opponent_replay),
            "latestValue": latest_value_summary,
            "latestOpponent": latest_opponent_summary,
            "checkpoints": checkpoint_paths,
            "valueSamplingTraceSha256": (value_sampling_trace.hexdigest() if train_value else None),
            "opponentSamplingTraceSha256": (
                opponent_sampling_trace.hexdigest() if train_opponent else None
            ),
            "elapsedMinutes": (time.perf_counter() - started_at) / 60.0,
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
    training_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"step": step}
    step_label = f"{step:07d}"
    if value_model is not None:
        value_path = run_dir / f"value-step-{step_label}.pt"
        save_value_checkpoint(
            model=value_model,
            output_path=value_path,
            metadata={**training_metadata, "step": step},
            optimizer_state_dict=value_optimizer.state_dict() if value_optimizer else None,
        )
        payload["value"] = str(value_path)
    if opponent_model is not None:
        opponent_path = run_dir / f"opponent-step-{step_label}.pt"
        save_opponent_checkpoint(
            model=opponent_model,
            output_path=opponent_path,
            metadata={**training_metadata, "step": step},
            optimizer_state_dict=(opponent_optimizer.state_dict() if opponent_optimizer else None),
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
    if args.value_target_mode not in TD_VALUE_TARGET_MODES:
        raise SystemExit(f"--value-target-mode must be one of {sorted(TD_VALUE_TARGET_MODES)}.")
    if args.td_lambda < 0.0 or args.td_lambda > 1.0:
        raise SystemExit("--td-lambda must be in [0, 1].")
    if args.save_every_steps < 0:
        raise SystemExit("--save-every-steps must be >= 0.")
    if args.progress_every_steps < 0:
        raise SystemExit("--progress-every-steps must be >= 0.")
    if args.value_replay_max_lines < 0:
        raise SystemExit("--value-replay-max-lines must be >= 0.")
    if args.opponent_replay_max_lines < 0:
        raise SystemExit("--opponent-replay-max-lines must be >= 0.")
    if args.num_threads is not None and args.num_threads <= 0:
        raise SystemExit("--num-threads must be > 0 when provided.")
    if args.num_interop_threads is not None and args.num_interop_threads <= 0:
        raise SystemExit("--num-interop-threads must be > 0 when provided.")
    if args.district_augmentation == "s4" and args.district_augmentation_seed is None:
        raise SystemExit(
            "--district-augmentation-seed is required when --district-augmentation=s4."
        )
    if args.value_target_mode == TD_VALUE_TARGET_MODE_TD_LAMBDA and args.value_replay_max_lines > 0:
        raise SystemExit(
            "--value-replay-max-lines is not supported with --value-target-mode td-lambda "
            "because raw line caps can split episode trajectories."
        )

    sha_arguments = (
        ("--expected-value-replay-content-sha256", args.expected_value_replay_content_sha256),
        (
            "--expected-opponent-replay-content-sha256",
            args.expected_opponent_replay_content_sha256,
        ),
        ("--expected-warm-start-value-sha256", args.expected_warm_start_value_sha256),
        (
            "--expected-warm-start-opponent-sha256",
            args.expected_warm_start_opponent_sha256,
        ),
        (
            "--expected-experiment-manifest-sha256",
            args.expected_experiment_manifest_sha256,
        ),
        ("--expected-implementation-sha256", args.expected_implementation_sha256),
    )
    for flag, value in sha_arguments:
        if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise SystemExit(f"{flag} must be a 64-character hexadecimal SHA-256.")

    experiment_values = (
        args.experiment_manifest,
        args.expected_experiment_manifest_sha256,
        args.provenance_repo_root,
        args.expected_implementation_sha256,
    )
    if any(value is not None for value in experiment_values) and not all(
        value is not None for value in experiment_values
    ):
        raise SystemExit(
            "--experiment-manifest, --expected-experiment-manifest-sha256, "
            "--provenance-repo-root, and --expected-implementation-sha256 "
            "must be provided together."
        )

    train_value = not args.disable_value
    train_opponent = not args.disable_opponent
    if not train_value and not train_opponent:
        raise SystemExit("At least one of value/opponent training must be enabled.")

    if train_value:
        if args.value_replay is None or not args.value_replay:
            raise SystemExit("--value-replay is required unless --disable-value is set.")
        missing_value_paths = [str(path) for path in args.value_replay if not path.exists()]
        if missing_value_paths:
            raise SystemExit(f"value replay not found: {missing_value_paths}")
    elif args.expected_value_replay_content_sha256 is not None:
        raise SystemExit(
            "--expected-value-replay-content-sha256 cannot be used with --disable-value."
        )
    if train_opponent:
        if args.opponent_replay is None or not args.opponent_replay:
            raise SystemExit("--opponent-replay is required unless --disable-opponent is set.")
        missing_opponent_paths = [str(path) for path in args.opponent_replay if not path.exists()]
        if missing_opponent_paths:
            raise SystemExit(f"opponent replay not found: {missing_opponent_paths}")
    elif args.expected_opponent_replay_content_sha256 is not None:
        raise SystemExit(
            "--expected-opponent-replay-content-sha256 cannot be used with --disable-opponent."
        )

    if (
        args.warm_start_value_checkpoint is not None
        and not args.warm_start_value_checkpoint.exists()
    ):
        raise SystemExit(
            f"warm-start value checkpoint not found: {args.warm_start_value_checkpoint}"
        )
    if (
        args.warm_start_opponent_checkpoint is not None
        and not args.warm_start_opponent_checkpoint.exists()
    ):
        raise SystemExit(
            f"warm-start opponent checkpoint not found: {args.warm_start_opponent_checkpoint}"
        )
    if (
        args.expected_warm_start_value_sha256 is not None
        and args.warm_start_value_checkpoint is None
    ):
        raise SystemExit(
            "--expected-warm-start-value-sha256 requires --warm-start-value-checkpoint."
        )
    if (
        args.expected_warm_start_opponent_sha256 is not None
        and args.warm_start_opponent_checkpoint is None
    ):
        raise SystemExit(
            "--expected-warm-start-opponent-sha256 requires --warm-start-opponent-checkpoint."
        )


def _resolve_training_provenance(args: argparse.Namespace) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "valueReplayContentSha256": None,
        "opponentReplayContentSha256": None,
        "warmStartValueSha256": None,
        "warmStartOpponentSha256": None,
        "experimentManifest": None,
        "experimentManifestSha256": None,
        "implementationSha256": None,
    }
    if args.expected_value_replay_content_sha256 is not None:
        assert args.value_replay is not None
        actual = replay_content_sha256(args.value_replay)
        _expect_sha256(
            label="value replay content",
            actual=actual,
            expected=args.expected_value_replay_content_sha256,
        )
        provenance["valueReplayContentSha256"] = actual
    if args.expected_opponent_replay_content_sha256 is not None:
        assert args.opponent_replay is not None
        actual = replay_content_sha256(args.opponent_replay)
        _expect_sha256(
            label="opponent replay content",
            actual=actual,
            expected=args.expected_opponent_replay_content_sha256,
        )
        provenance["opponentReplayContentSha256"] = actual

    if args.warm_start_value_checkpoint is not None:
        actual = sha256_file(args.warm_start_value_checkpoint)
        if args.expected_warm_start_value_sha256 is not None:
            _expect_sha256(
                label="warm-start value checkpoint",
                actual=actual,
                expected=args.expected_warm_start_value_sha256,
            )
        provenance["warmStartValueSha256"] = actual
    if args.warm_start_opponent_checkpoint is not None:
        actual = sha256_file(args.warm_start_opponent_checkpoint)
        if args.expected_warm_start_opponent_sha256 is not None:
            _expect_sha256(
                label="warm-start opponent checkpoint",
                actual=actual,
                expected=args.expected_warm_start_opponent_sha256,
            )
        provenance["warmStartOpponentSha256"] = actual

    if args.experiment_manifest is not None:
        assert args.expected_experiment_manifest_sha256 is not None
        assert args.provenance_repo_root is not None
        assert args.expected_implementation_sha256 is not None
        manifest_path = args.experiment_manifest.resolve()
        actual_manifest_sha = sha256_file(manifest_path)
        _expect_sha256(
            label="experiment manifest",
            actual=actual_manifest_sha,
            expected=args.expected_experiment_manifest_sha256,
        )
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Unable to read experiment manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest_payload, dict):
            raise SystemExit("Experiment manifest must contain a JSON object.")
        provenance_payload = manifest_payload.get("provenance")
        if not isinstance(provenance_payload, dict):
            raise SystemExit("Experiment manifest provenance must be an object.")
        raw_files = provenance_payload.get("implementationFiles")
        if (
            not isinstance(raw_files, list)
            or not raw_files
            or not all(isinstance(item, str) and item for item in raw_files)
        ):
            raise SystemExit(
                "Experiment manifest provenance.implementationFiles must be a "
                "non-empty string list."
            )
        implementation_files = [str(item) for item in raw_files]
        try:
            actual_implementation_sha = named_files_content_sha256(
                repo_root=args.provenance_repo_root.resolve(),
                relative_paths=implementation_files,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _expect_sha256(
            label="training implementation",
            actual=actual_implementation_sha,
            expected=args.expected_implementation_sha256,
        )
        manifest_expected = provenance_payload.get("expectedImplementationSha256")
        if manifest_expected != args.expected_implementation_sha256.lower():
            raise SystemExit(
                "Experiment manifest expectedImplementationSha256 does not match "
                "--expected-implementation-sha256."
            )
        provenance.update(
            {
                "experimentManifest": str(manifest_path),
                "experimentManifestSha256": actual_manifest_sha,
                "implementationSha256": actual_implementation_sha,
                "implementationFiles": implementation_files,
            }
        )
    return provenance


def _expect_sha256(*, label: str, actual: str, expected: str) -> None:
    if actual.lower() != expected.lower():
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected={expected.lower()} actual={actual.lower()}"
        )


def _merge_replay_paths(
    *,
    inline_paths: List[Path] | None,
    list_path: Path | None,
    label: str,
) -> List[Path] | None:
    paths = list(inline_paths or ())
    if list_path is None:
        return paths or None
    if not list_path.exists() or not list_path.is_file():
        raise SystemExit(f"{label} replay list not found: {list_path}")
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = list_path.parent / path
        paths.append(path.resolve())
    if not paths:
        raise SystemExit(f"{label} replay list is empty: {list_path}")
    if len(set(paths)) != len(paths):
        raise SystemExit(f"{label} replay paths contain duplicates.")
    return paths


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


def _update_sampling_trace(*, digest: Any, step: int, indices: List[int]) -> None:
    digest.update(f"{step}:".encode())
    for index in indices:
        digest.update(f"{index},".encode())
    digest.update(b"\n")


def _metric_or_na(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.12+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


if __name__ == "__main__":
    raise SystemExit(main())
