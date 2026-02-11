from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.checkpoint_manifest import DEFAULT_MANIFEST_RELATIVE_PATH
from scripts.opponent_pool import (
    PoolCheckpoint,
    filter_pool_excluding_checkpoint,
    load_promoted_checkpoints,
    split_evenly,
    weighted_game_split,
)
from scripts.promote_td_checkpoint import promote_checkpoint_pair
from scripts.td_loop_common import (
    LoopCheckpoint,
    build_train_command,
    checkpoints_from_train_summary,
    concat_jsonl_files,
    eligible_checkpoints_for_policy,
    read_json,
    run_step,
    select_latest_checkpoint,
)
from scripts.td_loop_eval_common import (
    EvalRow,
    PromotionThresholds,
    build_eval_payload,
    evaluate_promotion_gate,
    pool_eval_rows,
    read_eval_row,
)
from scripts.td_loop_selfplay_eval import (
    _build_eval_command_vs_incumbent,
    _build_eval_command_vs_search,
    _build_gate_command_vs_incumbent,
    _promotion_decision,
)

REPLAY_REGIME = "chunk-local-selfplay-mixed"


@dataclass(frozen=True)
class CollectProfile:
    label: str
    games: int
    player_a_policy: str
    player_b_policy: str
    player_a_td_search: PoolCheckpoint | None
    player_b_td_search: PoolCheckpoint | None


@dataclass(frozen=True)
class ReplayChunk:
    label: str
    value_path: Path
    opponent_path: Path
    value_lines: int
    opponent_lines: int


@dataclass(frozen=True)
class ReplayWindowResult:
    source: str
    chunks: Sequence[ReplayChunk]
    value_paths: Sequence[Path]
    opponent_paths: Sequence[Path]
    summary_path: Path
    value_lines: int
    opponent_lines: int
    max_value_lines: int
    max_opponent_lines: int
    window_size: int


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    chunks_dir: Path
    eval_dir: Path
    loop_summary_path: Path
    progress_path: Path
    incumbent_checkpoint: PoolCheckpoint
    latest_checkpoint: LoopCheckpoint
    loop_started: float


@dataclass(frozen=True)
class ChunkExecutionResult:
    chunk_label: str
    latest_checkpoint: LoopCheckpoint
    command_row: Dict[str, Any]
    chunk_row: Dict[str, Any]
    trained_latest_checkpoint: LoopCheckpoint | None = None
    checkpoint_selection: CheckpointSelectionResult | None = None
    replay_chunk: ReplayChunk | None = None
    replay_window: ReplayWindowResult | None = None


@dataclass(frozen=True)
class BlockCandidate:
    chunk_label: str
    checkpoint: LoopCheckpoint
    replay_chunk: ReplayChunk | None = None


@dataclass(frozen=True)
class CheckpointSelectionResult:
    enabled: bool
    reason: str
    selected_checkpoint: LoopCheckpoint
    trained_latest_checkpoint: LoopCheckpoint
    accepted_before: LoopCheckpoint
    seed_start_indices: List[int]
    candidates: List[Dict[str, Any]]
    commands: List[List[str]]
    summary_path: Path


@dataclass(frozen=True)
class ChunkGateResult:
    enabled: bool
    accepted: bool
    reason: str
    candidate_checkpoint: LoopCheckpoint
    accepted_before: LoopCheckpoint
    accepted_after: LoopCheckpoint
    seed_start_indices: List[int]
    windows: List[Dict[str, Any]]
    pooled: EvalRow | None
    checks: Dict[str, Any]
    window_checks: List[Dict[str, Any]]
    commands: List[List[str]]


@dataclass(frozen=True)
class BlockGeneratorUpdateResult:
    block_label: str
    block_index: int
    chunk_labels: List[str]
    candidates: List[Dict[str, Any]]
    checkpoint_selection: CheckpointSelectionResult
    generator_gate: ChunkGateResult
    selected_chunk_label: str | None
    summary_path: Path
    commands: Dict[str, Any]


@dataclass(frozen=True)
class PromotionStageResult:
    baseline_windows: List[Dict[str, Any]]
    incumbent_windows: List[Dict[str, Any]]
    pooled_baseline: EvalRow
    pooled_incumbent: EvalRow
    promotion: Dict[str, Any]
    commands: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TD self-play loop with opponent-pool collection, "
            "strict baseline evals, and incumbent head-to-head promotion gates."
        )
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/td_loops"))
    parser.add_argument("--run-label", type=str, default="td-loop-selfplay")
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--cloud-vcpus", type=int, default=8, choices=(8, 16, 32))

    # Keep chunked collection/training, but wait longer before promotion eval.
    parser.add_argument("--chunks-per-loop", type=int, default=12)
    parser.add_argument("--collect-games", type=int, default=600)
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="Parallel collection shards per profile. 1 disables sharding.",
    )
    parser.add_argument("--collect-seed-prefix", type=str, default="td-loop-collect-selfplay")
    parser.add_argument("--collect-progress-every-games", type=int, default=50)
    parser.add_argument("--collect-search-worlds", type=int, default=6)
    parser.add_argument("--collect-search-rollouts", type=int, default=1)
    parser.add_argument("--collect-search-depth", type=int, default=14)
    parser.add_argument("--collect-search-max-root-actions", type=int, default=6)
    parser.add_argument("--collect-search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--collect-td-worlds", type=int, default=8)
    parser.add_argument("--collect-td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--collect-td-search-sample-opponent-actions", action="store_true")
    parser.add_argument(
        "--collect-selfplay-share",
        type=float,
        default=0.60,
        help="Share for candidate td-search vs itself.",
    )
    parser.add_argument(
        "--collect-pool-share",
        type=float,
        default=0.25,
        help="Share for candidate td-search vs older promoted td-search opponents.",
    )
    parser.add_argument(
        "--collect-search-anchor-share",
        type=float,
        default=0.15,
        help="Share for candidate td-search vs search anchor games.",
    )
    parser.add_argument(
        "--collect-opponent-pool-size",
        type=int,
        default=4,
        help="Max promoted checkpoints kept in opponent pool (newest first).",
    )

    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--train-value-batch-size", type=int, default=128)
    parser.add_argument("--train-opponent-batch-size", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--train-hidden-dim", type=int, default=256)
    parser.add_argument("--train-gamma", type=float, default=0.995)
    parser.add_argument("--train-value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-value-weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-opponent-learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-opponent-weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-target-sync-interval", type=int, default=200)
    parser.add_argument(
        "--train-value-target-mode",
        type=str,
        choices=("td0", "td-lambda"),
        default="td-lambda",
    )
    parser.add_argument("--train-td-lambda", type=float, default=0.7)
    parser.add_argument("--train-use-mse-loss", action="store_true")
    parser.add_argument("--train-disable-value", action="store_true")
    parser.add_argument("--train-disable-opponent", action="store_true")
    parser.add_argument("--train-save-every-steps", type=int, default=1000)
    parser.add_argument("--train-progress-every-steps", type=int, default=50)
    parser.add_argument("--train-num-threads", type=int, default=None)
    parser.add_argument("--train-num-interop-threads", type=int, default=None)
    parser.add_argument("--train-warm-start-value-checkpoint", type=Path, default=None)
    parser.add_argument("--train-warm-start-opponent-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--train-replay-window-chunks",
        type=int,
        default=3,
        help=(
            "Train each chunk from the current chunk plus the last N-1 chunks from "
            "--train-replay-window-source."
        ),
    )
    parser.add_argument(
        "--train-replay-window-source",
        type=str,
        choices=("accepted", "recent"),
        default="accepted",
        help=(
            "Replay history source for the training window. 'accepted' uses chunks "
            "whose generator gate passed; 'recent' uses recent trained chunks even "
            "when the generator gate failed."
        ),
    )
    parser.add_argument(
        "--train-replay-window-max-value-lines",
        type=int,
        default=0,
        help="Optional cap for value replay lines in the merged training window; 0 is uncapped.",
    )
    parser.add_argument(
        "--train-replay-window-max-opponent-lines",
        type=int,
        default=0,
        help="Optional cap for opponent replay lines in the merged training window; 0 is uncapped.",
    )

    parser.add_argument("--eval-games-per-side", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--eval-seed-prefix", type=str, default="td-loop-eval-selfplay")
    parser.add_argument(
        "--eval-seed-start-indices",
        type=int,
        nargs="+",
        default=[0, 10000],
    )
    parser.add_argument("--eval-progress-every-games", type=int, default=10)
    parser.add_argument("--eval-progress-log-minutes", type=float, default=30.0)
    parser.add_argument("--eval-worker-torch-threads", type=int, default=1)
    parser.add_argument("--eval-worker-torch-interop-threads", type=int, default=1)
    parser.add_argument("--eval-worker-blas-threads", type=int, default=1)
    parser.add_argument("--eval-search-worlds", type=int, default=6)
    parser.add_argument("--eval-search-rollouts", type=int, default=1)
    parser.add_argument("--eval-search-depth", type=int, default=14)
    parser.add_argument("--eval-search-max-root-actions", type=int, default=6)
    parser.add_argument("--eval-search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--eval-td-worlds", type=int, default=8)
    parser.add_argument("--eval-td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--eval-td-search-sample-opponent-actions", action="store_true")

    parser.add_argument(
        "--incumbent-eval-games-per-side",
        type=int,
        default=200,
        help="Candidate vs incumbent td-search eval games per side.",
    )
    parser.add_argument(
        "--incumbent-eval-workers",
        type=int,
        default=None,
        help="Workers for incumbent eval windows (default: --eval-workers).",
    )
    parser.add_argument(
        "--incumbent-eval-seed-prefix",
        type=str,
        default="td-loop-eval-incumbent",
    )
    parser.add_argument(
        "--incumbent-eval-seed-start-indices",
        type=int,
        nargs="+",
        default=[20000, 30000],
    )

    parser.add_argument(
        "--disable-chunk-gate",
        action="store_true",
        help="Accept each trained chunk as the next data generator without a sequential incumbent gate.",
    )
    parser.add_argument(
        "--generator-update-chunks",
        type=int,
        default=1,
        help=(
            "Chunks per generator update attempt. Value 1 keeps the current per-chunk "
            "gate cadence; larger values select the best candidate from each block "
            "and gate only at block boundaries."
        ),
    )
    parser.add_argument("--block-selection-games-per-side", type=int, default=20)
    parser.add_argument(
        "--block-selection-seed-prefix",
        type=str,
        default="td-loop-block-selection",
    )
    parser.add_argument(
        "--block-selection-seed-start-indices",
        type=int,
        nargs="+",
        default=[50000],
    )
    parser.add_argument(
        "--chunk-gate-games-per-side",
        "--chunk-gate-max-games-per-side",
        type=int,
        default=200,
        help=(
            "Maximum games per side for the resumable candidate-vs-current-generator "
            "sequential gate."
        ),
    )
    parser.add_argument(
        "--chunk-gate-workers",
        type=int,
        default=None,
        help="Workers for per-chunk generator gates (default: --eval-workers).",
    )
    parser.add_argument("--chunk-gate-seed-prefix", type=str, default="td-loop-chunk-gate")
    parser.add_argument(
        "--chunk-gate-seed-start-indices",
        type=int,
        nargs="+",
        default=[40000],
    )
    parser.add_argument("--chunk-gate-h0-win-rate", type=float, default=0.50)
    parser.add_argument("--chunk-gate-h1-win-rate", type=float, default=0.55)
    parser.add_argument("--chunk-gate-alpha", type=float, default=0.05)
    parser.add_argument("--chunk-gate-beta", type=float, default=0.10)
    parser.add_argument("--chunk-gate-batch-games-per-side", type=int, default=25)
    parser.add_argument("--chunk-gate-min-win-rate", type=float, default=0.55)
    parser.add_argument("--chunk-gate-max-side-gap", type=float, default=0.15)
    parser.add_argument("--chunk-gate-min-ci-low", type=float, default=0.0)
    parser.add_argument("--chunk-gate-max-window-side-gap", type=float, default=0.20)
    parser.add_argument(
        "--checkpoint-selection-games-per-side",
        type=int,
        default=4,
        help=(
            "Cheap candidate-vs-current-generator eval games per side for each saved "
            "training checkpoint before the chunk gate."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-workers",
        type=int,
        default=None,
        help="Workers for checkpoint selection evals (default: chunk-gate workers, then eval workers).",
    )
    parser.add_argument(
        "--checkpoint-selection-seed-prefix",
        type=str,
        default="td-loop-checkpoint-selection",
    )
    parser.add_argument(
        "--checkpoint-selection-seed-start-indices",
        type=int,
        nargs="+",
        default=[45000],
    )

    # Baseline-vs-search gate.
    parser.add_argument("--promotion-min-win-rate", type=float, default=0.55)
    parser.add_argument("--promotion-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-min-ci-low", type=float, default=0.5)
    parser.add_argument("--promotion-max-window-side-gap", type=float, default=0.10)

    # Incumbent head-to-head gate.
    parser.add_argument("--promotion-incumbent-min-win-rate", type=float, default=0.52)
    parser.add_argument("--promotion-incumbent-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-incumbent-min-ci-low", type=float, default=0.5)
    parser.add_argument("--promotion-incumbent-max-window-side-gap", type=float, default=0.10)
    parser.add_argument("--promotion-manifest-path", type=Path, default=DEFAULT_MANIFEST_RELATIVE_PATH)
    parser.add_argument(
        "--promotion-checkpoint-root",
        type=Path,
        default=Path("models/td_checkpoints"),
    )
    parser.add_argument(
        "--promotion-manifest-key",
        type=str,
        default=None,
        help="Manifest key used when a loop promotion passes (default: derived from run id).",
    )
    parser.add_argument(
        "--disable-manifest-promotion",
        action="store_true",
        help="Do not copy promoted checkpoints into models/td_checkpoints or update manifest.",
    )

    parser.add_argument("--progress-heartbeat-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cloud:
        _apply_cloud_profile(args)
    _require_supported_runtime(args.python_bin)
    _validate_args(args)
    return run_selfplay_loop(args)


def run_selfplay_loop(args: argparse.Namespace) -> int:
    context = initialize_selfplay_run(args)
    blocks_dir = context.run_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    commands: Dict[str, Any] = {"chunks": [], "blocks": []}
    chunk_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []
    learner_checkpoint = context.latest_checkpoint
    generator_checkpoint = context.latest_checkpoint
    accepted_chunk_label: str | None = None
    accepted_replay_chunks: List[ReplayChunk] = []
    training_replay_chunks: List[ReplayChunk] = []
    current_block_candidates: List[BlockCandidate] = []

    for chunk_index in range(1, args.chunks_per_loop + 1):
        block_index = _block_index_for_chunk(args=args, chunk_index=chunk_index)
        block_position = _block_position_for_chunk(args=args, chunk_index=chunk_index)
        block_label = _block_label(block_index)
        learner_checkpoint_before = learner_checkpoint
        generator_checkpoint_before = generator_checkpoint
        chunk_result = run_selfplay_chunk(
            args=args,
            run_id=context.run_id,
            chunk_index=chunk_index,
            chunks_dir=context.chunks_dir,
            learner_checkpoint=learner_checkpoint_before,
            generator_checkpoint=generator_checkpoint_before,
            accepted_replay_chunks=tuple(accepted_replay_chunks),
            training_replay_chunks=tuple(training_replay_chunks),
            progress_path=context.progress_path,
        )
        learner_checkpoint = chunk_result.latest_checkpoint
        if chunk_result.replay_chunk is not None:
            training_replay_chunks.append(chunk_result.replay_chunk)
        current_block_candidates.append(
            BlockCandidate(
                chunk_label=chunk_result.chunk_label,
                checkpoint=learner_checkpoint,
                replay_chunk=chunk_result.replay_chunk,
            )
        )

        block_result: BlockGeneratorUpdateResult | None = None
        if _is_generator_update_boundary(args=args, chunk_index=chunk_index):
            block_result = run_block_generator_update(
                args=args,
                python_bin=args.python_bin,
                run_id=context.run_id,
                block_index=block_index,
                block_candidates=current_block_candidates,
                blocks_dir=blocks_dir,
                generator_checkpoint=generator_checkpoint_before,
                progress_path=context.progress_path,
                log_prefix="[td-loop-selfplay]",
            )
            generator_checkpoint = block_result.generator_gate.accepted_after
            commands["blocks"].append(block_result.commands)
            block_rows.append(block_update_payload(block_result))
            if block_result.generator_gate.accepted:
                accepted_chunk_label = block_result.selected_chunk_label
                accepted_replay = _replay_chunk_for_label(
                    block_result.selected_chunk_label,
                    current_block_candidates,
                )
                if accepted_replay is not None:
                    accepted_replay_chunks.append(accepted_replay)
            current_block_candidates = []
        else:
            generator_checkpoint = generator_checkpoint_before

        gate_result = (
            block_result.generator_gate
            if block_result is not None
            else deferred_generator_gate_result(
                candidate_checkpoint=learner_checkpoint,
                accepted_checkpoint=generator_checkpoint_before,
            )
        )
        command_row = {
            **chunk_result.command_row,
            "block": {
                "label": block_label,
                "index": block_index,
                "position": block_position,
                "size": args.generator_update_chunks,
                "boundary": block_result is not None,
            },
            "blockSelection": (
                block_result.checkpoint_selection.commands
                if block_result is not None
                else []
            ),
            "generatorGate": gate_result.commands,
        }
        chunk_row = build_gated_chunk_row(
            chunk_row=chunk_result.chunk_row,
            gate_result=gate_result,
            replay_chunk=chunk_result.replay_chunk,
            learner_checkpoint_before=learner_checkpoint_before,
            learner_checkpoint_after=learner_checkpoint,
            generator_checkpoint_before=generator_checkpoint_before,
            generator_checkpoint_after=generator_checkpoint,
        )
        chunk_row["block"] = {
            "label": block_label,
            "index": block_index,
            "position": block_position,
            "size": args.generator_update_chunks,
            "boundary": block_result is not None,
            "summary": str(block_result.summary_path) if block_result is not None else None,
        }
        if (
            block_result is not None
            and chunk_result.replay_chunk is not None
            and "replayForTraining" in chunk_row
        ):
            replay_is_selected = block_result.selected_chunk_label == chunk_result.chunk_label
            chunk_row["replayForTraining"]["eligible"] = bool(
                block_result.generator_gate.accepted and replay_is_selected
            )
            if block_result.generator_gate.accepted and not replay_is_selected:
                chunk_row["replayForTraining"]["reason"] = "block_gate_selected_different_chunk"
        chunk_summary_path = write_selfplay_chunk_summary(
            run_id=context.run_id,
            chunk_dir=context.chunks_dir / chunk_result.chunk_label,
            command_row=command_row,
            chunk_row=chunk_row,
        )
        chunk_row["chunkSummary"] = str(chunk_summary_path)
        commands["chunks"].append(command_row)
        chunk_rows.append(chunk_row)

    promotion_checkpoint = generator_checkpoint
    promotion_stage = run_promotion_stage(
        args=args,
        eval_dir=context.eval_dir,
        latest_checkpoint=promotion_checkpoint,
        incumbent_checkpoint=context.incumbent_checkpoint,
        progress_path=context.progress_path,
    )
    commands["promotionEvals"] = promotion_stage.commands
    manifest_promotion = _register_manifest_promotion_if_passed(
        args=args,
        context=context,
        latest_checkpoint=promotion_checkpoint,
        latest_chunk_label=accepted_chunk_label,
        promotion_stage=promotion_stage,
    )
    if manifest_promotion is not None:
        commands["manifestPromotion"] = manifest_promotion

    loop_elapsed_minutes = (time.perf_counter() - context.loop_started) / 60.0
    payload = build_selfplay_loop_summary(
        args=args,
        context=context,
        commands=commands,
        chunk_rows=chunk_rows,
        block_rows=block_rows,
        promotion_stage=promotion_stage,
        final_learner_checkpoint=learner_checkpoint,
        final_generator_checkpoint=generator_checkpoint,
        promotion_checkpoint=promotion_checkpoint,
        elapsed_minutes=loop_elapsed_minutes,
    )
    context.loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(build_selfplay_terminal_report(context=context, promotion_stage=promotion_stage), indent=2))
    return 0


def _block_label(block_index: int) -> str:
    return f"block-{block_index:03d}"


def _block_index_for_chunk(*, args: argparse.Namespace, chunk_index: int) -> int:
    return ((chunk_index - 1) // int(args.generator_update_chunks)) + 1


def _block_position_for_chunk(*, args: argparse.Namespace, chunk_index: int) -> int:
    return ((chunk_index - 1) % int(args.generator_update_chunks)) + 1


def _is_generator_update_boundary(*, args: argparse.Namespace, chunk_index: int) -> bool:
    return (
        chunk_index % int(args.generator_update_chunks) == 0
        or chunk_index == int(args.chunks_per_loop)
    )


def run_block_generator_update(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    run_id: str,
    block_index: int,
    block_candidates: Sequence[BlockCandidate],
    blocks_dir: Path,
    generator_checkpoint: LoopCheckpoint,
    progress_path: Path,
    log_prefix: str,
) -> BlockGeneratorUpdateResult:
    if not block_candidates:
        raise SystemExit("Cannot update generator from an empty block.")

    block_label = _block_label(block_index)
    block_dir = blocks_dir / block_label
    eval_dir = block_dir / "eval"
    block_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_selection = run_block_checkpoint_selection(
        args=args,
        python_bin=python_bin,
        block_label=block_label,
        eval_dir=eval_dir / "checkpoint_selection",
        block_candidates=block_candidates,
        accepted_checkpoint=generator_checkpoint,
        progress_path=progress_path,
        log_prefix=log_prefix,
    )
    selected_chunk_label = _selected_block_chunk_label(
        checkpoint=checkpoint_selection.selected_checkpoint,
        block_candidates=block_candidates,
    )
    if selected_chunk_label is None:
        raise SystemExit(
            f"Block selection for {block_label} selected a checkpoint that does not "
            "match any block candidate."
        )

    gate_result = run_chunk_gate(
        args=args,
        python_bin=python_bin,
        chunk_label=block_label,
        eval_dir=eval_dir / "generator_gate",
        candidate_checkpoint=checkpoint_selection.selected_checkpoint,
        accepted_checkpoint=generator_checkpoint,
        progress_path=progress_path,
        log_prefix=log_prefix,
    )
    candidate_rows = [
        {
            "chunk": candidate.chunk_label,
            "checkpoint": _checkpoint_payload(candidate.checkpoint),
            "replay": (
                {
                    "value": str(candidate.replay_chunk.value_path),
                    "opponent": str(candidate.replay_chunk.opponent_path),
                    "valueLines": candidate.replay_chunk.value_lines,
                    "opponentLines": candidate.replay_chunk.opponent_lines,
                }
                if candidate.replay_chunk is not None
                else None
            ),
        }
        for candidate in block_candidates
    ]
    commands = {
        "block": block_label,
        "chunks": [candidate.chunk_label for candidate in block_candidates],
        "selectedChunk": selected_chunk_label,
        "checkpointSelection": checkpoint_selection.commands,
        "generatorGate": gate_result.commands,
        "generatorCheckpointBefore": _checkpoint_payload(generator_checkpoint),
        "generatorCheckpointAfter": _checkpoint_payload(gate_result.accepted_after),
    }
    result = BlockGeneratorUpdateResult(
        block_label=block_label,
        block_index=block_index,
        chunk_labels=[candidate.chunk_label for candidate in block_candidates],
        candidates=candidate_rows,
        checkpoint_selection=checkpoint_selection,
        generator_gate=gate_result,
        selected_chunk_label=selected_chunk_label,
        summary_path=block_dir / "block.summary.json",
        commands=commands,
    )
    write_selfplay_block_summary(run_id=run_id, result=result)
    print(
        f"{log_prefix} generator-block block={block_label} "
        f"chunks={','.join(result.chunk_labels)} selected={selected_chunk_label} "
        f"accepted={gate_result.accepted}",
        flush=True,
    )
    return result


def run_block_checkpoint_selection(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    block_label: str,
    eval_dir: Path,
    block_candidates: Sequence[BlockCandidate],
    accepted_checkpoint: LoopCheckpoint,
    progress_path: Path,
    log_prefix: str,
) -> CheckpointSelectionResult:
    selection_args = argparse.Namespace(**vars(args))
    selection_args.checkpoint_selection_games_per_side = args.block_selection_games_per_side
    selection_args.checkpoint_selection_seed_prefix = args.block_selection_seed_prefix
    selection_args.checkpoint_selection_seed_start_indices = list(
        args.block_selection_seed_start_indices
    )
    return run_checkpoint_selection(
        args=selection_args,
        python_bin=python_bin,
        chunk_label=block_label,
        eval_dir=eval_dir,
        checkpoints=[candidate.checkpoint for candidate in block_candidates],
        accepted_checkpoint=accepted_checkpoint,
        progress_path=progress_path,
        log_prefix=log_prefix,
    )


def deferred_generator_gate_result(
    *,
    candidate_checkpoint: LoopCheckpoint,
    accepted_checkpoint: LoopCheckpoint,
) -> ChunkGateResult:
    return ChunkGateResult(
        enabled=False,
        accepted=False,
        reason="generator_gate_deferred",
        candidate_checkpoint=candidate_checkpoint,
        accepted_before=accepted_checkpoint,
        accepted_after=accepted_checkpoint,
        seed_start_indices=[],
        windows=[],
        pooled=None,
        checks={"deferred": True},
        window_checks=[],
        commands=[],
    )


def _selected_block_chunk_label(
    *, checkpoint: LoopCheckpoint, block_candidates: Sequence[BlockCandidate]
) -> str | None:
    for candidate in block_candidates:
        if _same_loop_checkpoint(candidate.checkpoint, checkpoint):
            return candidate.chunk_label
    return None


def _replay_chunk_for_label(
    chunk_label: str | None,
    block_candidates: Sequence[BlockCandidate],
) -> ReplayChunk | None:
    if chunk_label is None:
        return None
    for candidate in block_candidates:
        if candidate.chunk_label == chunk_label:
            return candidate.replay_chunk
    return None


def _same_loop_checkpoint(left: LoopCheckpoint, right: LoopCheckpoint) -> bool:
    return (
        left.step == right.step
        and left.value_path == right.value_path
        and left.opponent_path == right.opponent_path
    )


def initialize_selfplay_run(args: argparse.Namespace) -> RunContext:
    promoted_pool = load_promoted_checkpoints(
        artifact_dir=args.artifact_dir,
        max_entries=args.collect_opponent_pool_size,
        require_paths=True,
    )
    warm_value, warm_opponent = _resolve_warm_start(args=args, promoted_pool=promoted_pool)
    incumbent_checkpoint = PoolCheckpoint(
        run_id="incumbent",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        value_path=warm_value,
        opponent_path=warm_opponent,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_id = f"{stamp}-{_slug(args.run_label)}"
    run_dir = args.artifact_dir / run_id
    chunks_dir = run_dir / "chunks"
    eval_dir = run_dir / "evals"
    loop_summary_path = run_dir / "loop.summary.json"
    progress_path = run_dir / "progress.json"
    for path in (run_dir, chunks_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        run_dir=run_dir,
        chunks_dir=chunks_dir,
        eval_dir=eval_dir,
        loop_summary_path=loop_summary_path,
        progress_path=progress_path,
        incumbent_checkpoint=incumbent_checkpoint,
        latest_checkpoint=LoopCheckpoint(
            step=0,
            value_path=warm_value,
            opponent_path=warm_opponent,
        ),
        loop_started=time.perf_counter(),
    )


def run_selfplay_chunk(
    *,
    args: argparse.Namespace,
    run_id: str,
    chunk_index: int,
    chunks_dir: Path,
    learner_checkpoint: LoopCheckpoint,
    generator_checkpoint: LoopCheckpoint,
    accepted_replay_chunks: Sequence[ReplayChunk],
    training_replay_chunks: Sequence[ReplayChunk] = (),
    progress_path: Path,
) -> ChunkExecutionResult:
    chunk_label = f"chunk-{chunk_index:03d}"
    chunk_dir = chunks_dir / chunk_label
    replay_dir = chunk_dir / "replay"
    train_dir = chunk_dir / "train"
    for path in (chunk_dir, replay_dir, train_dir):
        path.mkdir(parents=True, exist_ok=True)

    collect_value_path = replay_dir / "self_play.value.jsonl"
    collect_opponent_path = replay_dir / "self_play.opponent.jsonl"
    collect_summary_path = replay_dir / "self_play.summary.json"
    train_summary_path = train_dir / "summary.json"
    train_checkpoint_root = train_dir / "checkpoints"

    promoted_pool = load_promoted_checkpoints(
        artifact_dir=args.artifact_dir,
        max_entries=args.collect_opponent_pool_size,
        require_paths=True,
    )
    collect_profiles = _build_collect_profiles(
        args=args,
        candidate=PoolCheckpoint(
            run_id=run_id,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            value_path=generator_checkpoint.value_path,
            opponent_path=generator_checkpoint.opponent_path,
        ),
        promoted_pool=promoted_pool,
    )
    collect_commands = _run_collect_profiles(
        python_bin=args.python_bin,
        args=args,
        profiles=collect_profiles,
        replay_dir=replay_dir,
        run_id=f"{run_id}-{chunk_label}",
        collect_value_path=collect_value_path,
        collect_opponent_path=collect_opponent_path,
        collect_summary_path=collect_summary_path,
        heartbeat_minutes=args.progress_heartbeat_minutes,
        progress_path=progress_path,
    )
    current_replay = ReplayChunk(
        label=chunk_label,
        value_path=collect_value_path,
        opponent_path=collect_opponent_path,
        value_lines=_collect_summary_line_count(
            collect_summary_path,
            key="valueTransitions",
            label=f"collect summary {chunk_label}",
        ),
        opponent_lines=_collect_summary_line_count(
            collect_summary_path,
            key="opponentSamples",
            label=f"collect summary {chunk_label}",
        ),
    )
    replay_window = build_replay_window(
        args=args,
        chunk_label=chunk_label,
        train_dir=train_dir,
        accepted_replay_chunks=accepted_replay_chunks,
        training_replay_chunks=training_replay_chunks,
        current_replay=current_replay,
    )
    train_command = build_train_command(
        python_bin=args.python_bin,
        args=args,
        value_replays=replay_window.value_paths,
        opponent_replays=replay_window.opponent_paths,
        train_summary_path=train_summary_path,
        train_checkpoint_root=train_checkpoint_root,
        run_id=f"{run_id}-{chunk_label}",
        warm_start_value=learner_checkpoint.value_path,
        warm_start_opponent=learner_checkpoint.opponent_path,
    )
    run_step(
        name=f"train[{chunk_label}]",
        command=train_command,
        heartbeat_minutes=args.progress_heartbeat_minutes,
        progress_path=progress_path,
        log_prefix="[td-loop-selfplay]",
    )

    train_summary = read_json(train_summary_path, label=f"train summary {chunk_label}")
    checkpoints = checkpoints_from_train_summary(train_summary)
    trained_latest_checkpoint = select_latest_checkpoint(
        checkpoints=checkpoints,
        candidate_policy="td-search",
    )
    checkpoint_selection = run_checkpoint_selection(
        args=args,
        python_bin=args.python_bin,
        chunk_label=chunk_label,
        eval_dir=chunk_dir / "eval" / "checkpoint_selection",
        checkpoints=checkpoints,
        accepted_checkpoint=generator_checkpoint,
        progress_path=progress_path,
        log_prefix="[td-loop-selfplay]",
    )
    next_checkpoint = checkpoint_selection.selected_checkpoint
    return ChunkExecutionResult(
        chunk_label=chunk_label,
        latest_checkpoint=next_checkpoint,
        command_row={
            "chunk": chunk_label,
            "collectProfiles": collect_commands,
            "replayWindow": replay_window_payload(replay_window),
            "train": train_command,
            "checkpointSelection": checkpoint_selection.commands,
            "learnerCheckpointBefore": _checkpoint_payload(learner_checkpoint),
            "learnerCheckpointAfter": _checkpoint_payload(next_checkpoint),
            "generatorCheckpointBefore": _checkpoint_payload(generator_checkpoint),
        },
        chunk_row={
            "chunk": chunk_label,
            "replayRegime": REPLAY_REGIME,
            "learnerCheckpointBefore": _checkpoint_payload(learner_checkpoint),
            "learnerCheckpointAfter": _checkpoint_payload(next_checkpoint),
            "generatorCheckpointBefore": _checkpoint_payload(generator_checkpoint),
            "collectSummary": str(collect_summary_path),
            "collectReplay": {
                "value": str(collect_value_path),
                "opponent": str(collect_opponent_path),
            },
            "replayWindow": replay_window_payload(replay_window),
            "trainSummary": str(train_summary_path),
            "trainedLatestCheckpoint": {
                "step": trained_latest_checkpoint.step,
                "value": str(trained_latest_checkpoint.value_path),
                "opponent": str(trained_latest_checkpoint.opponent_path),
            },
            "checkpointSelection": checkpoint_selection_payload(checkpoint_selection),
            "latestCheckpoint": {
                "step": next_checkpoint.step,
                "value": str(next_checkpoint.value_path),
                "opponent": str(next_checkpoint.opponent_path),
            },
        },
        trained_latest_checkpoint=trained_latest_checkpoint,
        checkpoint_selection=checkpoint_selection,
        replay_chunk=current_replay,
        replay_window=replay_window,
    )


def build_replay_window(
    *,
    args: argparse.Namespace,
    chunk_label: str,
    train_dir: Path,
    accepted_replay_chunks: Sequence[ReplayChunk],
    training_replay_chunks: Sequence[ReplayChunk] = (),
    current_replay: ReplayChunk,
) -> ReplayWindowResult:
    source = str(args.train_replay_window_source)
    if source not in {"accepted", "recent"}:
        raise SystemExit(f"Unsupported replay window source: {source!r}")
    window_size = int(args.train_replay_window_chunks)
    prior_limit = max(0, window_size - 1)
    replay_history = (
        list(accepted_replay_chunks)
        if source == "accepted"
        else list(training_replay_chunks)
    )
    prior_chunks = list(replay_history[-prior_limit:]) if prior_limit > 0 else []
    window_chunks = [*prior_chunks, current_replay]
    for replay_chunk in window_chunks:
        _require_replay_chunk_paths(replay_chunk)

    replay_window_dir = train_dir / "replay_window"
    summary_path = replay_window_dir / "window.summary.json"
    value_paths = [chunk.value_path for chunk in window_chunks]
    opponent_paths = [chunk.opponent_path for chunk in window_chunks]
    total_value_lines = sum(chunk.value_lines for chunk in window_chunks)
    total_opponent_lines = sum(chunk.opponent_lines for chunk in window_chunks)
    max_value_lines = int(args.train_replay_window_max_value_lines)
    max_opponent_lines = int(args.train_replay_window_max_opponent_lines)
    result = ReplayWindowResult(
        source=source,
        chunks=window_chunks,
        value_paths=value_paths,
        opponent_paths=opponent_paths,
        summary_path=summary_path,
        value_lines=(
            total_value_lines if max_value_lines == 0 else min(total_value_lines, max_value_lines)
        ),
        opponent_lines=(
            total_opponent_lines
            if max_opponent_lines == 0
            else min(total_opponent_lines, max_opponent_lines)
        ),
        max_value_lines=max_value_lines,
        max_opponent_lines=max_opponent_lines,
        window_size=window_size,
    )
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "chunk": chunk_label,
        **replay_window_payload(result),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def replay_window_payload(result: ReplayWindowResult) -> Dict[str, Any]:
    return {
        "source": result.source,
        "windowSize": result.window_size,
        "chunks": [
            {
                "chunk": chunk.label,
                "valueReplay": str(chunk.value_path),
                "opponentReplay": str(chunk.opponent_path),
                "valueLines": chunk.value_lines,
                "opponentLines": chunk.opponent_lines,
            }
            for chunk in result.chunks
        ],
        "valueReplayFiles": [str(path) for path in result.value_paths],
        "opponentReplayFiles": [str(path) for path in result.opponent_paths],
        "summary": str(result.summary_path),
        "valueLines": result.value_lines,
        "opponentLines": result.opponent_lines,
        "maxValueLines": result.max_value_lines,
        "maxOpponentLines": result.max_opponent_lines,
    }


def replay_training_payload(
    *, replay_chunk: ReplayChunk, gate_result: ChunkGateResult
) -> Dict[str, Any]:
    return {
        "eligible": gate_result.accepted,
        "reason": gate_result.reason,
        "chunk": replay_chunk.label,
        "valueReplay": str(replay_chunk.value_path),
        "opponentReplay": str(replay_chunk.opponent_path),
        "valueLines": replay_chunk.value_lines,
        "opponentLines": replay_chunk.opponent_lines,
    }


def _require_replay_chunk_paths(replay_chunk: ReplayChunk) -> None:
    if not replay_chunk.value_path.exists():
        raise SystemExit(
            f"Replay window is missing value replay for {replay_chunk.label}: "
            f"{replay_chunk.value_path}"
        )
    if not replay_chunk.opponent_path.exists():
        raise SystemExit(
            f"Replay window is missing opponent replay for {replay_chunk.label}: "
            f"{replay_chunk.opponent_path}"
        )


def _collect_summary_line_count(summary_path: Path, *, key: str, label: str) -> int:
    payload = read_json(summary_path, label=label)
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Collect summary is missing results payload: {summary_path}")
    raw = results.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SystemExit(f"Collect summary has invalid {key}: {raw!r}")
    return raw


def run_checkpoint_selection(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    chunk_label: str,
    eval_dir: Path,
    checkpoints: Sequence[LoopCheckpoint],
    accepted_checkpoint: LoopCheckpoint,
    progress_path: Path,
    log_prefix: str,
    force_eval: bool = False,
) -> CheckpointSelectionResult:
    eligible = eligible_checkpoints_for_policy(
        checkpoints=checkpoints,
        candidate_policy="td-search",
    )
    if not eligible:
        raise SystemExit("No eligible td-search checkpoints for checkpoint selection.")

    trained_latest_checkpoint = eligible[-1]
    _require_td_search_checkpoint(accepted_checkpoint, label="checkpoint selection accepted checkpoint")
    for checkpoint in eligible:
        _require_td_search_checkpoint(checkpoint, label="checkpoint selection candidate")

    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "summary.json"

    if len(eligible) == 1:
        selected_checkpoint = eligible[0]
        result = CheckpointSelectionResult(
            enabled=True,
            reason="single_candidate",
            selected_checkpoint=selected_checkpoint,
            trained_latest_checkpoint=trained_latest_checkpoint,
            accepted_before=accepted_checkpoint,
            seed_start_indices=[],
            candidates=[
                {
                    "checkpoint": _checkpoint_payload(selected_checkpoint),
                    "selected": True,
                    "evaluation": None,
                    "score": None,
                    "commands": [],
                }
            ],
            commands=[],
            summary_path=summary_path,
        )
        _write_checkpoint_selection_summary(chunk_label=chunk_label, result=result)
        print(
            f"{log_prefix} checkpoint-selection chunk={chunk_label} "
            f"reason=single_candidate step={selected_checkpoint.step}",
            flush=True,
        )
        return result

    workers = (
        args.checkpoint_selection_workers
        or args.chunk_gate_workers
        or args.eval_workers
    )
    accepted_pool = PoolCheckpoint(
        run_id=f"{chunk_label}-accepted-before-selection",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        value_path=accepted_checkpoint.value_path,
        opponent_path=accepted_checkpoint.opponent_path,
    )
    candidate_rows: List[Dict[str, Any]] = []
    all_commands: List[List[str]] = []
    ranked: List[tuple[tuple[float, float, float, int], int, LoopCheckpoint]] = []

    for candidate_index, checkpoint in enumerate(eligible, start=1):
        windows: List[Dict[str, Any]] = []
        candidate_commands: List[List[str]] = []
        for seed_start_index in args.checkpoint_selection_seed_start_indices:
            out_path = eval_dir / (
                f"candidate-{candidate_index:02d}.step-{checkpoint.step:07d}."
                f"seed-{seed_start_index:06d}.json"
            )
            command = _build_eval_command_vs_incumbent(
                python_bin=python_bin,
                args=args,
                checkpoint=checkpoint,
                incumbent=accepted_pool,
                out_path=out_path,
                seed_prefix=args.checkpoint_selection_seed_prefix,
                seed_start_index=seed_start_index,
                workers=workers,
                games_per_side=args.checkpoint_selection_games_per_side,
            )
            candidate_commands.append(command)
            all_commands.append(command)
            if force_eval or not out_path.exists():
                run_step(
                    name=(
                        f"checkpoint-selection[{chunk_label} "
                        f"step={checkpoint.step} seed={seed_start_index}]"
                    ),
                    command=command,
                    heartbeat_minutes=args.progress_heartbeat_minutes,
                    progress_path=progress_path,
                    log_prefix=log_prefix,
                )
            else:
                print(
                    f"{log_prefix} existing checkpoint selection artifact found, "
                    f"skipping: {out_path}"
                )
            windows.append(
                {
                    "command": command,
                    "row": read_eval_row(out_path, opponent_policy="td-search"),
                }
            )

        rows = [window["row"] for window in windows]
        pooled = pool_eval_rows(eval_rows=rows, opponent_policy="td-search")
        candidate_payload = {
            "checkpoint": _checkpoint_payload(checkpoint),
            "selected": False,
            "score": {
                "candidateWinRate": pooled.candidate_win_rate,
                "candidateWinRateCi95": {"low": pooled.ci_low, "high": pooled.ci_high},
                "sideGap": pooled.side_gap,
                "totalGames": pooled.total_games,
            },
            "evaluation": build_eval_payload(
                args.checkpoint_selection_seed_start_indices,
                rows,
                pooled,
            ),
            "commands": candidate_commands,
        }
        candidate_rows.append(candidate_payload)
        ranked.append(
            (
                (
                    pooled.candidate_win_rate,
                    pooled.ci_low,
                    -pooled.side_gap,
                    checkpoint.step,
                    candidate_index,
                ),
                candidate_index - 1,
                checkpoint,
            )
        )

    _, selected_index, selected_checkpoint = max(ranked, key=lambda row: row[0])
    candidate_rows[selected_index]["selected"] = True
    result = CheckpointSelectionResult(
        enabled=True,
        reason="best_eval_score",
        selected_checkpoint=selected_checkpoint,
        trained_latest_checkpoint=trained_latest_checkpoint,
        accepted_before=accepted_checkpoint,
        seed_start_indices=list(args.checkpoint_selection_seed_start_indices),
        candidates=candidate_rows,
        commands=all_commands,
        summary_path=summary_path,
    )
    _write_checkpoint_selection_summary(chunk_label=chunk_label, result=result)
    selected_score = candidate_rows[selected_index]["score"]
    assert isinstance(selected_score, dict)
    print(
        f"{log_prefix} checkpoint-selection chunk={chunk_label} "
        f"selectedStep={selected_checkpoint.step} "
        f"trainedLatestStep={trained_latest_checkpoint.step} "
        f"winRate={float(selected_score['candidateWinRate']):.3f} "
        f"ciLow={float(selected_score['candidateWinRateCi95']['low']):.3f} "
        f"sideGap={float(selected_score['sideGap']):.3f}",
        flush=True,
    )
    return result


def run_chunk_gate(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    chunk_label: str,
    eval_dir: Path,
    candidate_checkpoint: LoopCheckpoint,
    accepted_checkpoint: LoopCheckpoint,
    progress_path: Path,
    log_prefix: str,
    force_eval: bool = False,
) -> ChunkGateResult:
    if bool(getattr(args, "disable_chunk_gate", False)):
        return ChunkGateResult(
            enabled=False,
            accepted=True,
            reason="chunk_gate_disabled",
            candidate_checkpoint=candidate_checkpoint,
            accepted_before=accepted_checkpoint,
            accepted_after=candidate_checkpoint,
            seed_start_indices=[],
            windows=[],
            pooled=None,
            checks={},
            window_checks=[],
            commands=[],
        )
    _require_td_search_checkpoint(candidate_checkpoint, label="chunk gate candidate")
    _require_td_search_checkpoint(accepted_checkpoint, label="chunk gate accepted checkpoint")

    eval_dir.mkdir(parents=True, exist_ok=True)
    workers = args.chunk_gate_workers or args.eval_workers
    accepted_pool = PoolCheckpoint(
        run_id=f"{chunk_label}-accepted-before",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        value_path=accepted_checkpoint.value_path,
        opponent_path=accepted_checkpoint.opponent_path,
    )
    windows: List[Dict[str, Any]] = []
    commands: List[List[str]] = []
    for seed_start_index in args.chunk_gate_seed_start_indices:
        out_path = eval_dir / f"generator_gate.seed-{seed_start_index:06d}.json"
        command = _build_gate_command_vs_incumbent(
            python_bin=python_bin,
            args=args,
            checkpoint=candidate_checkpoint,
            incumbent=accepted_pool,
            out_path=out_path,
            seed_prefix=args.chunk_gate_seed_prefix,
            seed_start_index=seed_start_index,
            workers=workers,
            batch_games_per_side=args.chunk_gate_batch_games_per_side,
            max_games_per_side=args.chunk_gate_games_per_side,
        )
        commands.append(command)
        gate_payload: Dict[str, Any] | None = None
        if force_eval and out_path.exists():
            out_path.unlink()
        if out_path.exists():
            gate_payload = read_json(out_path, label=f"chunk gate artifact {out_path.name}")
            status = str(gate_payload.get("status", ""))
            if status in ("accepted", "rejected", "completed"):
                print(f"{log_prefix} existing chunk gate artifact found, skipping: {out_path}")
            else:
                gate_payload = None
        if gate_payload is None:
            run_step(
                name=f"chunk-gate[{chunk_label} seed={seed_start_index}]",
                command=command,
                heartbeat_minutes=args.progress_heartbeat_minutes,
                progress_path=progress_path,
                log_prefix=log_prefix,
            )
            gate_payload = read_json(out_path, label=f"chunk gate artifact {out_path.name}")

        decision = gate_payload.get("decision")
        progress = gate_payload.get("progress")
        sprt = gate_payload.get("sprt")
        if not isinstance(decision, dict):
            raise SystemExit(f"Chunk gate artifact is missing decision payload: {out_path}")
        windows.append(
            {
                "command": command,
                "row": read_eval_row(out_path, opponent_policy="td-search"),
                "status": str(gate_payload.get("status", "")),
                "decision": str(decision.get("state", "")),
                "decisionReason": str(decision.get("reason", "")),
                "progress": dict(progress) if isinstance(progress, dict) else None,
                "sprt": dict(sprt) if isinstance(sprt, dict) else None,
            }
        )

    rows = [window["row"] for window in windows]
    pooled = pool_eval_rows(eval_rows=rows, opponent_policy="td-search")
    gate = evaluate_promotion_gate(
        eval_row=pooled,
        eval_windows=rows,
        thresholds=PromotionThresholds(
            min_win_rate=args.chunk_gate_min_win_rate,
            max_side_gap=args.chunk_gate_max_side_gap,
            min_ci_low=args.chunk_gate_min_ci_low,
            max_window_side_gap=args.chunk_gate_max_window_side_gap,
        ),
    )
    all_windows_accepted = all(window["decision"] == "accepted" for window in windows)
    no_windows_rejected = all(window["decision"] != "rejected" for window in windows)
    checks = dict(gate["checks"])
    checks["allWindowsAccepted"] = all_windows_accepted
    checks["noWindowsRejected"] = no_windows_rejected
    window_checks = [
        {
            **dict(window_check),
            "status": str(window.get("status", "")),
            "decision": str(window.get("decision", "")),
            "decisionReason": str(window.get("decisionReason", "")),
            "gamesPerSideCompleted": (
                int(window["progress"]["gamesPerSideCompleted"])
                if isinstance(window.get("progress"), dict)
                and "gamesPerSideCompleted" in window["progress"]
                else None
            ),
            "batchesCompleted": (
                int(window["progress"]["batchesCompleted"])
                if isinstance(window.get("progress"), dict)
                and "batchesCompleted" in window["progress"]
                else None
            ),
        }
        for window_check, window in zip(gate["windowChecks"], windows, strict=False)
    ]
    accepted = bool(no_windows_rejected and gate["passed"])
    accepted_after = candidate_checkpoint if accepted else accepted_checkpoint
    reason = "chunk_gate_passed" if accepted else "chunk_gate_failed"
    print(
        f"{log_prefix} {reason} chunk={chunk_label} "
        f"winRate={pooled.candidate_win_rate:.3f} ciLow={pooled.ci_low:.3f} "
        f"sideGap={pooled.side_gap:.3f}",
        flush=True,
    )
    return ChunkGateResult(
        enabled=True,
        accepted=accepted,
        reason=reason,
        candidate_checkpoint=candidate_checkpoint,
        accepted_before=accepted_checkpoint,
        accepted_after=accepted_after,
        seed_start_indices=list(args.chunk_gate_seed_start_indices),
        windows=windows,
        pooled=pooled,
        checks=checks,
        window_checks=window_checks,
        commands=commands,
    )


def build_gated_chunk_row(
    *,
    chunk_row: Dict[str, Any],
    gate_result: ChunkGateResult,
    replay_chunk: ReplayChunk | None = None,
    learner_checkpoint_before: LoopCheckpoint | None = None,
    learner_checkpoint_after: LoopCheckpoint | None = None,
    generator_checkpoint_before: LoopCheckpoint | None = None,
    generator_checkpoint_after: LoopCheckpoint | None = None,
) -> Dict[str, Any]:
    out = dict(chunk_row)
    out["candidateCheckpoint"] = _checkpoint_payload(gate_result.candidate_checkpoint)
    out["acceptedCheckpoint"] = _checkpoint_payload(gate_result.accepted_after)
    if learner_checkpoint_before is not None:
        out["learnerCheckpointBefore"] = _checkpoint_payload(learner_checkpoint_before)
    if learner_checkpoint_after is not None:
        out["learnerCheckpointAfter"] = _checkpoint_payload(learner_checkpoint_after)
    if generator_checkpoint_before is not None:
        out["generatorCheckpointBefore"] = _checkpoint_payload(generator_checkpoint_before)
    if generator_checkpoint_after is not None:
        out["generatorCheckpointAfter"] = _checkpoint_payload(generator_checkpoint_after)
    else:
        out["generatorCheckpointAfter"] = _checkpoint_payload(gate_result.accepted_after)
    out["promotionEligibleCheckpoint"] = out["generatorCheckpointAfter"]
    # Compatibility alias for existing artifact readers; prefer explicit generator/learner fields.
    out["latestCheckpoint"] = _checkpoint_payload(gate_result.accepted_after)
    out["generatorGate"] = _chunk_gate_payload(gate_result)
    if replay_chunk is not None:
        out["replayForTraining"] = replay_training_payload(
            replay_chunk=replay_chunk,
            gate_result=gate_result,
        )
    return out


def write_selfplay_chunk_summary(
    *,
    run_id: str,
    chunk_dir: Path,
    command_row: Dict[str, Any],
    chunk_row: Dict[str, Any],
) -> Path:
    summary_path = chunk_dir / "chunk.summary.json"
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "chunkLabel": chunk_row["chunk"],
        "commands": command_row,
        "chunk": chunk_row,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def block_update_payload(result: BlockGeneratorUpdateResult) -> Dict[str, Any]:
    return {
        "block": result.block_label,
        "index": result.block_index,
        "chunks": list(result.chunk_labels),
        "candidates": result.candidates,
        "selectedChunk": result.selected_chunk_label,
        "checkpointSelection": checkpoint_selection_payload(result.checkpoint_selection),
        "generatorGate": _chunk_gate_payload(result.generator_gate),
        "generatorCheckpointBefore": _checkpoint_payload(result.generator_gate.accepted_before),
        "generatorCheckpointAfter": _checkpoint_payload(result.generator_gate.accepted_after),
        "summary": str(result.summary_path),
    }


def write_selfplay_block_summary(
    *,
    run_id: str,
    result: BlockGeneratorUpdateResult,
) -> Path:
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "blockLabel": result.block_label,
        "commands": result.commands,
        "block": block_update_payload(result),
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result.summary_path


def run_promotion_stage(
    *,
    args: argparse.Namespace,
    eval_dir: Path,
    latest_checkpoint: LoopCheckpoint,
    incumbent_checkpoint: PoolCheckpoint,
    progress_path: Path,
) -> PromotionStageResult:
    baseline_windows = _run_eval_windows_vs_search(
        args=args,
        python_bin=args.python_bin,
        eval_dir=eval_dir,
        checkpoint=latest_checkpoint,
        progress_path=progress_path,
    )
    incumbent_windows = _run_eval_windows_vs_incumbent(
        args=args,
        python_bin=args.python_bin,
        eval_dir=eval_dir,
        checkpoint=latest_checkpoint,
        incumbent=incumbent_checkpoint,
        progress_path=progress_path,
    )
    baseline_rows = [window["row"] for window in baseline_windows]
    incumbent_rows = [window["row"] for window in incumbent_windows]
    pooled_baseline = pool_eval_rows(eval_rows=baseline_rows, opponent_policy="search")
    pooled_incumbent = pool_eval_rows(eval_rows=incumbent_rows, opponent_policy="td-search")
    promotion = _promotion_decision(
        baseline_eval=pooled_baseline,
        baseline_windows=baseline_rows,
        incumbent_eval=pooled_incumbent,
        incumbent_windows=incumbent_rows,
        args=args,
    )
    return PromotionStageResult(
        baseline_windows=baseline_windows,
        incumbent_windows=incumbent_windows,
        pooled_baseline=pooled_baseline,
        pooled_incumbent=pooled_incumbent,
        promotion=promotion,
        commands={
            "baselineVsSearch": [window["command"] for window in baseline_windows],
            "candidateVsIncumbent": [window["command"] for window in incumbent_windows],
        },
    )


def build_selfplay_loop_summary(
    *,
    args: argparse.Namespace,
    context: RunContext,
    commands: Dict[str, Any],
    chunk_rows: Sequence[Dict[str, Any]],
    promotion_stage: PromotionStageResult,
    block_rows: Sequence[Dict[str, Any]] = (),
    final_learner_checkpoint: LoopCheckpoint | None = None,
    final_generator_checkpoint: LoopCheckpoint | None = None,
    promotion_checkpoint: LoopCheckpoint | None = None,
    elapsed_minutes: float,
) -> Dict[str, Any]:
    baseline_rows = [window["row"] for window in promotion_stage.baseline_windows]
    incumbent_rows = [window["row"] for window in promotion_stage.incumbent_windows]
    final_learner_checkpoint = final_learner_checkpoint or context.latest_checkpoint
    final_generator_checkpoint = final_generator_checkpoint or context.latest_checkpoint
    promotion_checkpoint = promotion_checkpoint or final_generator_checkpoint
    return {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": context.run_id,
        "elapsedMinutes": round(elapsed_minutes, 3),
        "config": _config_payload(args),
        "finalCheckpoints": {
            "learner": _checkpoint_payload(final_learner_checkpoint),
            "generator": _checkpoint_payload(final_generator_checkpoint),
            "promotion": _checkpoint_payload(promotion_checkpoint),
        },
        "commands": commands,
        "artifacts": {
            "runDir": str(context.run_dir),
            "loopSummary": str(context.loop_summary_path),
            "progress": str(context.progress_path),
            "chunksDir": str(context.chunks_dir),
            "blocksDir": str(context.run_dir / "blocks"),
            "evalDir": str(context.eval_dir),
        },
        "chunks": list(chunk_rows),
        "blocks": list(block_rows),
        "evaluation": {
            "baselineVsSearch": build_eval_payload(
                args.eval_seed_start_indices,
                baseline_rows,
                promotion_stage.pooled_baseline,
            ),
            "candidateVsIncumbent": build_eval_payload(
                args.incumbent_eval_seed_start_indices,
                incumbent_rows,
                promotion_stage.pooled_incumbent,
            ),
        },
        "promotion": promotion_stage.promotion,
    }


def _register_manifest_promotion_if_passed(
    *,
    args: argparse.Namespace,
    context: RunContext,
    latest_checkpoint: LoopCheckpoint,
    latest_chunk_label: str | None,
    promotion_stage: PromotionStageResult,
) -> Dict[str, Any] | None:
    if not bool(promotion_stage.promotion.get("promoted")):
        return None
    if bool(getattr(args, "disable_manifest_promotion", True)):
        return None
    if latest_checkpoint.value_path is None or latest_checkpoint.opponent_path is None:
        return None

    manifest_path = getattr(args, "promotion_manifest_path", DEFAULT_MANIFEST_RELATIVE_PATH)
    checkpoint_root = getattr(args, "promotion_checkpoint_root", Path("models/td_checkpoints"))
    key = getattr(args, "promotion_manifest_key", None) or f"{context.run_id}-promoted"
    eval_artifacts = [
        window["row"].artifact
        for window in [*promotion_stage.baseline_windows, *promotion_stage.incumbent_windows]
    ]
    result = promote_checkpoint_pair(
        manifest_path=manifest_path,
        checkpoint_root=checkpoint_root,
        key=key,
        value_checkpoint=latest_checkpoint.value_path,
        opponent_checkpoint=latest_checkpoint.opponent_path,
        source_run_id=context.run_id,
        source_loop_summary=context.loop_summary_path,
        source_chunk=latest_chunk_label,
        source_eval_artifacts=eval_artifacts,
        step=latest_checkpoint.step,
        label=f"Promoted self-play loop {context.run_id}",
        set_default=True,
        add_to_opponent_pool=True,
        force=False,
    )
    print(
        "[td-loop-selfplay] manifest promotion registered "
        f"key={result['key']} manifest={manifest_path}",
        flush=True,
    )
    return {
        "key": result["key"],
        "value": str(result["value"]),
        "opponent": str(result["opponent"]),
        "manifest": str(manifest_path),
    }


def build_selfplay_terminal_report(
    *,
    context: RunContext,
    promotion_stage: PromotionStageResult,
) -> Dict[str, Any]:
    return {
        "runId": context.run_id,
        "runDir": str(context.run_dir),
        "loopSummary": str(context.loop_summary_path),
        "baselineWinRate": promotion_stage.pooled_baseline.candidate_win_rate,
        "baselineCi95": {
            "low": promotion_stage.pooled_baseline.ci_low,
            "high": promotion_stage.pooled_baseline.ci_high,
        },
        "incumbentWinRate": promotion_stage.pooled_incumbent.candidate_win_rate,
        "incumbentCi95": {
            "low": promotion_stage.pooled_incumbent.ci_low,
            "high": promotion_stage.pooled_incumbent.ci_high,
        },
        "promoted": bool(promotion_stage.promotion["promoted"]),
        "promotionReason": promotion_stage.promotion["reason"],
    }


def _resolve_warm_start(
    *, args: argparse.Namespace, promoted_pool: Sequence[PoolCheckpoint]
) -> tuple[Path, Path]:
    if args.train_warm_start_value_checkpoint is not None:
        if args.train_warm_start_opponent_checkpoint is None:
            raise SystemExit(
                "--train-warm-start-opponent-checkpoint is required when "
                "--train-warm-start-value-checkpoint is provided."
            )
        return args.train_warm_start_value_checkpoint, args.train_warm_start_opponent_checkpoint
    if args.train_warm_start_opponent_checkpoint is not None:
        raise SystemExit(
            "--train-warm-start-value-checkpoint is required when "
            "--train-warm-start-opponent-checkpoint is provided."
        )
    if not promoted_pool:
        raise SystemExit(
            "No promoted checkpoints found. Run bootstrap loop first or pass explicit "
            "--train-warm-start-value-checkpoint/--train-warm-start-opponent-checkpoint."
        )
    latest = promoted_pool[0]
    print(f"[td-loop-selfplay] using promoted warm start from {latest.run_id}")
    return latest.value_path, latest.opponent_path


def _build_collect_profiles(
    *,
    args: argparse.Namespace,
    candidate: PoolCheckpoint,
    promoted_pool: Sequence[PoolCheckpoint],
) -> List[CollectProfile]:
    shares = weighted_game_split(
        total_games=args.collect_games,
        weights={
            "selfplay": args.collect_selfplay_share,
            "pool": args.collect_pool_share,
            "search": args.collect_search_anchor_share,
        },
    )
    profiles: List[CollectProfile] = []
    if shares["selfplay"] > 0:
        profiles.append(
            CollectProfile(
                label="selfplay",
                games=shares["selfplay"],
                player_a_policy="td-search",
                player_b_policy="td-search",
                player_a_td_search=candidate,
                player_b_td_search=candidate,
            )
        )
    if shares["search"] > 0:
        profiles.append(
            CollectProfile(
                label="search-anchor",
                games=shares["search"],
                player_a_policy="td-search",
                player_b_policy="search",
                player_a_td_search=candidate,
                player_b_td_search=None,
            )
        )

    older = filter_pool_excluding_checkpoint(
        checkpoints=promoted_pool,
        value_path=candidate.value_path,
        opponent_path=candidate.opponent_path,
    )
    if shares["pool"] > 0 and older:
        labels = [f"pool-{index + 1:02d}" for index in range(len(older))]
        by_label = split_evenly(shares["pool"], labels)
        for index, opponent in enumerate(older):
            games = by_label[labels[index]]
            if games <= 0:
                continue
            profiles.append(
                CollectProfile(
                    label=labels[index],
                    games=games,
                    player_a_policy="td-search",
                    player_b_policy="td-search",
                    player_a_td_search=candidate,
                    player_b_td_search=opponent,
                )
            )
    elif shares["pool"] > 0:
        # If pool is empty, fold those games into self-play; create a self-play
        # profile when shares intentionally disabled it.
        print(
            "[td-loop-selfplay] WARNING: collect opponent pool is empty; "
            f"folding {shares['pool']} pool games into self-play. "
            "Check models/td_checkpoints/manifest.json or promoted loop summaries.",
            flush=True,
        )
        folded = False
        for index, profile in enumerate(profiles):
            if profile.label == "selfplay":
                profiles[index] = CollectProfile(
                    label=profile.label,
                    games=profile.games + shares["pool"],
                    player_a_policy=profile.player_a_policy,
                    player_b_policy=profile.player_b_policy,
                    player_a_td_search=profile.player_a_td_search,
                    player_b_td_search=profile.player_b_td_search,
                )
                folded = True
                break
        if not folded:
            profiles.append(
                CollectProfile(
                    label="selfplay",
                    games=shares["pool"],
                    player_a_policy="td-search",
                    player_b_policy="td-search",
                    player_a_td_search=candidate,
                    player_b_td_search=candidate,
                )
            )

    if not profiles:
        raise SystemExit("Collect profile plan is empty; check collect share settings.")
    planned_games = sum(profile.games for profile in profiles)
    if planned_games != args.collect_games:
        raise SystemExit(
            "Collect profile allocation mismatch. "
            f"expected={args.collect_games} planned={planned_games}"
        )
    return profiles


def _run_collect_profiles(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    profiles: Sequence[CollectProfile],
    replay_dir: Path,
    run_id: str,
    collect_value_path: Path,
    collect_opponent_path: Path,
    collect_summary_path: Path,
    heartbeat_minutes: float,
    progress_path: Path,
) -> List[Dict[str, Any]]:
    shards_dir = replay_dir / "profiles"
    shards_dir.mkdir(parents=True, exist_ok=True)
    value_paths: List[Path] = []
    opponent_paths: List[Path] = []
    summary_paths: List[Path] = []
    rows: List[Dict[str, Any]] = []

    for profile in profiles:
        row = _run_collect_profile(
            python_bin=python_bin,
            args=args,
            profile=profile,
            run_id=run_id,
            out_dir=shards_dir,
            heartbeat_minutes=heartbeat_minutes,
            progress_path=progress_path,
        )
        value_paths.append(Path(row["value"]))
        opponent_paths.append(Path(row["opponent"]))
        summary_paths.append(Path(row["summary"]))
        rows.append(row)

    concat_jsonl_files(
        inputs=value_paths,
        output=collect_value_path,
        delete_inputs_after_merge=True,
    )
    concat_jsonl_files(
        inputs=opponent_paths,
        output=collect_opponent_path,
        delete_inputs_after_merge=True,
    )
    _write_merged_collect_summary(
        summary_paths=summary_paths,
        out_path=collect_summary_path,
        value_path=collect_value_path,
        opponent_path=collect_opponent_path,
        total_expected_games=args.collect_games,
        profiles=profiles,
    )
    return rows


def _run_collect_profile(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    profile: CollectProfile,
    run_id: str,
    out_dir: Path,
    heartbeat_minutes: float,
    progress_path: Path,
) -> Dict[str, Any]:
    profile_id = profile.label
    seed_prefix = _collect_profile_seed_prefix(
        collect_seed_prefix=args.collect_seed_prefix,
        run_id=run_id,
        profile_label=profile_id,
    )
    run_label = f"{run_id}-{profile_id}"
    value_out = out_dir / f"{profile_id}.value.jsonl"
    opponent_out = out_dir / f"{profile_id}.opponent.jsonl"
    summary_out = out_dir / f"{profile_id}.summary.json"

    worker_count = min(args.collect_workers, profile.games)
    if worker_count <= 1:
        command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            profile=profile,
            games=profile.games,
            seed_prefix=seed_prefix,
            run_label=run_label,
            value_out=value_out,
            opponent_out=opponent_out,
            summary_out=summary_out,
        )
        run_step(
            name=f"collect[{run_id} {profile_id}]",
            command=command,
            heartbeat_minutes=heartbeat_minutes,
            progress_path=progress_path,
            log_prefix="[td-loop-selfplay]",
        )
        return {
            "profile": profile.label,
            "games": profile.games,
            "mode": "single",
            "command": command,
            "value": str(value_out),
            "opponent": str(opponent_out),
            "summary": str(summary_out),
        }

    shard_sizes = _split_count(profile.games, worker_count)
    shard_dir = out_dir / f"{profile_id}.shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_value_paths: List[Path] = []
    shard_opponent_paths: List[Path] = []
    shard_summary_paths: List[Path] = []
    shard_commands: List[List[str]] = []
    shard_rows: List[Dict[str, Any]] = []
    for shard_index, shard_games in enumerate(shard_sizes):
        shard_id = f"s{shard_index + 1:02d}"
        shard_value = shard_dir / f"{shard_id}.value.jsonl"
        shard_opponent = shard_dir / f"{shard_id}.opponent.jsonl"
        shard_summary = shard_dir / f"{shard_id}.summary.json"
        shard_command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            profile=profile,
            games=shard_games,
            seed_prefix=f"{seed_prefix}-{shard_id}",
            run_label=f"{run_label}-{shard_id}",
            value_out=shard_value,
            opponent_out=shard_opponent,
            summary_out=shard_summary,
        )
        shard_commands.append(shard_command)
        shard_value_paths.append(shard_value)
        shard_opponent_paths.append(shard_opponent)
        shard_summary_paths.append(shard_summary)
        shard_rows.append(
            {
                "games": shard_games,
                "value": str(shard_value),
                "opponent": str(shard_opponent),
                "summary": str(shard_summary),
            }
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_step,
                name=f"collect[{run_id} {profile_id} {index + 1}/{worker_count}]",
                command=command,
                heartbeat_minutes=heartbeat_minutes,
                progress_path=None,
                log_prefix="[td-loop-selfplay]",
            ): index
            for index, command in enumerate(shard_commands)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                raise SystemExit(
                    f"[td-loop-selfplay] failed collect profile shard: "
                    f"profile={profile_id} shard={futures[future] + 1} error={exc}"
                ) from exc

    concat_jsonl_files(
        inputs=shard_value_paths,
        output=value_out,
        delete_inputs_after_merge=True,
    )
    concat_jsonl_files(
        inputs=shard_opponent_paths,
        output=opponent_out,
        delete_inputs_after_merge=True,
    )
    _write_merged_collect_summary(
        summary_paths=shard_summary_paths,
        out_path=summary_out,
        value_path=value_out,
        opponent_path=opponent_out,
        total_expected_games=profile.games,
        profiles=[profile],
    )

    return {
        "profile": profile.label,
        "games": profile.games,
        "mode": "sharded",
        "workers": worker_count,
        "commands": shard_commands,
        "value": str(value_out),
        "opponent": str(opponent_out),
        "summary": str(summary_out),
        "shards": shard_rows,
    }


def _build_collect_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    profile: CollectProfile,
    games: int,
    seed_prefix: str,
    run_label: str,
    value_out: Path,
    opponent_out: Path,
    summary_out: Path,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.collect_td_self_play",
        "--games",
        str(games),
        "--seed-prefix",
        seed_prefix,
        "--player-a-policy",
        profile.player_a_policy,
        "--player-b-policy",
        profile.player_b_policy,
        "--search-worlds",
        str(args.collect_search_worlds),
        "--search-rollouts",
        str(args.collect_search_rollouts),
        "--search-depth",
        str(args.collect_search_depth),
        "--search-max-root-actions",
        str(args.collect_search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.collect_search_rollout_epsilon),
        "--td-worlds",
        str(args.collect_td_worlds),
        "--td-search-opponent-temperature",
        str(args.collect_td_search_opponent_temperature),
        "--run-label",
        run_label,
        "--value-out",
        str(value_out),
        "--opponent-out",
        str(opponent_out),
        "--summary-out",
        str(summary_out),
        "--progress-every-games",
        str(args.collect_progress_every_games),
    ]
    if profile.player_a_td_search is not None:
        command.extend(
            [
                "--player-a-td-search-value-checkpoint",
                str(profile.player_a_td_search.value_path),
                "--player-a-td-search-opponent-checkpoint",
                str(profile.player_a_td_search.opponent_path),
            ]
        )
    if profile.player_b_td_search is not None:
        command.extend(
            [
                "--player-b-td-search-value-checkpoint",
                str(profile.player_b_td_search.value_path),
                "--player-b-td-search-opponent-checkpoint",
                str(profile.player_b_td_search.opponent_path),
            ]
        )
    if args.collect_td_search_sample_opponent_actions:
        command.append("--td-search-sample-opponent-actions")
    return command


def _collect_profile_seed_prefix(
    *, collect_seed_prefix: str, run_id: str, profile_label: str
) -> str:
    return f"{collect_seed_prefix}-{run_id}-{profile_label}"


def _split_count(total: int, workers: int) -> List[int]:
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _write_merged_collect_summary(
    *,
    summary_paths: Sequence[Path],
    out_path: Path,
    value_path: Path,
    opponent_path: Path,
    total_expected_games: int,
    profiles: Sequence[CollectProfile],
) -> None:
    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    total_games = 0
    total_turn_weighted = 0.0
    total_value_transitions = 0
    total_opponent_samples = 0
    shard_rows: List[Dict[str, Any]] = []

    for path in summary_paths:
        payload = read_json(path, label=f"collect profile summary {path.name}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise SystemExit(f"collect profile summary missing results payload: {path}")
        shard_games = int(results["games"])
        shard_avg_turn = float(results["averageTurn"])
        shard_winners = results.get("winners")
        if not isinstance(shard_winners, dict):
            raise SystemExit(f"collect profile summary missing winners payload: {path}")
        shard_value_count = int(results["valueTransitions"])
        shard_opponent_count = int(results["opponentSamples"])

        total_games += shard_games
        total_turn_weighted += shard_avg_turn * shard_games
        total_value_transitions += shard_value_count
        total_opponent_samples += shard_opponent_count
        for key in ("PlayerA", "PlayerB", "Draw"):
            winners[key] += int(shard_winners.get(key, 0))

        shard_rows.append(
            {
                "path": str(path),
                "games": shard_games,
                "averageTurn": shard_avg_turn,
                "winners": {
                    "PlayerA": int(shard_winners.get("PlayerA", 0)),
                    "PlayerB": int(shard_winners.get("PlayerB", 0)),
                    "Draw": int(shard_winners.get("Draw", 0)),
                },
                "valueTransitions": shard_value_count,
                "opponentSamples": shard_opponent_count,
            }
        )

    if total_games != total_expected_games:
        raise SystemExit(
            "Merged collect games mismatch. "
            f"expected={total_expected_games} actual={total_games}"
        )

    merged_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": total_expected_games,
            "profiles": [
                {
                    "label": profile.label,
                    "games": profile.games,
                    "playerAPolicy": profile.player_a_policy,
                    "playerBPolicy": profile.player_b_policy,
                    "playerATdSearchCheckpoint": (
                        {
                            "runId": profile.player_a_td_search.run_id,
                            "value": str(profile.player_a_td_search.value_path),
                            "opponent": str(profile.player_a_td_search.opponent_path),
                        }
                        if profile.player_a_td_search is not None
                        else None
                    ),
                    "playerBTdSearchCheckpoint": (
                        {
                            "runId": profile.player_b_td_search.run_id,
                            "value": str(profile.player_b_td_search.value_path),
                            "opponent": str(profile.player_b_td_search.opponent_path),
                        }
                        if profile.player_b_td_search is not None
                        else None
                    ),
                }
                for profile in profiles
            ],
        },
        "results": {
            "games": total_games,
            "winners": winners,
            "averageTurn": (total_turn_weighted / float(total_games)) if total_games > 0 else 0.0,
            "valueTransitions": total_value_transitions,
            "opponentSamples": total_opponent_samples,
        },
        "artifacts": {
            "valueTransitions": str(value_path),
            "opponentSamples": str(opponent_path),
            "summary": str(out_path),
        },
        "profiles": shard_rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")


def _run_eval_windows_vs_search(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    eval_dir: Path,
    checkpoint: LoopCheckpoint,
    progress_path: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed_start_index in args.eval_seed_start_indices:
        out_path = eval_dir / f"promotion_eval.baseline.seed-{seed_start_index:06d}.json"
        command = _build_eval_command_vs_search(
            python_bin=python_bin,
            args=args,
            checkpoint=checkpoint,
            out_path=out_path,
            seed_prefix=args.eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=args.eval_workers,
            games_per_side=args.eval_games_per_side,
        )
        run_step(
            name=f"promotion-eval[baseline seed={seed_start_index}]",
            command=command,
            heartbeat_minutes=args.progress_heartbeat_minutes,
            progress_path=progress_path,
            log_prefix="[td-loop-selfplay]",
        )
        rows.append({"command": command, "row": read_eval_row(out_path, opponent_policy="search")})
    return rows


def _run_eval_windows_vs_incumbent(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    eval_dir: Path,
    checkpoint: LoopCheckpoint,
    incumbent: PoolCheckpoint,
    progress_path: Path,
) -> List[Dict[str, Any]]:
    workers = args.incumbent_eval_workers or args.eval_workers
    rows: List[Dict[str, Any]] = []
    for seed_start_index in args.incumbent_eval_seed_start_indices:
        out_path = eval_dir / f"promotion_eval.incumbent.seed-{seed_start_index:06d}.json"
        command = _build_eval_command_vs_incumbent(
            python_bin=python_bin,
            args=args,
            checkpoint=checkpoint,
            incumbent=incumbent,
            out_path=out_path,
            seed_prefix=args.incumbent_eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=workers,
            games_per_side=args.incumbent_eval_games_per_side,
        )
        run_step(
            name=f"promotion-eval[incumbent seed={seed_start_index}]",
            command=command,
            heartbeat_minutes=args.progress_heartbeat_minutes,
            progress_path=progress_path,
            log_prefix="[td-loop-selfplay]",
        )
        rows.append({"command": command, "row": read_eval_row(out_path, opponent_policy="td-search")})
    return rows


def _checkpoint_payload(checkpoint: LoopCheckpoint) -> Dict[str, Any]:
    return {
        "step": checkpoint.step,
        "value": str(checkpoint.value_path),
        "opponent": str(checkpoint.opponent_path),
    }


def checkpoint_selection_payload(result: CheckpointSelectionResult) -> Dict[str, Any]:
    return {
        "enabled": result.enabled,
        "reason": result.reason,
        "summary": str(result.summary_path),
        "selectedCheckpoint": _checkpoint_payload(result.selected_checkpoint),
        "trainedLatestCheckpoint": _checkpoint_payload(result.trained_latest_checkpoint),
        "acceptedBefore": _checkpoint_payload(result.accepted_before),
        "seedStartIndices": list(result.seed_start_indices),
        "candidates": result.candidates,
    }


def _write_checkpoint_selection_summary(
    *,
    chunk_label: str,
    result: CheckpointSelectionResult,
) -> None:
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "chunkLabel": chunk_label,
        "checkpointSelection": checkpoint_selection_payload(result),
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chunk_gate_payload(gate_result: ChunkGateResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "enabled": gate_result.enabled,
        "accepted": gate_result.accepted,
        "reason": gate_result.reason,
        "candidateCheckpoint": _checkpoint_payload(gate_result.candidate_checkpoint),
        "acceptedBefore": _checkpoint_payload(gate_result.accepted_before),
        "acceptedAfter": _checkpoint_payload(gate_result.accepted_after),
    }
    if gate_result.enabled and gate_result.pooled is not None:
        rows = [window["row"] for window in gate_result.windows]
        evaluation = build_eval_payload(
            gate_result.seed_start_indices,
            rows,
            gate_result.pooled,
        )
        windows = evaluation["windows"]
        for evaluation_window, gate_window in zip(windows, gate_result.windows, strict=False):
            evaluation_window["status"] = gate_window.get("status")
            evaluation_window["decision"] = gate_window.get("decision")
            evaluation_window["decisionReason"] = gate_window.get("decisionReason")
            progress = gate_window.get("progress")
            if isinstance(progress, dict):
                evaluation_window["gamesPerSideCompleted"] = progress.get("gamesPerSideCompleted")
                evaluation_window["batchesCompleted"] = progress.get("batchesCompleted")
            sprt = gate_window.get("sprt")
            if isinstance(sprt, dict):
                evaluation_window["sprt"] = {
                    "logLikelihoodRatio": sprt.get("logLikelihoodRatio"),
                    "acceptBoundary": sprt.get("acceptBoundary"),
                    "rejectBoundary": sprt.get("rejectBoundary"),
                }
        payload["evaluation"] = evaluation
        payload["checks"] = gate_result.checks
        payload["windowChecks"] = gate_result.window_checks
    return payload


def _require_td_search_checkpoint(checkpoint: LoopCheckpoint, *, label: str) -> None:
    if checkpoint.value_path is None or checkpoint.opponent_path is None:
        raise SystemExit(
            f"{label} requires value+opponent checkpoints: "
            f"value={checkpoint.value_path} opponent={checkpoint.opponent_path}"
        )


def _config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pythonBin": str(args.python_bin),
        "cloud": bool(args.cloud),
        "cloudVcpus": args.cloud_vcpus,
        "chunksPerLoop": args.chunks_per_loop,
        "replayRegime": REPLAY_REGIME,
        "progressHeartbeatMinutes": args.progress_heartbeat_minutes,
        "generatorUpdate": {
            "chunks": args.generator_update_chunks,
            "blockSelection": {
                "gamesPerSide": args.block_selection_games_per_side,
                "seedPrefix": args.block_selection_seed_prefix,
                "seedStartIndices": list(args.block_selection_seed_start_indices),
            },
        },
        "collect": {
            "gamesPerChunk": args.collect_games,
            "workers": args.collect_workers,
            "seedPrefix": args.collect_seed_prefix,
            "selfplayShare": args.collect_selfplay_share,
            "poolShare": args.collect_pool_share,
            "searchAnchorShare": args.collect_search_anchor_share,
            "opponentPoolSize": args.collect_opponent_pool_size,
            "search": {
                "worlds": args.collect_search_worlds,
                "rollouts": args.collect_search_rollouts,
                "depth": args.collect_search_depth,
                "maxRootActions": args.collect_search_max_root_actions,
                "rolloutEpsilon": args.collect_search_rollout_epsilon,
            },
            "tdWorlds": args.collect_td_worlds,
            "tdSearchOpponentTemperature": args.collect_td_search_opponent_temperature,
            "tdSearchSampleOpponentActions": bool(args.collect_td_search_sample_opponent_actions),
        },
        "train": {
            "stepsPerChunk": args.train_steps,
            "replayWindow": {
                "source": args.train_replay_window_source,
                "chunks": args.train_replay_window_chunks,
                "maxValueLines": args.train_replay_window_max_value_lines,
                "maxOpponentLines": args.train_replay_window_max_opponent_lines,
            },
            "valueBatchSize": args.train_value_batch_size,
            "opponentBatchSize": args.train_opponent_batch_size,
            "seed": args.train_seed,
            "hiddenDim": args.train_hidden_dim,
            "gamma": args.train_gamma,
            "valueLearningRate": args.train_value_learning_rate,
            "valueWeightDecay": args.train_value_weight_decay,
            "opponentLearningRate": args.train_opponent_learning_rate,
            "opponentWeightDecay": args.train_opponent_weight_decay,
            "maxGradNorm": args.train_max_grad_norm,
            "targetSyncInterval": args.train_target_sync_interval,
            "valueTargetMode": args.train_value_target_mode,
            "tdLambda": args.train_td_lambda,
            "saveEverySteps": args.train_save_every_steps,
            "progressEverySteps": args.train_progress_every_steps,
            "numThreads": args.train_num_threads,
            "numInteropThreads": args.train_num_interop_threads,
            "useMseLoss": bool(args.train_use_mse_loss),
            "disableValue": bool(args.train_disable_value),
            "disableOpponent": bool(args.train_disable_opponent),
        },
        "evaluation": {
            "baselineVsSearch": {
                "gamesPerSide": args.eval_games_per_side,
                "workers": args.eval_workers,
                "seedPrefix": args.eval_seed_prefix,
                "seedStartIndices": list(args.eval_seed_start_indices),
            },
            "candidateVsIncumbent": {
                "gamesPerSide": args.incumbent_eval_games_per_side,
                "workers": args.incumbent_eval_workers or args.eval_workers,
                "seedPrefix": args.incumbent_eval_seed_prefix,
                "seedStartIndices": list(args.incumbent_eval_seed_start_indices),
            },
            "chunkGate": {
                "enabled": not bool(args.disable_chunk_gate),
                "workers": args.chunk_gate_workers or args.eval_workers,
                "seedPrefix": args.chunk_gate_seed_prefix,
                "seedStartIndices": list(args.chunk_gate_seed_start_indices),
                "sprt": {
                    "h0WinRate": args.chunk_gate_h0_win_rate,
                    "h1WinRate": args.chunk_gate_h1_win_rate,
                    "alpha": args.chunk_gate_alpha,
                    "beta": args.chunk_gate_beta,
                    "batchGamesPerSide": args.chunk_gate_batch_games_per_side,
                    "maxGamesPerSide": args.chunk_gate_games_per_side,
                    "maxSideGap": args.chunk_gate_max_side_gap,
                },
                "postChecks": {
                    "minWinRate": args.chunk_gate_min_win_rate,
                    "maxSideGap": args.chunk_gate_max_side_gap,
                    "minCiLow": args.chunk_gate_min_ci_low,
                    "maxWindowSideGap": args.chunk_gate_max_window_side_gap,
                },
            },
            "checkpointSelection": {
                "enabled": True,
                "gamesPerSide": args.checkpoint_selection_games_per_side,
                "workers": (
                    args.checkpoint_selection_workers
                    or args.chunk_gate_workers
                    or args.eval_workers
                ),
                "seedPrefix": args.checkpoint_selection_seed_prefix,
                "seedStartIndices": list(args.checkpoint_selection_seed_start_indices),
            },
        },
        "promotion": {
            "chunkGate": {
                "minWinRate": args.chunk_gate_min_win_rate,
                "maxSideGap": args.chunk_gate_max_side_gap,
                "minCiLow": args.chunk_gate_min_ci_low,
                "maxWindowSideGap": args.chunk_gate_max_window_side_gap,
            },
            "baselineVsSearch": {
                "minWinRate": args.promotion_min_win_rate,
                "maxSideGap": args.promotion_max_side_gap,
                "minCiLow": args.promotion_min_ci_low,
                "maxWindowSideGap": args.promotion_max_window_side_gap,
            },
            "candidateVsIncumbent": {
                "minWinRate": args.promotion_incumbent_min_win_rate,
                "maxSideGap": args.promotion_incumbent_max_side_gap,
                "minCiLow": args.promotion_incumbent_min_ci_low,
                "maxWindowSideGap": args.promotion_incumbent_max_window_side_gap,
            },
            "manifest": {
                "enabled": not bool(getattr(args, "disable_manifest_promotion", False)),
                "manifestPath": str(
                    getattr(args, "promotion_manifest_path", DEFAULT_MANIFEST_RELATIVE_PATH)
                ),
                "checkpointRoot": str(
                    getattr(args, "promotion_checkpoint_root", Path("models/td_checkpoints"))
                ),
                "key": getattr(args, "promotion_manifest_key", None),
            },
        },
    }


def _recommended_cloud_worker_count(vcpus: int) -> int:
    return max(1, vcpus // 2)


def _apply_cloud_profile(args: argparse.Namespace) -> None:
    workers = _recommended_cloud_worker_count(args.cloud_vcpus)
    args.collect_workers = workers
    args.eval_workers = workers
    if args.incumbent_eval_workers is None:
        args.incumbent_eval_workers = workers
    if args.train_num_threads is None:
        args.train_num_threads = args.cloud_vcpus
    if args.train_num_interop_threads is None:
        args.train_num_interop_threads = 1


def _require_supported_runtime(python_bin: Path) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.12+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")
    if not python_bin.exists():
        raise SystemExit(f"--python-bin does not exist: {python_bin}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.chunks_per_loop <= 0:
        raise SystemExit("--chunks-per-loop must be > 0.")
    if args.collect_games <= 0:
        raise SystemExit("--collect-games must be > 0.")
    if args.collect_workers <= 0:
        raise SystemExit("--collect-workers must be > 0.")
    if args.train_steps <= 0:
        raise SystemExit("--train-steps must be > 0.")
    if args.train_num_threads is not None and args.train_num_threads <= 0:
        raise SystemExit("--train-num-threads must be > 0 when provided.")
    if args.train_num_interop_threads is not None and args.train_num_interop_threads <= 0:
        raise SystemExit("--train-num-interop-threads must be > 0 when provided.")
    if args.train_replay_window_chunks <= 0:
        raise SystemExit("--train-replay-window-chunks must be > 0.")
    if args.train_replay_window_source not in {"accepted", "recent"}:
        raise SystemExit("--train-replay-window-source must be 'accepted' or 'recent'.")
    if args.train_replay_window_max_value_lines < 0:
        raise SystemExit("--train-replay-window-max-value-lines must be >= 0.")
    if args.train_replay_window_max_opponent_lines < 0:
        raise SystemExit("--train-replay-window-max-opponent-lines must be >= 0.")
    if (
        args.train_value_target_mode == "td-lambda"
        and args.train_replay_window_max_value_lines > 0
    ):
        raise SystemExit(
            "--train-replay-window-max-value-lines is not supported with "
            "--train-value-target-mode td-lambda because raw line caps can split "
            "episode trajectories."
        )
    if args.train_td_lambda < 0.0 or args.train_td_lambda > 1.0:
        raise SystemExit("--train-td-lambda must be in [0, 1].")
    if args.train_disable_value:
        raise SystemExit(
            "--train-disable-value is not supported in run_td_loop_selfplay; "
            "this loop requires both value+opponent checkpoints for td-search."
        )
    if args.train_disable_opponent:
        raise SystemExit(
            "--train-disable-opponent is not supported in run_td_loop_selfplay; "
            "this loop requires both value+opponent checkpoints for td-search."
        )
    if args.generator_update_chunks <= 0:
        raise SystemExit("--generator-update-chunks must be > 0.")
    if args.block_selection_games_per_side <= 0:
        raise SystemExit("--block-selection-games-per-side must be > 0.")
    if not args.block_selection_seed_start_indices:
        raise SystemExit("--block-selection-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.block_selection_seed_start_indices):
        raise SystemExit("--block-selection-seed-start-indices must be >= 0.")

    if args.eval_games_per_side <= 0:
        raise SystemExit("--eval-games-per-side must be > 0.")
    if args.eval_workers <= 0:
        raise SystemExit("--eval-workers must be > 0.")
    if not args.eval_seed_start_indices:
        raise SystemExit("--eval-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.eval_seed_start_indices):
        raise SystemExit("--eval-seed-start-indices must be >= 0.")

    if args.incumbent_eval_games_per_side <= 0:
        raise SystemExit("--incumbent-eval-games-per-side must be > 0.")
    if args.incumbent_eval_workers is not None and args.incumbent_eval_workers <= 0:
        raise SystemExit("--incumbent-eval-workers must be > 0 when provided.")
    if not args.incumbent_eval_seed_start_indices:
        raise SystemExit("--incumbent-eval-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.incumbent_eval_seed_start_indices):
        raise SystemExit("--incumbent-eval-seed-start-indices must be >= 0.")

    if args.chunk_gate_games_per_side <= 0:
        raise SystemExit("--chunk-gate-games-per-side must be > 0.")
    if args.chunk_gate_workers is not None and args.chunk_gate_workers <= 0:
        raise SystemExit("--chunk-gate-workers must be > 0 when provided.")
    if not args.chunk_gate_seed_start_indices:
        raise SystemExit("--chunk-gate-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.chunk_gate_seed_start_indices):
        raise SystemExit("--chunk-gate-seed-start-indices must be >= 0.")
    if args.chunk_gate_h0_win_rate < 0.0 or args.chunk_gate_h0_win_rate > 1.0:
        raise SystemExit("--chunk-gate-h0-win-rate must be in [0, 1].")
    if args.chunk_gate_h1_win_rate < 0.0 or args.chunk_gate_h1_win_rate > 1.0:
        raise SystemExit("--chunk-gate-h1-win-rate must be in [0, 1].")
    if args.chunk_gate_h0_win_rate >= args.chunk_gate_h1_win_rate:
        raise SystemExit(
            "--chunk-gate-h0-win-rate must be strictly less than --chunk-gate-h1-win-rate."
        )
    if args.chunk_gate_alpha <= 0.0 or args.chunk_gate_alpha >= 1.0:
        raise SystemExit("--chunk-gate-alpha must be in (0, 1).")
    if args.chunk_gate_beta <= 0.0 or args.chunk_gate_beta >= 1.0:
        raise SystemExit("--chunk-gate-beta must be in (0, 1).")
    if args.chunk_gate_batch_games_per_side <= 0:
        raise SystemExit("--chunk-gate-batch-games-per-side must be > 0.")
    if args.chunk_gate_batch_games_per_side > args.chunk_gate_games_per_side:
        raise SystemExit(
            "--chunk-gate-batch-games-per-side must be <= --chunk-gate-games-per-side."
        )
    if args.checkpoint_selection_games_per_side <= 0:
        raise SystemExit("--checkpoint-selection-games-per-side must be > 0.")
    if args.checkpoint_selection_workers is not None and args.checkpoint_selection_workers <= 0:
        raise SystemExit("--checkpoint-selection-workers must be > 0 when provided.")
    if not args.checkpoint_selection_seed_start_indices:
        raise SystemExit("--checkpoint-selection-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.checkpoint_selection_seed_start_indices):
        raise SystemExit("--checkpoint-selection-seed-start-indices must be >= 0.")

    if args.eval_progress_every_games < 0:
        raise SystemExit("--eval-progress-every-games must be >= 0.")
    if args.eval_progress_log_minutes < 0.0:
        raise SystemExit("--eval-progress-log-minutes must be >= 0.")
    if args.eval_worker_torch_threads <= 0:
        raise SystemExit("--eval-worker-torch-threads must be > 0.")
    if args.eval_worker_torch_interop_threads <= 0:
        raise SystemExit("--eval-worker-torch-interop-threads must be > 0.")
    if args.eval_worker_blas_threads <= 0:
        raise SystemExit("--eval-worker-blas-threads must be > 0.")

    if args.collect_selfplay_share < 0 or args.collect_pool_share < 0 or args.collect_search_anchor_share < 0:
        raise SystemExit("Collect shares must be >= 0.")
    if (
        args.collect_selfplay_share
        + args.collect_pool_share
        + args.collect_search_anchor_share
        <= 0
    ):
        raise SystemExit("At least one collect share must be > 0.")
    if args.collect_opponent_pool_size <= 0:
        raise SystemExit("--collect-opponent-pool-size must be > 0.")

    for name in (
        "promotion_min_win_rate",
        "promotion_max_side_gap",
        "promotion_min_ci_low",
        "promotion_max_window_side_gap",
        "chunk_gate_min_win_rate",
        "chunk_gate_max_side_gap",
        "chunk_gate_min_ci_low",
        "chunk_gate_max_window_side_gap",
        "promotion_incumbent_min_win_rate",
        "promotion_incumbent_max_side_gap",
        "promotion_incumbent_min_ci_low",
        "promotion_incumbent_max_window_side_gap",
    ):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1].")

    if args.progress_heartbeat_minutes < 0.0:
        raise SystemExit("--progress-heartbeat-minutes must be >= 0.")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower()).strip("-") or "run"


if __name__ == "__main__":
    raise SystemExit(main())
