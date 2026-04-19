from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.td_loop_common import (
    LoopCheckpoint,
)
from scripts.td_loop_common import (
    build_train_command as build_train_command_common,
)
from scripts.td_loop_common import (
    checkpoints_from_train_summary as checkpoints_from_train_summary_common,
)
from scripts.td_loop_common import (
    concat_jsonl_files as concat_jsonl_files_common,
)
from scripts.td_loop_common import (
    join_command as join_command_common,
)
from scripts.td_loop_common import (
    read_json as read_json_common,
)
from scripts.td_loop_common import (
    run_step as run_step_common,
)
from scripts.td_loop_common import (
    select_latest_checkpoint as select_latest_checkpoint_common,
)
from scripts.td_loop_common import (
    write_progress as write_progress_common,
)

REPLAY_REGIME = "chunk-local"


@dataclass(frozen=True)
class EvalRow:
    artifact: Path
    opponent_policy: str
    candidate_win_rate: float
    ci_low: float
    ci_high: float
    side_gap: float
    candidate_wins: int
    opponent_wins: int
    draws: int
    total_games: int
    candidate_win_rate_as_player_a: float
    candidate_win_rate_as_player_b: float


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    chunks_dir: Path
    eval_dir: Path
    loop_summary_path: Path
    progress_path: Path
    loop_started: float
    warm_value: Path | None
    warm_opponent: Path | None


@dataclass(frozen=True)
class ChunkExecutionResult:
    chunk_label: str
    latest_checkpoint: LoopCheckpoint
    command_row: Dict[str, Any]
    chunk_row: Dict[str, Any]


@dataclass(frozen=True)
class PromotionStageResult:
    eval_rows: List[EvalRow]
    promotion_eval_commands: List[List[str]]
    promotion_eval_artifacts: List[str]
    pooled_eval_row: EvalRow
    promotion: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TD loop with chunked collect/train, followed by fixed-size "
            "promotion eval windows and pooled promotion checks."
        )
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run stage scripts.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/td_loops"),
        help="Root output directory for loop artifacts.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="td-loop",
        help="Run label used in artifact naming.",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Apply preset worker/thread profile for cloud machines.",
    )
    parser.add_argument(
        "--cloud-vcpus",
        type=int,
        default=8,
        choices=(8, 16, 32),
        help="vCPU preset used when --cloud is set (8|16|32).",
    )

    parser.add_argument(
        "--chunks-per-loop",
        type=int,
        default=3,
        help="Number of collect/train chunks before promotion eval.",
    )
    parser.add_argument(
        "--chunks-per-gate",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--collect-games", type=int, default=1200)
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="Parallel collection shards. 1 disables sharding.",
    )
    parser.add_argument("--collect-seed-prefix", type=str, default="td-loop-collect")
    parser.add_argument("--collect-player-a-policy", type=str, default="search")
    parser.add_argument("--collect-player-b-policy", type=str, default="search")
    parser.add_argument("--collect-progress-every-games", type=int, default=50)
    parser.add_argument("--collect-search-worlds", type=int, default=6)
    parser.add_argument("--collect-search-rollouts", type=int, default=1)
    parser.add_argument("--collect-search-depth", type=int, default=14)
    parser.add_argument("--collect-search-max-root-actions", type=int, default=6)
    parser.add_argument("--collect-search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--collect-td-value-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-worlds", type=int, default=8)
    parser.add_argument("--collect-td-search-value-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-search-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--collect-td-search-sample-opponent-actions", action="store_true")

    parser.add_argument("--train-steps", type=int, default=20000)
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
        default="td0",
    )
    parser.add_argument("--train-td-lambda", type=float, default=0.7)
    parser.add_argument("--train-use-mse-loss", action="store_true")
    parser.add_argument("--train-disable-value", action="store_true")
    parser.add_argument("--train-disable-opponent", action="store_true")
    parser.add_argument("--train-warm-start-value-checkpoint", type=Path, default=None)
    parser.add_argument("--train-warm-start-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--train-save-every-steps", type=int, default=1000)
    parser.add_argument("--train-progress-every-steps", type=int, default=50)
    parser.add_argument(
        "--train-num-threads",
        type=int,
        default=None,
        help=(
            "Optional torch intra-op CPU thread count for scripts.train_td. "
            "If --cloud is set and omitted, defaults to --cloud-vcpus."
        ),
    )
    parser.add_argument(
        "--train-num-interop-threads",
        type=int,
        default=None,
        help=(
            "Optional torch inter-op CPU thread count for scripts.train_td. "
            "If --cloud is set and omitted, defaults to 1."
        ),
    )

    parser.add_argument(
        "--eval-candidate-policy",
        type=str,
        choices=("td-search", "td-value"),
        default="td-search",
    )
    parser.add_argument(
        "--eval-opponent-policy",
        type=str,
        choices=("random", "heuristic", "search"),
        default="search",
    )
    parser.add_argument("--eval-games-per-side", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--eval-seed-prefix", type=str, default="td-loop-eval")
    parser.add_argument("--eval-seed-start-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--eval-seed-start-indices",
        type=int,
        nargs="+",
        default=[0, 10000],
        help="Seed index windows for promotion evals; results are pooled for promotion.",
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

    parser.add_argument("--promotion-min-win-rate", type=float, default=0.55)
    parser.add_argument("--promotion-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-min-ci-low", type=float, default=0.5)
    parser.add_argument(
        "--promotion-max-window-side-gap",
        type=float,
        default=0.10,
        help="Each eval window must stay at or below this side gap.",
    )

    parser.add_argument(
        "--progress-heartbeat-minutes",
        type=float,
        default=30.0,
        help="Emit parent stage heartbeat every N minutes (0 disables).",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunks_per_gate is not None:
        args.chunks_per_loop = args.chunks_per_gate
    if args.eval_seed_start_index is not None:
        args.eval_seed_start_indices = [args.eval_seed_start_index]
    if args.cloud:
        _apply_cloud_profile(args)

    _require_supported_runtime(args.python_bin)
    _validate_args(args)
    return run_td_loop(args)


def run_td_loop(args: argparse.Namespace) -> int:
    context = initialize_td_loop_run(args)
    commands: Dict[str, Any] = {"chunks": []}
    chunk_rows: List[Dict[str, Any]] = []

    latest_checkpoint: LoopCheckpoint | None = None
    warm_value = context.warm_value
    warm_opponent = context.warm_opponent

    for chunk_index in range(1, args.chunks_per_loop + 1):
        chunk_result = run_td_loop_chunk(
            args=args,
            run_id=context.run_id,
            chunk_index=chunk_index,
            chunks_dir=context.chunks_dir,
            warm_value=warm_value,
            warm_opponent=warm_opponent,
            progress_path=context.progress_path,
        )
        latest_checkpoint = chunk_result.latest_checkpoint
        if latest_checkpoint.value_path is not None:
            warm_value = latest_checkpoint.value_path
        if latest_checkpoint.opponent_path is not None:
            warm_opponent = latest_checkpoint.opponent_path
        commands["chunks"].append(chunk_result.command_row)
        chunk_rows.append(chunk_result.chunk_row)

    if latest_checkpoint is None:
        raise SystemExit("No latest checkpoint available after chunk training.")

    promotion_stage = run_td_loop_promotion_stage(
        args=args,
        eval_dir=context.eval_dir,
        latest_checkpoint=latest_checkpoint,
        progress_path=context.progress_path,
    )
    commands["promotionEvals"] = promotion_stage.promotion_eval_commands

    loop_elapsed_minutes = (time.perf_counter() - context.loop_started) / 60.0
    payload = build_td_loop_summary(
        args=args,
        context=context,
        commands=commands,
        chunk_rows=chunk_rows,
        promotion_stage=promotion_stage,
        elapsed_minutes=loop_elapsed_minutes,
    )
    context.loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(build_td_loop_terminal_report(context=context, promotion_stage=promotion_stage), indent=2))
    return 0


def initialize_td_loop_run(args: argparse.Namespace) -> RunContext:
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
        loop_started=time.perf_counter(),
        warm_value=args.train_warm_start_value_checkpoint,
        warm_opponent=args.train_warm_start_opponent_checkpoint,
    )


def run_td_loop_chunk(
    *,
    args: argparse.Namespace,
    run_id: str,
    chunk_index: int,
    chunks_dir: Path,
    warm_value: Path | None,
    warm_opponent: Path | None,
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

    collect_stage = _run_collect_stage(
        python_bin=args.python_bin,
        args=args,
        replay_dir=replay_dir,
        collect_value_path=collect_value_path,
        collect_opponent_path=collect_opponent_path,
        collect_summary_path=collect_summary_path,
        run_id=f"{run_id}-{chunk_label}",
        seed_prefix=f"{args.collect_seed_prefix}-{chunk_label}",
        heartbeat_minutes=args.progress_heartbeat_minutes,
    )
    train_command = _build_train_command(
        python_bin=args.python_bin,
        args=args,
        value_replay=collect_value_path,
        opponent_replay=collect_opponent_path,
        train_summary_path=train_summary_path,
        train_checkpoint_root=train_checkpoint_root,
        run_id=f"{run_id}-{chunk_label}",
        warm_start_value=warm_value,
        warm_start_opponent=warm_opponent,
    )
    _run_step(
        name=f"train[{chunk_label}]",
        command=train_command,
        heartbeat_minutes=args.progress_heartbeat_minutes,
        progress_path=progress_path,
    )
    train_summary = _read_json(train_summary_path, label=f"train summary {chunk_label}")
    checkpoints = _checkpoints_from_train_summary(train_summary)
    latest_checkpoint = _select_latest_checkpoint(
        checkpoints=checkpoints,
        candidate_policy=args.eval_candidate_policy,
    )
    return ChunkExecutionResult(
        chunk_label=chunk_label,
        latest_checkpoint=latest_checkpoint,
        command_row={
            "chunk": chunk_label,
            "collect": collect_stage,
            "train": train_command,
        },
        chunk_row={
            "chunk": chunk_label,
            "replayRegime": REPLAY_REGIME,
            "collectSummary": str(collect_summary_path),
            "trainSummary": str(train_summary_path),
            "latestCheckpoint": {
                "step": latest_checkpoint.step,
                "value": str(latest_checkpoint.value_path)
                if latest_checkpoint.value_path is not None
                else None,
                "opponent": str(latest_checkpoint.opponent_path)
                if latest_checkpoint.opponent_path is not None
                else None,
            },
        },
    )


def run_td_loop_promotion_stage(
    *,
    args: argparse.Namespace,
    eval_dir: Path,
    latest_checkpoint: LoopCheckpoint,
    progress_path: Path,
) -> PromotionStageResult:
    eval_rows: List[EvalRow] = []
    promotion_eval_commands: List[List[str]] = []
    promotion_eval_artifacts: List[str] = []
    for index, seed_start_index in enumerate(args.eval_seed_start_indices, start=1):
        eval_artifact = eval_dir / f"promotion_eval.seed-{seed_start_index:06d}.json"
        eval_command = _build_eval_command(
            python_bin=args.python_bin,
            args=args,
            checkpoint=latest_checkpoint,
            opponent_policy=args.eval_opponent_policy,
            seed_prefix=args.eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=args.eval_workers,
            games_per_side=args.eval_games_per_side,
            out_path=eval_artifact,
        )
        promotion_eval_commands.append(eval_command)
        promotion_eval_artifacts.append(str(eval_artifact))
        _run_step(
            name=f"promotion-eval[{index}/{len(args.eval_seed_start_indices)} seed={seed_start_index}]",
            command=eval_command,
            heartbeat_minutes=args.progress_heartbeat_minutes,
            progress_path=progress_path,
        )
        eval_rows.append(_read_eval_row(path=eval_artifact, opponent_policy=args.eval_opponent_policy))

    pooled_eval_row = _pool_eval_rows(eval_rows=eval_rows, opponent_policy=args.eval_opponent_policy)
    promotion = _promotion_decision(eval_row=pooled_eval_row, eval_windows=eval_rows, args=args)
    return PromotionStageResult(
        eval_rows=eval_rows,
        promotion_eval_commands=promotion_eval_commands,
        promotion_eval_artifacts=promotion_eval_artifacts,
        pooled_eval_row=pooled_eval_row,
        promotion=promotion,
    )


def build_td_loop_summary(
    *,
    args: argparse.Namespace,
    context: RunContext,
    commands: Dict[str, Any],
    chunk_rows: Sequence[Dict[str, Any]],
    promotion_stage: PromotionStageResult,
    elapsed_minutes: float,
) -> Dict[str, Any]:
    return {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": context.run_id,
        "elapsedMinutes": round(elapsed_minutes, 3),
        "config": _config_payload(args),
        "commands": commands,
        "artifacts": {
            "runDir": str(context.run_dir),
            "loopSummary": str(context.loop_summary_path),
            "progress": str(context.progress_path),
            "chunksDir": str(context.chunks_dir),
            "evalDir": str(context.eval_dir),
            "promotionEvalArtifact": promotion_stage.promotion_eval_artifacts[0],
            "promotionEvalArtifacts": promotion_stage.promotion_eval_artifacts,
        },
        "chunks": list(chunk_rows),
        "evaluation": {
            "opponentPolicy": args.eval_opponent_policy,
            "windows": [
                {
                    "seedStartIndex": seed_start_index,
                    "artifact": str(row.artifact),
                    "candidateWinRate": row.candidate_win_rate,
                    "candidateWinRateCi95": {"low": row.ci_low, "high": row.ci_high},
                    "candidateWinRateAsPlayerA": row.candidate_win_rate_as_player_a,
                    "candidateWinRateAsPlayerB": row.candidate_win_rate_as_player_b,
                    "sideGap": row.side_gap,
                    "candidateWins": row.candidate_wins,
                    "opponentWins": row.opponent_wins,
                    "draws": row.draws,
                    "totalGames": row.total_games,
                }
                for seed_start_index, row in zip(
                    args.eval_seed_start_indices, promotion_stage.eval_rows, strict=False
                )
            ],
            "pooled": {
                "candidateWinRate": promotion_stage.pooled_eval_row.candidate_win_rate,
                "candidateWinRateCi95": {
                    "low": promotion_stage.pooled_eval_row.ci_low,
                    "high": promotion_stage.pooled_eval_row.ci_high,
                },
                "candidateWinRateAsPlayerA": promotion_stage.pooled_eval_row.candidate_win_rate_as_player_a,
                "candidateWinRateAsPlayerB": promotion_stage.pooled_eval_row.candidate_win_rate_as_player_b,
                "sideGap": promotion_stage.pooled_eval_row.side_gap,
                "candidateWins": promotion_stage.pooled_eval_row.candidate_wins,
                "opponentWins": promotion_stage.pooled_eval_row.opponent_wins,
                "draws": promotion_stage.pooled_eval_row.draws,
                "totalGames": promotion_stage.pooled_eval_row.total_games,
            },
        },
        "promotion": promotion_stage.promotion,
    }


def build_td_loop_terminal_report(
    *,
    context: RunContext,
    promotion_stage: PromotionStageResult,
) -> Dict[str, Any]:
    return {
        "runId": context.run_id,
        "runDir": str(context.run_dir),
        "loopSummary": str(context.loop_summary_path),
        "promotionOpponent": promotion_stage.pooled_eval_row.opponent_policy,
        "evalWindows": len(promotion_stage.eval_rows),
        "candidateWinRate": promotion_stage.pooled_eval_row.candidate_win_rate,
        "candidateWinRateCi95": {
            "low": promotion_stage.pooled_eval_row.ci_low,
            "high": promotion_stage.pooled_eval_row.ci_high,
        },
        "sideGap": promotion_stage.pooled_eval_row.side_gap,
        "promoted": bool(promotion_stage.promotion["promoted"]),
        "promotionReason": promotion_stage.promotion["reason"],
    }


def _build_collect_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
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
        args.collect_player_a_policy,
        "--player-b-policy",
        args.collect_player_b_policy,
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
    if args.collect_td_value_checkpoint is not None:
        command.extend(["--td-value-checkpoint", str(args.collect_td_value_checkpoint)])
    if args.collect_td_search_value_checkpoint is not None:
        command.extend(
            ["--td-search-value-checkpoint", str(args.collect_td_search_value_checkpoint)]
        )
    if args.collect_td_search_opponent_checkpoint is not None:
        command.extend(
            [
                "--td-search-opponent-checkpoint",
                str(args.collect_td_search_opponent_checkpoint),
            ]
        )
    if args.collect_td_search_sample_opponent_actions:
        command.append("--td-search-sample-opponent-actions")
    return command


def _run_collect_stage(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    replay_dir: Path,
    collect_value_path: Path,
    collect_opponent_path: Path,
    collect_summary_path: Path,
    run_id: str,
    seed_prefix: str,
    heartbeat_minutes: float,
) -> Dict[str, Any]:
    if args.collect_workers == 1:
        command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            games=args.collect_games,
            seed_prefix=seed_prefix,
            run_label=run_id,
            value_out=collect_value_path,
            opponent_out=collect_opponent_path,
            summary_out=collect_summary_path,
        )
        _run_step(name=f"collect[{run_id}]", command=command, heartbeat_minutes=heartbeat_minutes)
        return {"mode": "single", "commands": [command]}

    worker_count = min(args.collect_workers, args.collect_games)
    shard_sizes = _split_count(args.collect_games, worker_count)
    shards_dir = replay_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_commands: List[List[str]] = []
    shard_value_paths: List[Path] = []
    shard_opponent_paths: List[Path] = []
    shard_summary_paths: List[Path] = []

    for shard_index, shard_games in enumerate(shard_sizes):
        shard_id = f"s{shard_index + 1:02d}"
        shard_value = shards_dir / f"{shard_id}.value.jsonl"
        shard_opponent = shards_dir / f"{shard_id}.opponent.jsonl"
        shard_summary = shards_dir / f"{shard_id}.summary.json"
        shard_command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            games=shard_games,
            seed_prefix=f"{seed_prefix}-{shard_id}",
            run_label=f"{run_id}-{shard_id}",
            value_out=shard_value,
            opponent_out=shard_opponent,
            summary_out=shard_summary,
        )
        shard_commands.append(shard_command)
        shard_value_paths.append(shard_value)
        shard_opponent_paths.append(shard_opponent)
        shard_summary_paths.append(shard_summary)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_step,
                name=f"collect[{run_id} {index + 1}/{worker_count}]",
                command=command,
                heartbeat_minutes=heartbeat_minutes,
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
                raise SystemExit(f"[td-loop] failed collect shard: {exc}") from exc

    _concat_jsonl_files(
        inputs=shard_value_paths,
        output=collect_value_path,
        delete_inputs_after_merge=True,
    )
    _concat_jsonl_files(
        inputs=shard_opponent_paths,
        output=collect_opponent_path,
        delete_inputs_after_merge=True,
    )
    _write_merged_collect_summary(
        args=args,
        shard_summaries=shard_summary_paths,
        merged_summary_path=collect_summary_path,
        merged_value_path=collect_value_path,
        merged_opponent_path=collect_opponent_path,
    )

    return {
        "mode": "sharded",
        "workers": worker_count,
        "commands": shard_commands,
        "shards": [
            {
                "value": str(shard_value_paths[index]),
                "opponent": str(shard_opponent_paths[index]),
                "summary": str(shard_summary_paths[index]),
                "games": shard_sizes[index],
            }
            for index in range(worker_count)
        ],
        "merged": {
            "value": str(collect_value_path),
            "opponent": str(collect_opponent_path),
            "summary": str(collect_summary_path),
        },
    }


def _concat_jsonl_files(
    *,
    inputs: Sequence[Path],
    output: Path,
    delete_inputs_after_merge: bool = False,
) -> None:
    concat_jsonl_files_common(
        inputs=inputs,
        output=output,
        delete_inputs_after_merge=delete_inputs_after_merge,
    )


def _write_merged_collect_summary(
    *,
    args: argparse.Namespace,
    shard_summaries: Sequence[Path],
    merged_summary_path: Path,
    merged_value_path: Path,
    merged_opponent_path: Path,
) -> None:
    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    total_games = 0
    total_turn_weighted = 0.0
    total_value_transitions = 0
    total_opponent_samples = 0
    shard_rows: List[Dict[str, Any]] = []

    for shard_path in shard_summaries:
        payload = _read_json(shard_path, label=f"collect shard summary {shard_path.name}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise SystemExit(f"collect shard summary missing results payload: {shard_path}")
        shard_games = int(results["games"])
        shard_avg_turn = float(results["averageTurn"])
        shard_winners = results.get("winners")
        if not isinstance(shard_winners, dict):
            raise SystemExit(f"collect shard summary missing winners payload: {shard_path}")
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
                "path": str(shard_path),
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

    if total_games != args.collect_games:
        raise SystemExit(
            "Merged collect games mismatch. "
            f"expected={args.collect_games} actual={total_games}"
        )

    merged_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.collect_games,
            "seedPrefix": args.collect_seed_prefix,
            "playerAPolicy": args.collect_player_a_policy,
            "playerBPolicy": args.collect_player_b_policy,
            "collectWorkers": args.collect_workers,
            "search": {
                "worlds": args.collect_search_worlds,
                "rollouts": args.collect_search_rollouts,
                "depth": args.collect_search_depth,
                "maxRootActions": args.collect_search_max_root_actions,
                "rolloutEpsilon": args.collect_search_rollout_epsilon,
            },
        },
        "results": {
            "games": total_games,
            "winners": winners,
            "averageTurn": (total_turn_weighted / float(total_games)) if total_games > 0 else 0.0,
            "valueTransitions": total_value_transitions,
            "opponentSamples": total_opponent_samples,
        },
        "artifacts": {
            "valueTransitions": str(merged_value_path),
            "opponentSamples": str(merged_opponent_path),
            "summary": str(merged_summary_path),
        },
        "shards": shard_rows,
    }
    merged_summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged_summary_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")


def _split_count(total: int, workers: int) -> List[int]:
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _build_train_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    value_replay: Path,
    opponent_replay: Path,
    train_summary_path: Path,
    train_checkpoint_root: Path,
    run_id: str,
    warm_start_value: Path | None,
    warm_start_opponent: Path | None,
) -> List[str]:
    return build_train_command_common(
        python_bin=python_bin,
        args=args,
        value_replay=value_replay,
        opponent_replay=opponent_replay,
        train_summary_path=train_summary_path,
        train_checkpoint_root=train_checkpoint_root,
        run_id=run_id,
        warm_start_value=warm_start_value,
        warm_start_opponent=warm_start_opponent,
    )


def _build_eval_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    checkpoint: LoopCheckpoint,
    opponent_policy: str,
    seed_prefix: str,
    seed_start_index: int,
    workers: int,
    games_per_side: int,
    out_path: Path,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.eval_suite",
        "--mode",
        "certify",
        "--games-per-side",
        str(games_per_side),
        "--workers",
        str(workers),
        "--seed-prefix",
        seed_prefix,
        "--seed-start-index",
        str(seed_start_index),
        "--candidate-policy",
        args.eval_candidate_policy,
        "--opponent-policy",
        opponent_policy,
        "--search-worlds",
        str(args.eval_search_worlds),
        "--search-rollouts",
        str(args.eval_search_rollouts),
        "--search-depth",
        str(args.eval_search_depth),
        "--search-max-root-actions",
        str(args.eval_search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.eval_search_rollout_epsilon),
        "--td-worlds",
        str(args.eval_td_worlds),
        "--progress-every-games",
        str(args.eval_progress_every_games),
        "--progress-log-minutes",
        str(args.eval_progress_log_minutes),
        "--worker-torch-threads",
        str(args.eval_worker_torch_threads),
        "--worker-torch-interop-threads",
        str(args.eval_worker_torch_interop_threads),
        "--worker-blas-threads",
        str(args.eval_worker_blas_threads),
        "--out",
        str(out_path),
    ]

    if args.eval_candidate_policy == "td-search":
        if checkpoint.value_path is None or checkpoint.opponent_path is None:
            raise SystemExit("td-search evaluation requires both value and opponent checkpoints.")
        command.extend(["--td-search-value-checkpoint", str(checkpoint.value_path)])
        command.extend(["--td-search-opponent-checkpoint", str(checkpoint.opponent_path)])
        command.extend(
            [
                "--td-search-opponent-temperature",
                str(args.eval_td_search_opponent_temperature),
            ]
        )
        if args.eval_td_search_sample_opponent_actions:
            command.append("--td-search-sample-opponent-actions")
    else:
        if checkpoint.value_path is None:
            raise SystemExit("td-value evaluation requires a value checkpoint.")
        command.extend(["--td-value-checkpoint", str(checkpoint.value_path)])

    return command


def _checkpoints_from_train_summary(payload: Dict[str, Any]) -> List[LoopCheckpoint]:
    return checkpoints_from_train_summary_common(payload)


def _select_latest_checkpoint(
    *,
    checkpoints: Sequence[LoopCheckpoint],
    candidate_policy: str,
) -> LoopCheckpoint:
    return select_latest_checkpoint_common(
        checkpoints=checkpoints,
        candidate_policy=candidate_policy,
    )


def _read_eval_row(*, path: Path, opponent_policy: str) -> EvalRow:
    payload = _read_json(path, label=f"eval artifact {path.name}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Eval artifact is missing results payload: {path}")
    return EvalRow(
        artifact=path,
        opponent_policy=opponent_policy,
        candidate_win_rate=float(results["candidateWinRate"]),
        ci_low=float(results["candidateWinRateCi95"]["low"]),
        ci_high=float(results["candidateWinRateCi95"]["high"]),
        side_gap=float(results["sideGap"]),
        candidate_wins=int(results["candidateWins"]),
        opponent_wins=int(results["opponentWins"]),
        draws=int(results["draws"]),
        total_games=int(results["totalGames"]),
        candidate_win_rate_as_player_a=float(results["candidateWinRateAsPlayerA"]),
        candidate_win_rate_as_player_b=float(results["candidateWinRateAsPlayerB"]),
    )


def _pool_eval_rows(*, eval_rows: Sequence[EvalRow], opponent_policy: str) -> EvalRow:
    if not eval_rows:
        raise SystemExit("No eval rows to pool for promotion decision.")
    total_games = sum(row.total_games for row in eval_rows)
    if total_games <= 0:
        raise SystemExit("Pooled eval total games must be > 0.")

    candidate_wins = sum(row.candidate_wins for row in eval_rows)
    opponent_wins = sum(row.opponent_wins for row in eval_rows)
    draws = sum(row.draws for row in eval_rows)
    candidate_win_rate = float(candidate_wins) / float(total_games)
    ci_low, ci_high = _wilson_interval_95(successes=candidate_wins, trials=total_games)

    weighted_rate_a = (
        sum(row.candidate_win_rate_as_player_a * row.total_games for row in eval_rows)
        / float(total_games)
    )
    weighted_rate_b = (
        sum(row.candidate_win_rate_as_player_b * row.total_games for row in eval_rows)
        / float(total_games)
    )

    return EvalRow(
        artifact=Path("pooled"),
        opponent_policy=opponent_policy,
        candidate_win_rate=candidate_win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        side_gap=abs(weighted_rate_a - weighted_rate_b),
        candidate_wins=candidate_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        total_games=total_games,
        candidate_win_rate_as_player_a=weighted_rate_a,
        candidate_win_rate_as_player_b=weighted_rate_b,
    )


def _promotion_decision(*, eval_row: EvalRow, eval_windows: Sequence[EvalRow], args: argparse.Namespace) -> Dict[str, Any]:
    window_checks = [
        {
            "artifact": str(row.artifact),
            "sideGap": row.side_gap,
            "maxWindowSideGap": row.side_gap <= args.promotion_max_window_side_gap,
        }
        for row in eval_windows
    ]
    checks = {
        "minWinRate": eval_row.candidate_win_rate >= args.promotion_min_win_rate,
        "maxSideGap": eval_row.side_gap <= args.promotion_max_side_gap,
        "minCiLow": eval_row.ci_low >= args.promotion_min_ci_low,
        "maxWindowSideGap": all(window_check["maxWindowSideGap"] for window_check in window_checks),
    }
    promoted = bool(all(checks.values()))
    return {
        "promoted": promoted,
        "checks": checks,
        "windowChecks": window_checks,
        "reason": (
            "pooled_eval_passed"
            if promoted
            else "pooled_eval_failed"
        ),
    }


def _config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pythonBin": str(args.python_bin),
        "cloud": bool(args.cloud),
        "cloudVcpus": args.cloud_vcpus,
        "chunksPerLoop": args.chunks_per_loop,
        "replayRegime": REPLAY_REGIME,
        "progressHeartbeatMinutes": args.progress_heartbeat_minutes,
        "collect": {
            "gamesPerChunk": args.collect_games,
            "workers": args.collect_workers,
            "seedPrefix": args.collect_seed_prefix,
            "playerAPolicy": args.collect_player_a_policy,
            "playerBPolicy": args.collect_player_b_policy,
            "search": {
                "worlds": args.collect_search_worlds,
                "rollouts": args.collect_search_rollouts,
                "depth": args.collect_search_depth,
                "maxRootActions": args.collect_search_max_root_actions,
                "rolloutEpsilon": args.collect_search_rollout_epsilon,
            },
            "progressEveryGames": args.collect_progress_every_games,
        },
        "train": {
            "stepsPerChunk": args.train_steps,
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
            "candidatePolicy": args.eval_candidate_policy,
            "opponentPolicy": args.eval_opponent_policy,
            "gamesPerSide": args.eval_games_per_side,
            "workers": args.eval_workers,
            "seedPrefix": args.eval_seed_prefix,
            "seedStartIndices": list(args.eval_seed_start_indices),
            "progressEveryGames": args.eval_progress_every_games,
            "progressLogMinutes": args.eval_progress_log_minutes,
            "workerTorchThreads": args.eval_worker_torch_threads,
            "workerTorchInteropThreads": args.eval_worker_torch_interop_threads,
            "workerBlasThreads": args.eval_worker_blas_threads,
            "search": {
                "worlds": args.eval_search_worlds,
                "rollouts": args.eval_search_rollouts,
                "depth": args.eval_search_depth,
                "maxRootActions": args.eval_search_max_root_actions,
                "rolloutEpsilon": args.eval_search_rollout_epsilon,
            },
            "tdWorlds": args.eval_td_worlds,
            "tdSearchOpponentTemperature": args.eval_td_search_opponent_temperature,
            "tdSearchSampleOpponentActions": bool(args.eval_td_search_sample_opponent_actions),
        },
        "promotion": {
            "minWinRate": args.promotion_min_win_rate,
            "maxSideGap": args.promotion_max_side_gap,
            "minCiLow": args.promotion_min_ci_low,
            "maxWindowSideGap": args.promotion_max_window_side_gap,
        },
    }


def _require_supported_runtime(python_bin: Path) -> None:
    if sys.version_info < (3, 12):
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
    if args.train_td_lambda < 0.0 or args.train_td_lambda > 1.0:
        raise SystemExit("--train-td-lambda must be in [0, 1].")

    if args.eval_games_per_side <= 0:
        raise SystemExit("--eval-games-per-side must be > 0.")
    if args.eval_workers <= 0:
        raise SystemExit("--eval-workers must be > 0.")
    if not args.eval_seed_start_indices:
        raise SystemExit("--eval-seed-start-indices must contain at least one value.")
    if any(seed < 0 for seed in args.eval_seed_start_indices):
        raise SystemExit("--eval-seed-start-indices must be >= 0.")
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

    if args.promotion_min_win_rate < 0.0 or args.promotion_min_win_rate > 1.0:
        raise SystemExit("--promotion-min-win-rate must be in [0, 1].")
    if args.promotion_max_side_gap < 0.0 or args.promotion_max_side_gap > 1.0:
        raise SystemExit("--promotion-max-side-gap must be in [0, 1].")
    if args.promotion_min_ci_low < 0.0 or args.promotion_min_ci_low > 1.0:
        raise SystemExit("--promotion-min-ci-low must be in [0, 1].")
    if args.promotion_max_window_side_gap < 0.0 or args.promotion_max_window_side_gap > 1.0:
        raise SystemExit("--promotion-max-window-side-gap must be in [0, 1].")

    if args.progress_heartbeat_minutes < 0.0:
        raise SystemExit("--progress-heartbeat-minutes must be >= 0.")

    if args.train_disable_value and args.train_disable_opponent:
        raise SystemExit("At least one of value/opponent training must be enabled.")


def _run_step(
    *,
    name: str,
    command: Sequence[str],
    heartbeat_minutes: float = 0.0,
    progress_path: Path | None = None,
) -> None:
    run_step_common(
        name=name,
        command=command,
        heartbeat_minutes=heartbeat_minutes,
        progress_path=progress_path,
        log_prefix="[td-loop]",
    )


def _write_progress(path: Path, payload: Dict[str, Any]) -> None:
    write_progress_common(path, payload)


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    return read_json_common(path, label=label)


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "run"


def _join_command(parts: Sequence[str]) -> str:
    return join_command_common(parts)


def _recommended_cloud_worker_count(vcpus: int) -> int:
    # Keep workers moderate for eval inference to reduce oversubscription.
    return max(1, vcpus // 2)


def _wilson_interval_95(*, successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        raise SystemExit("Wilson interval requires trials > 0.")
    p_hat = float(successes) / float(trials)
    z = 1.959963984540054
    z2_over_n = (z * z) / float(trials)
    denom = 1.0 + z2_over_n
    center = (p_hat + (z * z) / (2.0 * float(trials))) / denom
    margin = (
        z
        * sqrt((p_hat * (1.0 - p_hat) / float(trials)) + ((z * z) / (4.0 * float(trials) * float(trials))))
        / denom
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def _apply_cloud_profile(args: argparse.Namespace) -> None:
    workers = _recommended_cloud_worker_count(args.cloud_vcpus)
    args.collect_workers = workers
    args.eval_workers = workers
    if args.train_num_threads is None:
        args.train_num_threads = args.cloud_vcpus
    if args.train_num_interop_threads is None:
        args.train_num_interop_threads = 1


if __name__ == "__main__":
    raise SystemExit(main())
