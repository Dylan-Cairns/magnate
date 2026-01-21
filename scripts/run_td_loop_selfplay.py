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

from scripts.opponent_pool import (
    PoolCheckpoint,
    filter_pool_excluding_checkpoint,
    load_promoted_checkpoints,
    split_evenly,
    weighted_game_split,
)
from scripts.td_loop_common import (
    LoopCheckpoint,
    build_train_command,
    checkpoints_from_train_summary,
    concat_jsonl_files,
    read_json,
    run_step,
    select_latest_checkpoint,
)
from scripts.td_loop_selfplay_common import (
    EvalRow,
    _build_eval_command_vs_incumbent,
    _build_eval_command_vs_search,
    _eval_payload,
    _pool_eval_rows,
    _promotion_decision,
    _read_eval_row,
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

    # Keep chunk-local training/eval, but wait longer before promotion eval.
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
        default="td0",
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
    commands: Dict[str, Any] = {"chunks": []}
    chunk_rows: List[Dict[str, Any]] = []
    latest_checkpoint = context.latest_checkpoint

    for chunk_index in range(1, args.chunks_per_loop + 1):
        chunk_result = run_selfplay_chunk(
            args=args,
            run_id=context.run_id,
            chunk_index=chunk_index,
            chunks_dir=context.chunks_dir,
            latest_checkpoint=latest_checkpoint,
            progress_path=context.progress_path,
        )
        latest_checkpoint = chunk_result.latest_checkpoint
        commands["chunks"].append(chunk_result.command_row)
        chunk_rows.append(chunk_result.chunk_row)

    promotion_stage = run_promotion_stage(
        args=args,
        eval_dir=context.eval_dir,
        latest_checkpoint=latest_checkpoint,
        incumbent_checkpoint=context.incumbent_checkpoint,
        progress_path=context.progress_path,
    )
    commands["promotionEvals"] = promotion_stage.commands

    loop_elapsed_minutes = (time.perf_counter() - context.loop_started) / 60.0
    payload = build_selfplay_loop_summary(
        args=args,
        context=context,
        commands=commands,
        chunk_rows=chunk_rows,
        promotion_stage=promotion_stage,
        elapsed_minutes=loop_elapsed_minutes,
    )
    context.loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(build_selfplay_terminal_report(context=context, promotion_stage=promotion_stage), indent=2))
    return 0


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
    latest_checkpoint: LoopCheckpoint,
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
            value_path=latest_checkpoint.value_path,
            opponent_path=latest_checkpoint.opponent_path,
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
    train_command = build_train_command(
        python_bin=args.python_bin,
        args=args,
        value_replay=collect_value_path,
        opponent_replay=collect_opponent_path,
        train_summary_path=train_summary_path,
        train_checkpoint_root=train_checkpoint_root,
        run_id=f"{run_id}-{chunk_label}",
        warm_start_value=latest_checkpoint.value_path,
        warm_start_opponent=latest_checkpoint.opponent_path,
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
    next_checkpoint = select_latest_checkpoint(
        checkpoints=checkpoints,
        candidate_policy="td-search",
    )
    return ChunkExecutionResult(
        chunk_label=chunk_label,
        latest_checkpoint=next_checkpoint,
        command_row={
            "chunk": chunk_label,
            "collectProfiles": collect_commands,
            "train": train_command,
        },
        chunk_row={
            "chunk": chunk_label,
            "replayRegime": REPLAY_REGIME,
            "collectSummary": str(collect_summary_path),
            "trainSummary": str(train_summary_path),
            "latestCheckpoint": {
                "step": next_checkpoint.step,
                "value": str(next_checkpoint.value_path),
                "opponent": str(next_checkpoint.opponent_path),
            },
        },
    )


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
    pooled_baseline = _pool_eval_rows(eval_rows=baseline_rows, opponent_policy="search")
    pooled_incumbent = _pool_eval_rows(eval_rows=incumbent_rows, opponent_policy="td-search")
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
    elapsed_minutes: float,
) -> Dict[str, Any]:
    baseline_rows = [window["row"] for window in promotion_stage.baseline_windows]
    incumbent_rows = [window["row"] for window in promotion_stage.incumbent_windows]
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
        },
        "chunks": list(chunk_rows),
        "evaluation": {
            "baselineVsSearch": _eval_payload(
                args.eval_seed_start_indices,
                baseline_rows,
                promotion_stage.pooled_baseline,
            ),
            "candidateVsIncumbent": _eval_payload(
                args.incumbent_eval_seed_start_indices,
                incumbent_rows,
                promotion_stage.pooled_incumbent,
            ),
        },
        "promotion": promotion_stage.promotion,
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
        rows.append({"command": command, "row": _read_eval_row(out_path, opponent_policy="search")})
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
        rows.append({"command": command, "row": _read_eval_row(out_path, opponent_policy="td-search")})
    return rows


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
        },
        "promotion": {
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
