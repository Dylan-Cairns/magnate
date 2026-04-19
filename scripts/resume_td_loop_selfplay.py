from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from scripts.opponent_pool import PoolCheckpoint
from scripts.run_td_loop_selfplay import (
    REPLAY_REGIME,
    CollectProfile,
    _config_payload,
    _require_supported_runtime,
    _run_collect_profiles,
    _validate_args,
)
from scripts.td_loop_common import (
    LoopCheckpoint,
    build_train_command,
    checkpoints_from_train_summary,
    read_json,
    run_step,
    select_latest_checkpoint,
)
from scripts.td_loop_eval_common import (
    build_eval_payload,
    pool_eval_rows,
    read_eval_row,
)
from scripts.td_loop_selfplay_eval import (
    _build_eval_command_vs_incumbent,
    _build_eval_command_vs_search,
    _promotion_decision,
)


@dataclass(frozen=True)
class ResumeCollectTemplate:
    label: str
    games: int
    player_a_policy: str
    player_b_policy: str
    player_b_fixed_td_search: PoolCheckpoint | None
    player_b_uses_candidate: bool


@dataclass(frozen=True)
class CompletedChunk:
    index: int
    label: str
    chunk_dir: Path
    collect_summary: Path
    train_summary: Path
    latest_checkpoint: LoopCheckpoint


@dataclass(frozen=True)
class ResumeState:
    run_id: str
    run_dir: Path
    chunks_dir: Path
    eval_dir: Path
    loop_summary_path: Path
    progress_path: Path
    completed_chunks: Sequence[CompletedChunk]
    latest_checkpoint: LoopCheckpoint
    incumbent_checkpoint: PoolCheckpoint
    collect_templates: Sequence[ResumeCollectTemplate]
    collect_games_per_chunk: int
    collect_workers: int
    train_config: Dict[str, Any]
    partial_chunk_label: str | None
    highest_existing_chunk_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an interrupted TD self-play loop from the latest fully completed "
            "chunk, rerun the next partial chunk from scratch, then finish promotion eval."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Interrupted self-play run id under artifacts/td_loops.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/td_loops"),
        help="Root directory containing TD loop run folders.",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for collect/train/eval commands.",
    )
    parser.add_argument(
        "--chunks-per-loop",
        type=int,
        default=12,
        help="Target chunk count for the resumed run. Defaults to the self-play loop cadence.",
    )
    parser.add_argument(
        "--collect-seed-prefix",
        type=str,
        default="td-loop-collect-selfplay",
    )
    parser.add_argument(
        "--collect-progress-every-games",
        type=int,
        default=3,
        help="Progress cadence for resumed collect commands.",
    )
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=None,
        help="Optional collect shard override. Defaults to the recovered worker count.",
    )
    parser.add_argument(
        "--collect-opponent-pool-size",
        type=int,
        default=4,
        help="Recorded in summary config; resumed collect uses the recovered frozen profile set.",
    )

    parser.add_argument("--train-num-threads", type=int, default=None)
    parser.add_argument("--train-num-interop-threads", type=int, default=None)

    parser.add_argument("--eval-games-per-side", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=None)
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

    parser.add_argument("--incumbent-eval-games-per-side", type=int, default=200)
    parser.add_argument("--incumbent-eval-workers", type=int, default=None)
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

    parser.add_argument("--promotion-min-win-rate", type=float, default=0.55)
    parser.add_argument("--promotion-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-min-ci-low", type=float, default=0.5)
    parser.add_argument("--promotion-max-window-side-gap", type=float, default=0.10)
    parser.add_argument("--promotion-incumbent-min-win-rate", type=float, default=0.52)
    parser.add_argument("--promotion-incumbent-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-incumbent-min-ci-low", type=float, default=0.5)
    parser.add_argument("--promotion-incumbent-max-window-side-gap", type=float, default=0.10)

    parser.add_argument("--progress-heartbeat-minutes", type=float, default=30.0)
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Force rerun promotion eval artifacts even if they already exist.",
    )
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Overwrite loop.summary.json if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime(args.python_bin)

    state = _discover_resume_state(run_id=args.run_id, artifact_dir=args.artifact_dir)
    if state.loop_summary_path.exists() and not args.force_summary:
        raise SystemExit(
            f"Refusing to overwrite existing loop summary: {state.loop_summary_path}. "
            "Pass --force-summary to overwrite."
        )

    resolved_args = _resolve_resume_args(args=args, state=state)
    _validate_args(resolved_args)

    completed_count = len(state.completed_chunks)
    if completed_count >= resolved_args.chunks_per_loop:
        print(
            "[td-loop-selfplay-resume] all collect/train chunks are already complete; "
            "running eval/summary only."
        )
    else:
        print(
            "[td-loop-selfplay-resume] resuming after "
            f"{state.completed_chunks[-1].label}; restarting {state.partial_chunk_label or f'chunk-{completed_count + 1:03d}'}."
        )

    started = time.perf_counter()
    commands: Dict[str, Any] = {
        "completedChunks": [chunk.label for chunk in state.completed_chunks],
        "resumedChunks": [],
    }
    chunk_rows = [
        _chunk_row(
            chunk_label=chunk.label,
            collect_summary=chunk.collect_summary,
            train_summary=chunk.train_summary,
            checkpoint=chunk.latest_checkpoint,
        )
        for chunk in state.completed_chunks
    ]

    latest_checkpoint = state.latest_checkpoint
    next_chunk_index = completed_count + 1
    for chunk_index in range(next_chunk_index, resolved_args.chunks_per_loop + 1):
        chunk_label = f"chunk-{chunk_index:03d}"
        chunk_dir = state.chunks_dir / chunk_label
        replay_dir = chunk_dir / "replay"
        train_dir = chunk_dir / "train"
        for path in (chunk_dir, replay_dir, train_dir):
            path.mkdir(parents=True, exist_ok=True)

        collect_value_path = replay_dir / "self_play.value.jsonl"
        collect_opponent_path = replay_dir / "self_play.opponent.jsonl"
        collect_summary_path = replay_dir / "self_play.summary.json"

        candidate_checkpoint = PoolCheckpoint(
            run_id=state.run_id,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            value_path=latest_checkpoint.value_path,
            opponent_path=latest_checkpoint.opponent_path,
        )
        collect_profiles = _build_collect_profiles_from_templates(
            templates=state.collect_templates,
            candidate=candidate_checkpoint,
        )
        collect_commands = _run_collect_profiles(
            python_bin=resolved_args.python_bin,
            args=resolved_args,
            profiles=collect_profiles,
            replay_dir=replay_dir,
            run_id=f"{state.run_id}-{chunk_label}",
            collect_value_path=collect_value_path,
            collect_opponent_path=collect_opponent_path,
            collect_summary_path=collect_summary_path,
            heartbeat_minutes=resolved_args.progress_heartbeat_minutes,
            progress_path=state.progress_path,
        )

        train_summary_path = train_dir / "summary.json"
        train_checkpoint_root = train_dir / "checkpoints"
        train_command = build_train_command(
            python_bin=resolved_args.python_bin,
            args=resolved_args,
            value_replay=collect_value_path,
            opponent_replay=collect_opponent_path,
            train_summary_path=train_summary_path,
            train_checkpoint_root=train_checkpoint_root,
            run_id=f"{state.run_id}-{chunk_label}",
            warm_start_value=latest_checkpoint.value_path,
            warm_start_opponent=latest_checkpoint.opponent_path,
        )
        run_step(
            name=f"train[{chunk_label}]",
            command=train_command,
            heartbeat_minutes=resolved_args.progress_heartbeat_minutes,
            progress_path=state.progress_path,
            log_prefix="[td-loop-selfplay-resume]",
        )

        train_summary = read_json(train_summary_path, label=f"train summary {chunk_label}")
        checkpoints = checkpoints_from_train_summary(train_summary)
        latest_checkpoint = select_latest_checkpoint(
            checkpoints=checkpoints,
            candidate_policy="td-search",
        )

        commands["resumedChunks"].append(
            {
                "chunk": chunk_label,
                "collectProfiles": collect_commands,
                "train": train_command,
            }
        )
        chunk_rows.append(
            _chunk_row(
                chunk_label=chunk_label,
                collect_summary=collect_summary_path,
                train_summary=train_summary_path,
                checkpoint=latest_checkpoint,
            )
        )

    baseline_windows = _run_or_load_eval_windows_vs_search(
        args=resolved_args,
        eval_dir=state.eval_dir,
        checkpoint=latest_checkpoint,
        progress_path=state.progress_path,
        force_eval=bool(args.force_eval),
    )
    incumbent_windows = _run_or_load_eval_windows_vs_incumbent(
        args=resolved_args,
        eval_dir=state.eval_dir,
        checkpoint=latest_checkpoint,
        incumbent=state.incumbent_checkpoint,
        progress_path=state.progress_path,
        force_eval=bool(args.force_eval),
    )
    commands["promotionEvals"] = {
        "baselineVsSearch": [window["command"] for window in baseline_windows],
        "candidateVsIncumbent": [window["command"] for window in incumbent_windows],
    }

    baseline_rows = [window["row"] for window in baseline_windows]
    incumbent_rows = [window["row"] for window in incumbent_windows]
    pooled_baseline = pool_eval_rows(
        eval_rows=baseline_rows,
        opponent_policy="search",
    )
    pooled_incumbent = pool_eval_rows(
        eval_rows=incumbent_rows,
        opponent_policy="td-search",
    )
    promotion = _promotion_decision(
        baseline_eval=pooled_baseline,
        baseline_windows=baseline_rows,
        incumbent_eval=pooled_incumbent,
        incumbent_windows=incumbent_rows,
        args=resolved_args,
    )

    elapsed_minutes = (time.perf_counter() - started) / 60.0
    payload: Dict[str, Any] = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": state.run_id,
        "elapsedMinutes": round(elapsed_minutes, 3),
        "resume": {
            "resumedFromFailure": True,
            "resumePolicy": "restart-next-incomplete-chunk-from-scratch",
            "resumedAfterChunk": state.completed_chunks[-1].label,
            "discardedPartialChunk": state.partial_chunk_label,
            "script": "scripts.resume_td_loop_selfplay",
        },
        "config": _config_payload(resolved_args),
        "commands": commands,
        "artifacts": {
            "runDir": str(state.run_dir),
            "loopSummary": str(state.loop_summary_path),
            "progress": str(state.progress_path),
            "chunksDir": str(state.chunks_dir),
            "evalDir": str(state.eval_dir),
        },
        "chunks": chunk_rows,
        "evaluation": {
            "baselineVsSearch": build_eval_payload(
                resolved_args.eval_seed_start_indices,
                baseline_rows,
                pooled_baseline,
            ),
            "candidateVsIncumbent": build_eval_payload(
                resolved_args.incumbent_eval_seed_start_indices,
                incumbent_rows,
                pooled_incumbent,
            ),
        },
        "promotion": promotion,
    }
    state.loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "runId": state.run_id,
                "loopSummary": str(state.loop_summary_path),
                "baselineWinRate": pooled_baseline.candidate_win_rate,
                "baselineCi95": {"low": pooled_baseline.ci_low, "high": pooled_baseline.ci_high},
                "incumbentWinRate": pooled_incumbent.candidate_win_rate,
                "incumbentCi95": {"low": pooled_incumbent.ci_low, "high": pooled_incumbent.ci_high},
                "promoted": bool(promotion["promoted"]),
                "promotionReason": promotion["reason"],
            },
            indent=2,
        )
    )
    return 0


def _discover_resume_state(*, run_id: str, artifact_dir: Path) -> ResumeState:
    run_dir = artifact_dir / run_id
    chunks_dir = run_dir / "chunks"
    eval_dir = run_dir / "evals"
    loop_summary_path = run_dir / "loop.summary.json"
    progress_path = run_dir / "progress.json"

    if not run_dir.exists():
        raise SystemExit(f"Missing run directory: {run_dir}")
    if not chunks_dir.exists():
        raise SystemExit(f"Missing chunks directory: {chunks_dir}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(
        (
            chunk_dir
            for chunk_dir in chunks_dir.iterdir()
            if chunk_dir.is_dir() and _parse_chunk_index(chunk_dir.name) is not None
        ),
        key=lambda path: _parse_chunk_index(path.name),
    )
    if not chunk_dirs:
        raise SystemExit("No chunk directories found; nothing to resume.")

    completed_chunks: List[CompletedChunk] = []
    partial_chunk_label: str | None = None
    highest_existing_chunk_index = 0
    expected_index = 1
    for chunk_dir in chunk_dirs:
        chunk_index = _parse_chunk_index(chunk_dir.name)
        assert chunk_index is not None
        highest_existing_chunk_index = max(highest_existing_chunk_index, chunk_index)
        if chunk_index != expected_index:
            raise SystemExit(
                "Chunk directory sequence has a gap before resume point: "
                f"expected chunk-{expected_index:03d}, found {chunk_dir.name}"
            )
        expected_index += 1

        collect_summary = chunk_dir / "replay" / "self_play.summary.json"
        train_summary = chunk_dir / "train" / "summary.json"
        has_collect = collect_summary.exists()
        has_train = train_summary.exists()
        if has_collect and has_train:
            train_payload = read_json(train_summary, label=f"train summary {chunk_dir.name}")
            latest_checkpoint = select_latest_checkpoint(
                checkpoints=checkpoints_from_train_summary(train_payload),
                candidate_policy="td-search",
            )
            completed_chunks.append(
                CompletedChunk(
                    index=chunk_index,
                    label=chunk_dir.name,
                    chunk_dir=chunk_dir,
                    collect_summary=collect_summary,
                    train_summary=train_summary,
                    latest_checkpoint=latest_checkpoint,
                )
            )
            continue

        partial_chunk_label = chunk_dir.name
        break

    if not completed_chunks:
        raise SystemExit(
            "Self-play resume requires at least one fully completed chunk "
            "(collect + train summaries)."
        )

    latest_completed = completed_chunks[-1]
    latest_collect_payload = read_json(
        latest_completed.collect_summary,
        label=f"collect summary {latest_completed.label}",
    )
    latest_train_payload = read_json(
        latest_completed.train_summary,
        label=f"train summary {latest_completed.label}",
    )
    first_train_payload = read_json(
        completed_chunks[0].train_summary,
        label=f"train summary {completed_chunks[0].label}",
    )

    return ResumeState(
        run_id=run_id,
        run_dir=run_dir,
        chunks_dir=chunks_dir,
        eval_dir=eval_dir,
        loop_summary_path=loop_summary_path,
        progress_path=progress_path,
        completed_chunks=completed_chunks,
        latest_checkpoint=latest_completed.latest_checkpoint,
        incumbent_checkpoint=_incumbent_checkpoint_from_train_summary(first_train_payload),
        collect_templates=_collect_templates_from_summary(
            payload=latest_collect_payload,
            current_run_id=run_id,
        ),
        collect_games_per_chunk=_collect_games_from_summary(latest_collect_payload),
        collect_workers=_recover_collect_workers(latest_completed.chunk_dir),
        train_config=_train_config_from_summary(latest_train_payload),
        partial_chunk_label=partial_chunk_label,
        highest_existing_chunk_index=highest_existing_chunk_index,
    )


def _resolve_resume_args(*, args: argparse.Namespace, state: ResumeState) -> argparse.Namespace:
    collect_share_weights = _collect_share_weights(state.collect_templates)
    collect_workers = args.collect_workers or state.collect_workers
    eval_workers = args.eval_workers or collect_workers
    incumbent_eval_workers = args.incumbent_eval_workers or eval_workers
    train_num_threads = (
        args.train_num_threads
        if args.train_num_threads is not None
        else state.train_config.get("numThreads")
    )
    train_num_interop_threads = (
        args.train_num_interop_threads
        if args.train_num_interop_threads is not None
        else state.train_config.get("numInteropThreads")
    )

    return argparse.Namespace(
        python_bin=args.python_bin,
        artifact_dir=args.artifact_dir,
        run_label=_run_label_from_run_id(state.run_id),
        cloud=False,
        cloud_vcpus=8,
        chunks_per_loop=args.chunks_per_loop,
        collect_games=state.collect_games_per_chunk,
        collect_workers=collect_workers,
        collect_seed_prefix=args.collect_seed_prefix,
        collect_progress_every_games=args.collect_progress_every_games,
        collect_search_worlds=6,
        collect_search_rollouts=1,
        collect_search_depth=14,
        collect_search_max_root_actions=6,
        collect_search_rollout_epsilon=0.04,
        collect_td_worlds=8,
        collect_td_search_opponent_temperature=1.0,
        collect_td_search_sample_opponent_actions=False,
        collect_selfplay_share=collect_share_weights["selfplay"],
        collect_pool_share=collect_share_weights["pool"],
        collect_search_anchor_share=collect_share_weights["search"],
        collect_opponent_pool_size=args.collect_opponent_pool_size,
        train_steps=int(state.train_config["steps"]),
        train_value_batch_size=int(state.train_config["valueBatchSize"]),
        train_opponent_batch_size=int(state.train_config["opponentBatchSize"]),
        train_seed=int(state.train_config["seed"]),
        train_hidden_dim=int(state.train_config["hiddenDim"]),
        train_gamma=float(state.train_config["gamma"]),
        train_value_learning_rate=float(state.train_config["valueLearningRate"]),
        train_value_weight_decay=float(state.train_config["valueWeightDecay"]),
        train_opponent_learning_rate=float(state.train_config["opponentLearningRate"]),
        train_opponent_weight_decay=float(state.train_config["opponentWeightDecay"]),
        train_max_grad_norm=float(state.train_config["maxGradNorm"]),
        train_target_sync_interval=int(state.train_config["targetSyncInterval"]),
        train_value_target_mode=str(state.train_config["valueTargetMode"]),
        train_td_lambda=float(state.train_config["tdLambda"]),
        train_use_mse_loss=bool(state.train_config["useMseLoss"]),
        train_disable_value=bool(state.train_config["disableValue"]),
        train_disable_opponent=bool(state.train_config["disableOpponent"]),
        train_save_every_steps=int(state.train_config["saveEverySteps"]),
        train_progress_every_steps=int(state.train_config["progressEverySteps"]),
        train_num_threads=train_num_threads,
        train_num_interop_threads=train_num_interop_threads,
        train_warm_start_value_checkpoint=None,
        train_warm_start_opponent_checkpoint=None,
        eval_games_per_side=args.eval_games_per_side,
        eval_workers=eval_workers,
        eval_seed_prefix=args.eval_seed_prefix,
        eval_seed_start_indices=list(args.eval_seed_start_indices),
        eval_progress_every_games=args.eval_progress_every_games,
        eval_progress_log_minutes=args.eval_progress_log_minutes,
        eval_worker_torch_threads=args.eval_worker_torch_threads,
        eval_worker_torch_interop_threads=args.eval_worker_torch_interop_threads,
        eval_worker_blas_threads=args.eval_worker_blas_threads,
        eval_search_worlds=args.eval_search_worlds,
        eval_search_rollouts=args.eval_search_rollouts,
        eval_search_depth=args.eval_search_depth,
        eval_search_max_root_actions=args.eval_search_max_root_actions,
        eval_search_rollout_epsilon=args.eval_search_rollout_epsilon,
        eval_td_worlds=args.eval_td_worlds,
        eval_td_search_opponent_temperature=args.eval_td_search_opponent_temperature,
        eval_td_search_sample_opponent_actions=bool(args.eval_td_search_sample_opponent_actions),
        incumbent_eval_games_per_side=args.incumbent_eval_games_per_side,
        incumbent_eval_workers=incumbent_eval_workers,
        incumbent_eval_seed_prefix=args.incumbent_eval_seed_prefix,
        incumbent_eval_seed_start_indices=list(args.incumbent_eval_seed_start_indices),
        promotion_min_win_rate=args.promotion_min_win_rate,
        promotion_max_side_gap=args.promotion_max_side_gap,
        promotion_min_ci_low=args.promotion_min_ci_low,
        promotion_max_window_side_gap=args.promotion_max_window_side_gap,
        promotion_incumbent_min_win_rate=args.promotion_incumbent_min_win_rate,
        promotion_incumbent_max_side_gap=args.promotion_incumbent_max_side_gap,
        promotion_incumbent_min_ci_low=args.promotion_incumbent_min_ci_low,
        promotion_incumbent_max_window_side_gap=args.promotion_incumbent_max_window_side_gap,
        progress_heartbeat_minutes=args.progress_heartbeat_minutes,
    )


def _collect_templates_from_summary(
    *, payload: Dict[str, Any], current_run_id: str
) -> Sequence[ResumeCollectTemplate]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("Collect summary is missing config payload.")
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise SystemExit("Collect summary is missing config.profiles entries.")

    templates: List[ResumeCollectTemplate] = []
    for row in profiles:
        if not isinstance(row, dict):
            raise SystemExit("Collect summary profile entry must be an object.")
        label = row.get("label")
        games = row.get("games")
        player_a_policy = row.get("playerAPolicy")
        player_b_policy = row.get("playerBPolicy")
        if not isinstance(label, str) or not label:
            raise SystemExit("Collect summary profile is missing label.")
        if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
            raise SystemExit(f"Collect summary profile {label} has invalid games count.")
        if not isinstance(player_a_policy, str) or not isinstance(player_b_policy, str):
            raise SystemExit(f"Collect summary profile {label} is missing player policies.")

        a_checkpoint = _checkpoint_from_collect_profile(row.get("playerATdSearchCheckpoint"))
        b_checkpoint = _checkpoint_from_collect_profile(row.get("playerBTdSearchCheckpoint"))
        if player_a_policy == "td-search" and a_checkpoint is None:
            raise SystemExit(f"Collect summary profile {label} is missing PlayerA td-search checkpoint.")
        if player_a_policy == "td-search" and a_checkpoint is not None:
            # Completed chunks should always show the current run as the candidate owner.
            # Keep resume logic strict so we do not silently reinterpret a different run's checkpoint.
            if a_checkpoint.run_id != current_run_id:
                raise SystemExit(
                    "Collect summary PlayerA checkpoint does not match the interrupted run: "
                    f"profile={label} checkpointRunId={a_checkpoint.run_id} runId={current_run_id}"
                )

        player_b_uses_candidate = False
        player_b_fixed_td_search: PoolCheckpoint | None = None
        if player_b_policy == "td-search":
            if b_checkpoint is None:
                raise SystemExit(
                    f"Collect summary profile {label} is missing PlayerB td-search checkpoint."
                )
            if a_checkpoint is not None and _same_checkpoint(a_checkpoint, b_checkpoint):
                player_b_uses_candidate = True
            else:
                player_b_fixed_td_search = b_checkpoint

        templates.append(
            ResumeCollectTemplate(
                label=label,
                games=games,
                player_a_policy=player_a_policy,
                player_b_policy=player_b_policy,
                player_b_fixed_td_search=player_b_fixed_td_search,
                player_b_uses_candidate=player_b_uses_candidate,
            )
        )
    return templates


def _build_collect_profiles_from_templates(
    *, templates: Sequence[ResumeCollectTemplate], candidate: PoolCheckpoint
) -> List[CollectProfile]:
    profiles: List[CollectProfile] = []
    for template in templates:
        profiles.append(
            CollectProfile(
                label=template.label,
                games=template.games,
                player_a_policy=template.player_a_policy,
                player_b_policy=template.player_b_policy,
                player_a_td_search=candidate if template.player_a_policy == "td-search" else None,
                player_b_td_search=(
                    candidate
                    if template.player_b_uses_candidate
                    else template.player_b_fixed_td_search
                ),
            )
        )
    return profiles


def _run_or_load_eval_windows_vs_search(
    *,
    args: argparse.Namespace,
    eval_dir: Path,
    checkpoint: LoopCheckpoint,
    progress_path: Path,
    force_eval: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed_start_index in args.eval_seed_start_indices:
        out_path = eval_dir / f"promotion_eval.baseline.seed-{seed_start_index:06d}.json"
        command = _build_eval_command_vs_search(
            python_bin=args.python_bin,
            args=args,
            checkpoint=checkpoint,
            out_path=out_path,
            seed_prefix=args.eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=args.eval_workers,
            games_per_side=args.eval_games_per_side,
        )
        if force_eval or not out_path.exists():
            run_step(
                name=f"promotion-eval[baseline seed={seed_start_index}]",
                command=command,
                heartbeat_minutes=args.progress_heartbeat_minutes,
                progress_path=progress_path,
                log_prefix="[td-loop-selfplay-resume]",
            )
        else:
            print(
                "[td-loop-selfplay-resume] existing baseline eval artifact found, skipping: "
                f"{out_path}"
            )
        rows.append({"command": command, "row": read_eval_row(out_path, opponent_policy="search")})
    return rows


def _run_or_load_eval_windows_vs_incumbent(
    *,
    args: argparse.Namespace,
    eval_dir: Path,
    checkpoint: LoopCheckpoint,
    incumbent: PoolCheckpoint,
    progress_path: Path,
    force_eval: bool,
) -> List[Dict[str, Any]]:
    workers = args.incumbent_eval_workers or args.eval_workers
    rows: List[Dict[str, Any]] = []
    for seed_start_index in args.incumbent_eval_seed_start_indices:
        out_path = eval_dir / f"promotion_eval.incumbent.seed-{seed_start_index:06d}.json"
        command = _build_eval_command_vs_incumbent(
            python_bin=args.python_bin,
            args=args,
            checkpoint=checkpoint,
            incumbent=incumbent,
            out_path=out_path,
            seed_prefix=args.incumbent_eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=workers,
            games_per_side=args.incumbent_eval_games_per_side,
        )
        if force_eval or not out_path.exists():
            run_step(
                name=f"promotion-eval[incumbent seed={seed_start_index}]",
                command=command,
                heartbeat_minutes=args.progress_heartbeat_minutes,
                progress_path=progress_path,
                log_prefix="[td-loop-selfplay-resume]",
            )
        else:
            print(
                "[td-loop-selfplay-resume] existing incumbent eval artifact found, skipping: "
                f"{out_path}"
            )
        rows.append({"command": command, "row": read_eval_row(out_path, opponent_policy="td-search")})
    return rows


def _checkpoint_from_collect_profile(raw: Any) -> PoolCheckpoint | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SystemExit("Collect summary checkpoint entry must be an object or null.")
    run_id = raw.get("runId")
    value_raw = raw.get("value")
    opponent_raw = raw.get("opponent")
    if not isinstance(run_id, str) or not run_id:
        raise SystemExit("Collect summary checkpoint is missing runId.")
    if not isinstance(value_raw, str) or not isinstance(opponent_raw, str):
        raise SystemExit("Collect summary checkpoint is missing value/opponent paths.")
    value_path = Path(value_raw)
    opponent_path = Path(opponent_raw)
    if not value_path.exists() or not opponent_path.exists():
        raise SystemExit(
            "Collect summary checkpoint paths are missing: "
            f"value={value_path} opponent={opponent_path}"
        )
    return PoolCheckpoint(
        run_id=run_id,
        generated_at_utc="",
        value_path=value_path,
        opponent_path=opponent_path,
    )


def _same_checkpoint(left: PoolCheckpoint, right: PoolCheckpoint) -> bool:
    return (
        left.value_path.resolve() == right.value_path.resolve()
        and left.opponent_path.resolve() == right.opponent_path.resolve()
    )


def _collect_games_from_summary(payload: Dict[str, Any]) -> int:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("Collect summary is missing config payload.")
    games = config.get("games")
    if isinstance(games, bool) or not isinstance(games, int) or games <= 0:
        raise SystemExit("Collect summary config.games must be a positive integer.")
    return games


def _recover_collect_workers(chunk_dir: Path) -> int:
    profiles_dir = chunk_dir / "replay" / "profiles"
    if not profiles_dir.exists():
        return 1

    worker_count = 1
    for shard_dir in profiles_dir.glob("*.shards"):
        if not shard_dir.is_dir():
            continue
        shard_summaries = list(shard_dir.glob("*.summary.json"))
        worker_count = max(worker_count, len(shard_summaries))
    return worker_count


def _incumbent_checkpoint_from_train_summary(payload: Dict[str, Any]) -> PoolCheckpoint:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("Train summary is missing config payload.")
    value_raw = config.get("warmStartValueCheckpoint")
    opponent_raw = config.get("warmStartOpponentCheckpoint")
    if not isinstance(value_raw, str) or not isinstance(opponent_raw, str):
        raise SystemExit(
            "First completed chunk train summary is missing warm-start checkpoint paths."
        )
    value_path = Path(value_raw)
    opponent_path = Path(opponent_raw)
    if not value_path.exists() or not opponent_path.exists():
        raise SystemExit(
            "Warm-start checkpoint paths are missing for incumbent recovery: "
            f"value={value_path} opponent={opponent_path}"
        )
    return PoolCheckpoint(
        run_id="incumbent",
        generated_at_utc="",
        value_path=value_path,
        opponent_path=opponent_path,
    )


def _train_config_from_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("Train summary is missing config payload.")
    required_keys = (
        "steps",
        "seed",
        "valueBatchSize",
        "opponentBatchSize",
        "hiddenDim",
        "gamma",
        "valueLearningRate",
        "valueWeightDecay",
        "opponentLearningRate",
        "opponentWeightDecay",
        "maxGradNorm",
        "targetSyncInterval",
        "valueTargetMode",
        "tdLambda",
        "saveEverySteps",
        "progressEverySteps",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise SystemExit(f"Train summary config is missing required keys: {missing}")
    normalized = dict(config)
    if "useMseLoss" not in normalized:
        normalized["useMseLoss"] = normalized.get("valueLoss") == "mse"
    if "disableValue" not in normalized:
        train_value = normalized.get("trainValue")
        normalized["disableValue"] = False if train_value is None else (not bool(train_value))
    if "disableOpponent" not in normalized:
        train_opponent = normalized.get("trainOpponent")
        normalized["disableOpponent"] = (
            False if train_opponent is None else (not bool(train_opponent))
        )
    return normalized


def _collect_share_weights(templates: Sequence[ResumeCollectTemplate]) -> Dict[str, float]:
    total_games = sum(template.games for template in templates)
    if total_games <= 0:
        raise SystemExit("Collect template total games must be > 0.")

    selfplay_games = sum(template.games for template in templates if template.player_b_uses_candidate)
    search_games = sum(
        template.games for template in templates if template.label == "search-anchor"
    )
    pool_games = total_games - selfplay_games - search_games
    return {
        "selfplay": float(selfplay_games) / float(total_games),
        "pool": float(pool_games) / float(total_games),
        "search": float(search_games) / float(total_games),
    }


def _run_label_from_run_id(run_id: str) -> str:
    parts = run_id.split("-", 2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].endswith("Z"):
        return parts[2]
    return run_id


def _chunk_row(
    *,
    chunk_label: str,
    collect_summary: Path,
    train_summary: Path,
    checkpoint: LoopCheckpoint,
) -> Dict[str, Any]:
    return {
        "chunk": chunk_label,
        "replayRegime": REPLAY_REGIME,
        "collectSummary": str(collect_summary),
        "trainSummary": str(train_summary),
        "latestCheckpoint": {
            "step": checkpoint.step,
            "value": str(checkpoint.value_path),
            "opponent": str(checkpoint.opponent_path),
        },
    }


def _parse_chunk_index(name: str) -> int | None:
    prefix = "chunk-"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if len(suffix) != 3 or not suffix.isdigit():
        return None
    return int(suffix)


if __name__ == "__main__":
    raise SystemExit(main())
