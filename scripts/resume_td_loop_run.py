from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class LoopCheckpoint:
    step: int
    value_path: Path
    opponent_path: Path


@dataclass(frozen=True)
class EvalRow:
    artifact: Path
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an interrupted TD loop run from chunk-003 training and complete promotion eval."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="20260306-061428Z-td-loop-r2-overnight",
        help="Interrupted run id under artifacts/td_loops.",
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
        help="Python executable used for train/eval stage commands.",
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
        "--train-num-threads",
        type=int,
        default=None,
        help=(
            "Optional torch intra-op CPU thread count override for resumed chunk-003 training. "
            "When omitted, inherits from chunk-002 training config."
        ),
    )
    parser.add_argument(
        "--train-num-interop-threads",
        type=int,
        default=None,
        help=(
            "Optional torch inter-op CPU thread count override for resumed chunk-003 training. "
            "When omitted, inherits from chunk-002 training config."
        ),
    )
    parser.add_argument(
        "--eval-opponent-policy",
        type=str,
        choices=("random", "heuristic", "search"),
        default="search",
    )
    parser.add_argument(
        "--eval-games-per-side",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval-seed-prefix",
        type=str,
        default="td-loop-eval",
    )
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
    parser.add_argument("--promotion-min-win-rate", type=float, default=0.55)
    parser.add_argument("--promotion-max-side-gap", type=float, default=0.08)
    parser.add_argument("--promotion-min-ci-low", type=float, default=0.5)
    parser.add_argument("--promotion-max-window-side-gap", type=float, default=0.10)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force rerun chunk-003 training even if train summary already exists.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Force rerun promotion eval windows even if artifacts already exist.",
    )
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Overwrite loop.summary.json if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cloud:
        _apply_cloud_profile(args)
    _validate_args(args)
    _require_supported_runtime(args.python_bin)

    run_dir = args.artifact_dir / args.run_id
    chunks_dir = run_dir / "chunks"
    eval_dir = run_dir / "evals"
    loop_summary_path = run_dir / "loop.summary.json"
    progress_path = run_dir / "progress.json"

    chunk1_collect = chunks_dir / "chunk-001/replay/self_play.summary.json"
    chunk2_collect = chunks_dir / "chunk-002/replay/self_play.summary.json"
    chunk3_collect = chunks_dir / "chunk-003/replay/self_play.summary.json"
    chunk1_train = chunks_dir / "chunk-001/train/summary.json"
    chunk2_train = chunks_dir / "chunk-002/train/summary.json"
    chunk3_train = chunks_dir / "chunk-003/train/summary.json"
    chunk3_value = chunks_dir / "chunk-003/replay/self_play.value.jsonl"
    chunk3_opponent = chunks_dir / "chunk-003/replay/self_play.opponent.jsonl"
    chunk3_checkpoint_root = chunks_dir / "chunk-003/train/checkpoints"

    _require_paths(
        label="run resume preflight",
        paths=(
            run_dir,
            eval_dir,
            chunk1_collect,
            chunk2_collect,
            chunk3_collect,
            chunk1_train,
            chunk2_train,
            chunk3_value,
            chunk3_opponent,
        ),
    )
    _assert_collect_complete(chunk3_collect)

    if loop_summary_path.exists() and not args.force_summary:
        raise SystemExit(
            f"Refusing to overwrite existing loop summary: {loop_summary_path}. "
            "Pass --force-summary to overwrite."
        )

    started = time.perf_counter()
    chunk2_train_payload = _read_json(chunk2_train, label="chunk-002 train summary")
    warm_start = _latest_checkpoint_from_train_summary(chunk2_train_payload)
    chunk2_train_config = _train_config_from_summary(chunk2_train_payload)

    train_command = _build_train_command(
        python_bin=args.python_bin,
        run_id=args.run_id,
        value_replay=chunk3_value,
        opponent_replay=chunk3_opponent,
        train_summary_path=chunk3_train,
        train_checkpoint_root=chunk3_checkpoint_root,
        train_config=chunk2_train_config,
        warm_start=warm_start,
        args=args,
    )

    should_run_train = args.force_train or (not chunk3_train.exists())
    if should_run_train:
        _run_command(
            step="train[chunk-003-resume]",
            command=train_command,
            progress_path=progress_path,
        )
    else:
        existing = _read_json(chunk3_train, label="chunk-003 train summary")
        latest_existing = _latest_checkpoint_from_train_summary(existing)
        if latest_existing.step < int(chunk2_train_config["steps"]):
            _run_command(
                step="train[chunk-003-resume]",
                command=train_command,
                progress_path=progress_path,
            )
        else:
            print(
                "[td-loop-resume] chunk-003 train summary already complete; skipping train stage."
            )

    chunk1_train_payload = _read_json(chunk1_train, label="chunk-001 train summary")
    chunk3_train_payload = _read_json(chunk3_train, label="chunk-003 train summary")
    chunk1_latest = _latest_checkpoint_from_train_summary(chunk1_train_payload)
    chunk2_latest = _latest_checkpoint_from_train_summary(chunk2_train_payload)
    chunk3_latest = _latest_checkpoint_from_train_summary(chunk3_train_payload)

    eval_rows: List[EvalRow] = []
    eval_commands: List[List[str]] = []
    eval_artifacts: List[str] = []
    for seed_start_index in args.eval_seed_start_indices:
        eval_artifact = eval_dir / f"promotion_eval.seed-{seed_start_index:06d}.json"
        eval_command = _build_eval_command(
            python_bin=args.python_bin,
            checkpoint=chunk3_latest,
            opponent_policy=args.eval_opponent_policy,
            seed_prefix=args.eval_seed_prefix,
            seed_start_index=seed_start_index,
            workers=args.eval_workers,
            games_per_side=args.eval_games_per_side,
            out_path=eval_artifact,
            args=args,
        )
        eval_commands.append(eval_command)
        eval_artifacts.append(str(eval_artifact))

        if args.force_eval or (not eval_artifact.exists()):
            _run_command(
                step=f"promotion-eval[seed={seed_start_index}]",
                command=eval_command,
                progress_path=progress_path,
            )
        else:
            print(f"[td-loop-resume] existing eval artifact found, skipping: {eval_artifact}")
        eval_rows.append(_read_eval_row(eval_artifact))

    pooled = _pool_eval_rows(eval_rows)
    promotion = _promotion_decision(eval_row=pooled, eval_windows=eval_rows, args=args)

    loop_elapsed_minutes = (time.perf_counter() - started) / 60.0
    payload: Dict[str, Any] = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": args.run_id,
        "elapsedMinutes": round(loop_elapsed_minutes, 3),
        "resume": {
            "resumedFromFailure": True,
            "resumePoint": "train[chunk-003]",
            "script": "scripts.resume_td_loop_run",
        },
        "commands": {
            "resumeTrain": train_command,
            "promotionEvals": eval_commands,
        },
        "artifacts": {
            "runDir": str(run_dir),
            "loopSummary": str(loop_summary_path),
            "progress": str(progress_path),
            "chunksDir": str(chunks_dir),
            "evalDir": str(eval_dir),
            "promotionEvalArtifact": eval_artifacts[0],
            "promotionEvalArtifacts": eval_artifacts,
        },
        "chunks": [
            _chunk_row(
                chunk_label="chunk-001",
                collect_summary=chunk1_collect,
                train_summary=chunk1_train,
                checkpoint=chunk1_latest,
            ),
            _chunk_row(
                chunk_label="chunk-002",
                collect_summary=chunk2_collect,
                train_summary=chunk2_train,
                checkpoint=chunk2_latest,
            ),
            _chunk_row(
                chunk_label="chunk-003",
                collect_summary=chunk3_collect,
                train_summary=chunk3_train,
                checkpoint=chunk3_latest,
            ),
        ],
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
                for seed_start_index, row in zip(args.eval_seed_start_indices, eval_rows)
            ],
            "pooled": {
                "candidateWinRate": pooled.candidate_win_rate,
                "candidateWinRateCi95": {"low": pooled.ci_low, "high": pooled.ci_high},
                "candidateWinRateAsPlayerA": pooled.candidate_win_rate_as_player_a,
                "candidateWinRateAsPlayerB": pooled.candidate_win_rate_as_player_b,
                "sideGap": pooled.side_gap,
                "candidateWins": pooled.candidate_wins,
                "opponentWins": pooled.opponent_wins,
                "draws": pooled.draws,
                "totalGames": pooled.total_games,
            },
        },
        "promotion": promotion,
    }
    loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "runId": args.run_id,
                "loopSummary": str(loop_summary_path),
                "candidateWinRate": pooled.candidate_win_rate,
                "candidateWinRateCi95": {"low": pooled.ci_low, "high": pooled.ci_high},
                "sideGap": pooled.side_gap,
                "promoted": bool(promotion["promoted"]),
                "promotionReason": promotion["reason"],
            },
            indent=2,
        )
    )
    return 0


def _build_train_command(
    *,
    python_bin: Path,
    run_id: str,
    value_replay: Path,
    opponent_replay: Path,
    train_summary_path: Path,
    train_checkpoint_root: Path,
    train_config: Dict[str, Any],
    warm_start: LoopCheckpoint,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--value-replay",
        str(value_replay),
        "--opponent-replay",
        str(opponent_replay),
        "--steps",
        str(train_config["steps"]),
        "--value-batch-size",
        str(train_config["valueBatchSize"]),
        "--opponent-batch-size",
        str(train_config["opponentBatchSize"]),
        "--seed",
        str(train_config["seed"]),
        "--hidden-dim",
        str(train_config["hiddenDim"]),
        "--gamma",
        str(train_config["gamma"]),
        "--value-learning-rate",
        str(train_config["valueLearningRate"]),
        "--value-weight-decay",
        str(train_config["valueWeightDecay"]),
        "--opponent-learning-rate",
        str(train_config["opponentLearningRate"]),
        "--opponent-weight-decay",
        str(train_config["opponentWeightDecay"]),
        "--max-grad-norm",
        str(train_config["maxGradNorm"]),
        "--target-sync-interval",
        str(train_config["targetSyncInterval"]),
        "--value-target-mode",
        str(train_config["valueTargetMode"]),
        "--td-lambda",
        str(train_config["tdLambda"]),
        "--save-every-steps",
        str(train_config["saveEverySteps"]),
        "--progress-every-steps",
        str(train_config["progressEverySteps"]),
        "--out-dir",
        str(train_checkpoint_root),
        "--run-label",
        f"{run_id}-chunk-003-resume",
        "--summary-out",
        str(train_summary_path),
        "--warm-start-value-checkpoint",
        str(warm_start.value_path),
        "--warm-start-opponent-checkpoint",
        str(warm_start.opponent_path),
    ]

    if bool(train_config["useMseLoss"]):
        command.append("--use-mse-loss")
    if bool(train_config["disableValue"]):
        command.append("--disable-value")
    if bool(train_config["disableOpponent"]):
        command.append("--disable-opponent")

    num_threads = (
        args.train_num_threads
        if args.train_num_threads is not None
        else train_config.get("numThreads")
    )
    if num_threads is not None:
        command.extend(["--num-threads", str(num_threads)])
    num_interop_threads = (
        args.train_num_interop_threads
        if args.train_num_interop_threads is not None
        else train_config.get("numInteropThreads")
    )
    if num_interop_threads is not None:
        command.extend(["--num-interop-threads", str(num_interop_threads)])
    return command


def _build_eval_command(
    *,
    python_bin: Path,
    checkpoint: LoopCheckpoint,
    opponent_policy: str,
    seed_prefix: str,
    seed_start_index: int,
    workers: int,
    games_per_side: int,
    out_path: Path,
    args: argparse.Namespace,
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
        "td-search",
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
        "--td-search-value-checkpoint",
        str(checkpoint.value_path),
        "--td-search-opponent-checkpoint",
        str(checkpoint.opponent_path),
        "--td-search-opponent-temperature",
        str(args.eval_td_search_opponent_temperature),
    ]
    if args.eval_td_search_sample_opponent_actions:
        command.append("--td-search-sample-opponent-actions")
    return command


def _read_eval_row(path: Path) -> EvalRow:
    payload = _read_json(path, label=f"eval artifact {path.name}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Eval artifact is missing results payload: {path}")
    return EvalRow(
        artifact=path,
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


def _pool_eval_rows(eval_rows: Sequence[EvalRow]) -> EvalRow:
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


def _promotion_decision(
    *,
    eval_row: EvalRow,
    eval_windows: Sequence[EvalRow],
    args: argparse.Namespace,
) -> Dict[str, Any]:
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
        "reason": "pooled_eval_passed" if promoted else "pooled_eval_failed",
    }


def _chunk_row(
    *,
    chunk_label: str,
    collect_summary: Path,
    train_summary: Path,
    checkpoint: LoopCheckpoint,
) -> Dict[str, Any]:
    return {
        "chunk": chunk_label,
        "replayRegime": "chunk-local",
        "collectSummary": str(collect_summary),
        "trainSummary": str(train_summary),
        "latestCheckpoint": {
            "step": checkpoint.step,
            "value": str(checkpoint.value_path),
            "opponent": str(checkpoint.opponent_path),
        },
    }


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
        value_loss = normalized.get("valueLoss")
        normalized["useMseLoss"] = value_loss == "mse"
    if "disableValue" not in normalized:
        train_value = normalized.get("trainValue")
        normalized["disableValue"] = False if train_value is None else (not bool(train_value))
    if "disableOpponent" not in normalized:
        train_opponent = normalized.get("trainOpponent")
        normalized["disableOpponent"] = (
            False if train_opponent is None else (not bool(train_opponent))
        )
    return normalized


def _latest_checkpoint_from_train_summary(payload: Dict[str, Any]) -> LoopCheckpoint:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit("Train summary is missing results payload.")
    checkpoints = results.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise SystemExit("Train summary has no checkpoints.")

    latest: LoopCheckpoint | None = None
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise SystemExit("Train checkpoint entry must be an object.")
        step = entry.get("step")
        value_raw = entry.get("value")
        opponent_raw = entry.get("opponent")
        if isinstance(step, bool) or not isinstance(step, int):
            raise SystemExit(f"Train checkpoint step is invalid: {step!r}")
        if not isinstance(value_raw, str) or not isinstance(opponent_raw, str):
            continue
        candidate = LoopCheckpoint(
            step=step,
            value_path=Path(value_raw),
            opponent_path=Path(opponent_raw),
        )
        if latest is None or candidate.step > latest.step:
            latest = candidate

    if latest is None:
        raise SystemExit("Train summary has no checkpoint with both value/opponent paths.")

    _require_paths(
        label="checkpoint preflight",
        paths=(latest.value_path, latest.opponent_path),
    )
    return latest


def _assert_collect_complete(summary_path: Path) -> None:
    payload = _read_json(summary_path, label="chunk-003 collect summary")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit("Chunk-003 collect summary missing results payload.")
    games = results.get("games")
    value_transitions = results.get("valueTransitions")
    opponent_samples = results.get("opponentSamples")
    if not isinstance(games, int) or games <= 0:
        raise SystemExit("Chunk-003 collect summary has invalid game count.")
    if not isinstance(value_transitions, int) or value_transitions <= 0:
        raise SystemExit("Chunk-003 collect summary has invalid valueTransitions count.")
    if not isinstance(opponent_samples, int) or opponent_samples <= 0:
        raise SystemExit("Chunk-003 collect summary has invalid opponentSamples count.")


def _require_paths(*, label: str, paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing paths for {label}: {missing}")


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _run_command(*, step: str, command: Sequence[str], progress_path: Path) -> None:
    print(f"[td-loop-resume] step {step}: {_join_command(command)}")
    _write_progress(
        progress_path,
        {
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "status": "running",
        },
    )
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    if completed.returncode != 0:
        _write_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "step": step,
                "status": "failed",
                "elapsedMinutes": round(elapsed_minutes, 3),
                "returnCode": int(completed.returncode),
            },
        )
        raise SystemExit(
            f"[td-loop-resume] failed step={step} returnCode={completed.returncode} "
            f"elapsedMin={elapsed_minutes:.1f}"
        )
    _write_progress(
        progress_path,
        {
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "status": "completed",
            "elapsedMinutes": round(elapsed_minutes, 3),
            "returnCode": int(completed.returncode),
        },
    )
    print(f"[td-loop-resume] completed step={step} elapsedMin={elapsed_minutes:.1f}")


def _write_progress(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _join_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _recommended_cloud_worker_count(vcpus: int) -> int:
    return max(1, vcpus // 2)


def _apply_cloud_profile(args: argparse.Namespace) -> None:
    args.eval_workers = _recommended_cloud_worker_count(args.cloud_vcpus)
    if args.train_num_threads is None:
        args.train_num_threads = args.cloud_vcpus
    if args.train_num_interop_threads is None:
        args.train_num_interop_threads = 1


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


def _require_supported_runtime(python_bin: Path) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")
    if not python_bin.exists():
        raise SystemExit(f"--python-bin does not exist: {python_bin}")


def _validate_args(args: argparse.Namespace) -> None:
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
    if args.train_num_threads is not None and args.train_num_threads <= 0:
        raise SystemExit("--train-num-threads must be > 0 when provided.")
    if args.train_num_interop_threads is not None and args.train_num_interop_threads <= 0:
        raise SystemExit("--train-num-interop-threads must be > 0 when provided.")
    if args.promotion_min_win_rate < 0.0 or args.promotion_min_win_rate > 1.0:
        raise SystemExit("--promotion-min-win-rate must be in [0, 1].")
    if args.promotion_max_side_gap < 0.0 or args.promotion_max_side_gap > 1.0:
        raise SystemExit("--promotion-max-side-gap must be in [0, 1].")
    if args.promotion_min_ci_low < 0.0 or args.promotion_min_ci_low > 1.0:
        raise SystemExit("--promotion-min-ci-low must be in [0, 1].")
    if args.promotion_max_window_side_gap < 0.0 or args.promotion_max_window_side_gap > 1.0:
        raise SystemExit("--promotion-max-window-side-gap must be in [0, 1].")


if __name__ == "__main__":
    raise SystemExit(main())
