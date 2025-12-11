from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.eval_suite import wilson_interval
from trainer.policies import (
    SearchConfig,
    TDSearchPolicyConfig,
    TDValuePolicyConfig,
    policy_from_name,
)

TERMINAL_GATE_STATUSES = frozenset(("accepted", "rejected", "completed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical side-swapped paired-seed evaluation. "
            "Mode is required: gate (sequential SPRT) or certify (fixed-size)."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("gate", "certify"),
        help="Evaluation mode: gate (sequential SPRT) or certify (fixed-size report).",
    )
    parser.add_argument(
        "--games-per-side",
        type=int,
        default=400,
        help="Games with candidate on each seat for certify mode (total games = 2x).",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default="eval-suite",
        help="Shared seed prefix used for side-swapped legs.",
    )
    parser.add_argument(
        "--candidate-policy",
        type=str,
        required=True,
        help="Candidate policy (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        required=True,
        help="Opponent policy (random|heuristic|search|td-value|td-search).",
    )
    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument(
        "--td-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when candidate/opponent policy is td-value.",
    )
    parser.add_argument(
        "--td-worlds",
        type=int,
        default=8,
        help="Determinization world samples per decision for td-value policy.",
    )
    parser.add_argument(
        "--td-search-value-checkpoint",
        type=Path,
        default=None,
        help="Path to TD value checkpoint used when candidate/opponent policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-checkpoint",
        type=Path,
        default=None,
        help="TD opponent checkpoint used when candidate/opponent policy is td-search.",
    )
    parser.add_argument(
        "--td-search-opponent-temperature",
        type=float,
        default=1.0,
        help="Opponent policy temperature for td-search (lower is greedier).",
    )
    parser.add_argument(
        "--td-search-sample-opponent-actions",
        action="store_true",
        help="Sample opponent rollout actions from opponent model distribution in td-search.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON artifact path. "
            "Default: artifacts/evals/<utc>-<seed-prefix>-<mode>-<candidate>-vs-<opponent>.json"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Each worker runs an independent bridge process.",
    )
    parser.add_argument(
        "--seed-start-index",
        type=int,
        default=0,
        help="Seed index offset for deterministic sharding.",
    )
    parser.add_argument(
        "--progress-every-games",
        type=int,
        default=10,
        help=(
            "Print in-run progress every N games per leg per shard. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--progress-log-minutes",
        type=float,
        default=30.0,
        help=(
            "Minimum minutes between progress log lines for the same leg/shard. "
            "Final completion logs are always emitted."
        ),
    )
    parser.add_argument(
        "--progress-log-seconds",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--progress-out",
        type=Path,
        default=None,
        help=(
            "Optional progress artifact path updated during evaluation. "
            "Default: sibling eval.progress.json next to --out artifact."
        ),
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=1,
        help="Torch intra-op thread count per worker process.",
    )
    parser.add_argument(
        "--worker-torch-interop-threads",
        type=int,
        default=1,
        help="Torch inter-op thread count per worker process.",
    )
    parser.add_argument(
        "--worker-blas-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP thread count per worker process.",
    )

    # Gate-mode sequential test controls.
    parser.add_argument("--gate-h0-win-rate", type=float, default=0.50)
    parser.add_argument("--gate-h1-win-rate", type=float, default=0.55)
    parser.add_argument("--gate-alpha", type=float, default=0.05)
    parser.add_argument("--gate-beta", type=float, default=0.10)
    parser.add_argument("--gate-batch-games-per-side", type=int, default=25)
    parser.add_argument("--gate-max-games-per-side", type=int, default=400)
    parser.add_argument("--gate-max-side-gap", type=float, default=0.08)
    parser.add_argument(
        "--resume-from-artifact",
        type=Path,
        default=None,
        help="Gate mode only: resume/update this artifact in place.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_log_seconds is not None:
        args.progress_log_minutes = args.progress_log_seconds / 60.0
    _require_supported_runtime()
    _validate_policy_args(args)
    _validate_args(args)

    output_path = args.out or _default_output_path(
        seed_prefix=args.seed_prefix,
        mode=args.mode,
        candidate_policy=args.candidate_policy,
        opponent_policy=args.opponent_policy,
    )
    progress_path = _resolve_progress_path(args=args, output_path=output_path)

    if args.mode == "gate":
        payload = _run_gate_mode(args=args, output_path=output_path, progress_path=progress_path)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict):
            raise SystemExit("Gate artifact is missing results payload.")
        decision = payload.get("decision") if isinstance(payload, dict) else None
        status = str(payload.get("status")) if isinstance(payload, dict) else "unknown"
        decision_state = (
            str(decision.get("state"))
            if isinstance(decision, dict) and isinstance(decision.get("state"), str)
            else status
        )
        print(
            json.dumps(
                {
                    "artifact": str(output_path),
                    "mode": args.mode,
                    "status": status,
                    "decision": decision_state,
                    "candidateWinRate": results["candidateWinRate"],
                    "candidateWinRateCi95": results["candidateWinRateCi95"],
                    "sideGap": results["sideGap"],
                    "candidateWins": results["candidateWins"],
                    "opponentWins": results["opponentWins"],
                    "draws": results["draws"],
                    "totalGames": results["totalGames"],
                    "gamesPerSide": results["gamesPerSide"],
                },
                indent=2,
            )
        )
        return 0

    results = _evaluate_results(
        args=args,
        games_per_side=args.games_per_side,
        seed_start_index=args.seed_start_index,
        seed_prefix=args.seed_prefix,
        workers=args.workers,
        progress_path=progress_path,
        progress_label="certify",
    )
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "mode": "certify",
        "config": _base_config_payload(args),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, payload)
    _write_eval_progress(
        progress_path,
        {
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "mode": "certify",
            "status": "completed",
            "artifact": str(output_path),
            "results": {
                "candidateWinRate": payload["results"]["candidateWinRate"],
                "candidateWinRateCi95": payload["results"]["candidateWinRateCi95"],
                "sideGap": payload["results"]["sideGap"],
                "candidateWins": payload["results"]["candidateWins"],
                "opponentWins": payload["results"]["opponentWins"],
                "draws": payload["results"]["draws"],
                "totalGames": payload["results"]["totalGames"],
            },
        },
    )

    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "mode": "certify",
                "status": "completed",
                "candidateWinRate": payload["results"]["candidateWinRate"],
                "candidateWinRateCi95": payload["results"]["candidateWinRateCi95"],
                "sideGap": payload["results"]["sideGap"],
                "candidateWins": payload["results"]["candidateWins"],
                "opponentWins": payload["results"]["opponentWins"],
                "draws": payload["results"]["draws"],
                "totalGames": payload["results"]["totalGames"],
            },
            indent=2,
        )
    )
    return 0


def _run_gate_mode(
    *,
    args: argparse.Namespace,
    output_path: Path,
    progress_path: Path,
) -> Dict[str, object]:
    artifact_path = _resolve_gate_artifact_path(args=args, output_path=output_path)

    payload: Dict[str, object]
    if artifact_path.exists():
        payload = _read_json_object(artifact_path, label="gate artifact")
        _validate_gate_resume_payload(args=args, payload=payload)
    else:
        payload = _new_gate_payload(args=args, artifact_path=artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(artifact_path, payload)

    status = str(payload.get("status", ""))
    if status in TERMINAL_GATE_STATUSES:
        return payload

    while True:
        progress = payload.get("progress")
        if not isinstance(progress, dict):
            raise SystemExit("Gate artifact progress payload is invalid.")
        completed_games_per_side = int(progress.get("gamesPerSideCompleted", 0))
        if completed_games_per_side < 0:
            raise SystemExit("Gate artifact has negative gamesPerSideCompleted.")
        if completed_games_per_side >= args.gate_max_games_per_side:
            payload = _finalize_gate_at_cap(args=args, payload=payload)
            _write_json_atomic(artifact_path, payload)
            return payload

        remaining = args.gate_max_games_per_side - completed_games_per_side
        batch_games_per_side = min(args.gate_batch_games_per_side, remaining)
        batch_seed_start_index = args.seed_start_index + completed_games_per_side

        batch_result = _evaluate_results(
            args=args,
            games_per_side=batch_games_per_side,
            seed_start_index=batch_seed_start_index,
            seed_prefix=args.seed_prefix,
            workers=args.workers,
            progress_path=progress_path,
            progress_label=f"gate-batch-{len(history) + 1}",
        )

        existing_results = payload.get("results")
        if isinstance(existing_results, dict):
            merged = _merge_shard_results([existing_results, batch_result])
        else:
            merged = batch_result

        candidate_wins = int(merged["candidateWins"])
        total_games = int(merged["totalGames"])
        llr = _sprt_log_likelihood_ratio(
            successes=candidate_wins,
            trials=total_games,
            h0=args.gate_h0_win_rate,
            h1=args.gate_h1_win_rate,
        )
        accept_boundary, reject_boundary = _sprt_boundaries(alpha=args.gate_alpha, beta=args.gate_beta)

        decision_state = "running"
        decision_reason = "insufficient_evidence"
        side_gap = float(merged["sideGap"])
        next_completed_games_per_side = completed_games_per_side + batch_games_per_side

        if llr <= reject_boundary:
            decision_state = "rejected"
            decision_reason = "sprt_reject"
        elif llr >= accept_boundary and side_gap <= args.gate_max_side_gap:
            decision_state = "accepted"
            decision_reason = "sprt_accept"
        elif llr >= accept_boundary and side_gap > args.gate_max_side_gap:
            decision_state = "rejected"
            decision_reason = "side_gap_exceeded"
        elif next_completed_games_per_side >= args.gate_max_games_per_side:
            if side_gap > args.gate_max_side_gap:
                decision_state = "rejected"
                decision_reason = "max_games_reached_side_gap_exceeded"
            else:
                decision_state = "completed"
                decision_reason = "max_games_reached_inconclusive"

        history = payload.get("history")
        if not isinstance(history, list):
            raise SystemExit("Gate artifact history payload is invalid.")
        history.append(
            {
                "batchIndex": len(history) + 1,
                "seedStartIndex": batch_seed_start_index,
                "gamesPerSideBatch": batch_games_per_side,
                "gamesPerSideCompleted": next_completed_games_per_side,
                "candidateWins": int(merged["candidateWins"]),
                "opponentWins": int(merged["opponentWins"]),
                "draws": int(merged["draws"]),
                "totalGames": int(merged["totalGames"]),
                "candidateWinRate": float(merged["candidateWinRate"]),
                "candidateWinRateCi95": {
                    "low": float(merged["candidateWinRateCi95"]["low"]),
                    "high": float(merged["candidateWinRateCi95"]["high"]),
                },
                "sideGap": float(merged["sideGap"]),
                "logLikelihoodRatio": llr,
                "acceptBoundary": accept_boundary,
                "rejectBoundary": reject_boundary,
                "decision": decision_state,
                "decisionReason": decision_reason,
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            }
        )

        payload["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        payload["status"] = decision_state if decision_state in TERMINAL_GATE_STATUSES else "running"
        payload["results"] = merged
        payload["progress"] = {
            "gamesPerSideCompleted": next_completed_games_per_side,
            "gamesPerSideRemaining": max(0, args.gate_max_games_per_side - next_completed_games_per_side),
            "batchesCompleted": len(history),
            "nextSeedStartIndex": args.seed_start_index + next_completed_games_per_side,
        }
        payload["sprt"] = {
            "h0WinRate": args.gate_h0_win_rate,
            "h1WinRate": args.gate_h1_win_rate,
            "alpha": args.gate_alpha,
            "beta": args.gate_beta,
            "acceptBoundary": accept_boundary,
            "rejectBoundary": reject_boundary,
            "logLikelihoodRatio": llr,
        }
        payload["decision"] = {
            "state": decision_state,
            "reason": decision_reason,
            "maxSideGap": args.gate_max_side_gap,
        }

        _write_json_atomic(artifact_path, payload)
        _write_eval_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "mode": "gate",
                "status": str(payload["status"]),
                "artifact": str(artifact_path),
                "progress": dict(payload["progress"]),
                "decision": dict(payload["decision"]),
                "results": dict(payload["results"]) if isinstance(payload.get("results"), dict) else None,
            },
        )

        if decision_state in TERMINAL_GATE_STATUSES:
            return payload


def _resolve_gate_artifact_path(*, args: argparse.Namespace, output_path: Path) -> Path:
    if args.resume_from_artifact is not None:
        if args.out is not None and args.resume_from_artifact.resolve() != output_path.resolve():
            raise SystemExit("--resume-from-artifact and --out must reference the same path.")
        return args.resume_from_artifact
    return output_path


def _new_gate_payload(*, args: argparse.Namespace, artifact_path: Path) -> Dict[str, object]:
    accept_boundary, reject_boundary = _sprt_boundaries(alpha=args.gate_alpha, beta=args.gate_beta)
    return {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": "gate",
        "artifact": str(artifact_path),
        "config": {
            **_base_config_payload(args),
            "gate": {
                "h0WinRate": args.gate_h0_win_rate,
                "h1WinRate": args.gate_h1_win_rate,
                "alpha": args.gate_alpha,
                "beta": args.gate_beta,
                "batchGamesPerSide": args.gate_batch_games_per_side,
                "maxGamesPerSide": args.gate_max_games_per_side,
                "maxSideGap": args.gate_max_side_gap,
            },
        },
        "progress": {
            "gamesPerSideCompleted": 0,
            "gamesPerSideRemaining": args.gate_max_games_per_side,
            "batchesCompleted": 0,
            "nextSeedStartIndex": args.seed_start_index,
        },
        "sprt": {
            "h0WinRate": args.gate_h0_win_rate,
            "h1WinRate": args.gate_h1_win_rate,
            "alpha": args.gate_alpha,
            "beta": args.gate_beta,
            "acceptBoundary": accept_boundary,
            "rejectBoundary": reject_boundary,
            "logLikelihoodRatio": 0.0,
        },
        "decision": {
            "state": "running",
            "reason": "not_started",
            "maxSideGap": args.gate_max_side_gap,
        },
        "history": [],
        "results": None,
    }


def _validate_gate_resume_payload(*, args: argparse.Namespace, payload: Dict[str, object]) -> None:
    mode = payload.get("mode")
    if mode != "gate":
        raise SystemExit("Resume artifact mode mismatch: expected gate mode artifact.")

    status = str(payload.get("status", ""))
    if status and status not in TERMINAL_GATE_STATUSES and status != "running":
        raise SystemExit(f"Resume artifact has unknown status: {status!r}")

    config = payload.get("config")
    if not isinstance(config, dict):
        raise SystemExit("Resume artifact is missing config payload.")

    expected = _base_config_payload(args)
    expected_keys = (
        "mode",
        "seedPrefix",
        "seedStartIndex",
        "workers",
        "candidatePolicy",
        "opponentPolicy",
    )
    for key in expected_keys:
        if config.get(key) != expected.get(key):
            raise SystemExit(
                "Resume artifact config mismatch for "
                f"{key}: expected={expected.get(key)!r} actual={config.get(key)!r}"
            )

    gate_cfg = config.get("gate")
    if not isinstance(gate_cfg, dict):
        raise SystemExit("Resume artifact is missing gate config payload.")
    expected_gate = {
        "h0WinRate": args.gate_h0_win_rate,
        "h1WinRate": args.gate_h1_win_rate,
        "alpha": args.gate_alpha,
        "beta": args.gate_beta,
        "batchGamesPerSide": args.gate_batch_games_per_side,
        "maxGamesPerSide": args.gate_max_games_per_side,
        "maxSideGap": args.gate_max_side_gap,
    }
    for key, value in expected_gate.items():
        if gate_cfg.get(key) != value:
            raise SystemExit(
                "Resume artifact gate config mismatch for "
                f"{key}: expected={value!r} actual={gate_cfg.get(key)!r}"
            )


def _finalize_gate_at_cap(*, args: argparse.Namespace, payload: Dict[str, object]) -> Dict[str, object]:
    results = payload.get("results")
    if not isinstance(results, dict):
        payload["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        payload["status"] = "completed"
        payload["decision"] = {
            "state": "completed",
            "reason": "max_games_reached_no_results",
            "maxSideGap": args.gate_max_side_gap,
        }
        return payload

    side_gap = float(results["sideGap"])
    if side_gap > args.gate_max_side_gap:
        decision_state = "rejected"
        decision_reason = "max_games_reached_side_gap_exceeded"
    else:
        decision_state = "completed"
        decision_reason = "max_games_reached_inconclusive"

    payload["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = decision_state
    payload["decision"] = {
        "state": decision_state,
        "reason": decision_reason,
        "maxSideGap": args.gate_max_side_gap,
    }
    return payload


def _sprt_boundaries(*, alpha: float, beta: float) -> tuple[float, float]:
    accept = math.log((1.0 - beta) / alpha)
    reject = math.log(beta / (1.0 - alpha))
    return accept, reject


def _sprt_log_likelihood_ratio(*, successes: int, trials: int, h0: float, h1: float) -> float:
    if trials <= 0:
        return 0.0
    failures = trials - successes

    # Guard against log(0) when p is exactly 0 or 1.
    def _safe_log(x: float) -> float:
        epsilon = 1e-12
        return math.log(max(epsilon, min(1.0 - epsilon, x)))

    return (
        successes * (_safe_log(h1) - _safe_log(h0))
        + failures * (_safe_log(1.0 - h1) - _safe_log(1.0 - h0))
    )


def _base_config_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "mode": args.mode,
        "gamesPerSide": args.games_per_side,
        "seedPrefix": args.seed_prefix,
        "seedStartIndex": args.seed_start_index,
        "workers": args.workers,
        "progressEveryGames": args.progress_every_games,
        "progressLogMinutes": args.progress_log_minutes,
        "workerTorchThreads": args.worker_torch_threads,
        "workerTorchInteropThreads": args.worker_torch_interop_threads,
        "workerBlasThreads": args.worker_blas_threads,
        "candidatePolicy": args.candidate_policy,
        "opponentPolicy": args.opponent_policy,
        "search": {
            "worlds": args.search_worlds,
            "rollouts": args.search_rollouts,
            "depth": args.search_depth,
            "maxRootActions": args.search_max_root_actions,
            "rolloutEpsilon": args.search_rollout_epsilon,
        },
        "tdValue": {
            "checkpoint": str(args.td_value_checkpoint) if args.td_value_checkpoint else None,
            "worlds": args.td_worlds,
        },
        "tdSearch": {
            "valueCheckpoint": (
                str(args.td_search_value_checkpoint)
                if args.td_search_value_checkpoint
                else None
            ),
            "opponentCheckpoint": (
                str(args.td_search_opponent_checkpoint)
                if args.td_search_opponent_checkpoint
                else None
            ),
            "opponentTemperature": args.td_search_opponent_temperature,
            "sampleOpponentActions": args.td_search_sample_opponent_actions,
        },
    }


def _evaluate_results(
    *,
    args: argparse.Namespace,
    games_per_side: int,
    seed_start_index: int,
    seed_prefix: str,
    workers: int,
    progress_path: Path,
    progress_label: str,
) -> Dict[str, object]:
    started = time.perf_counter()
    _write_eval_progress(
        progress_path,
        {
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "mode": args.mode,
            "label": progress_label,
            "artifact": str(args.out) if args.out is not None else None,
            "gamesPerSide": games_per_side,
            "workers": workers,
            "completedShards": 0,
            "totalShards": max(1, min(workers, games_per_side)),
            "elapsedMinutes": 0.0,
        },
    )
    if workers == 1:
        summary = _run_eval_shard(
            games_per_side=games_per_side,
            seed_prefix=seed_prefix,
            seed_start_index=seed_start_index,
            candidate_policy=args.candidate_policy,
            opponent_policy=args.opponent_policy,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
            td_value_checkpoint=args.td_value_checkpoint,
            td_worlds=args.td_worlds,
            td_search_value_checkpoint=args.td_search_value_checkpoint,
            td_search_opponent_checkpoint=args.td_search_opponent_checkpoint,
            td_search_opponent_temperature=args.td_search_opponent_temperature,
            td_search_sample_opponent_actions=args.td_search_sample_opponent_actions,
            progress_every_games=args.progress_every_games,
            progress_log_minutes=args.progress_log_minutes,
            worker_torch_threads=args.worker_torch_threads,
            worker_torch_interop_threads=args.worker_torch_interop_threads,
            worker_blas_threads=args.worker_blas_threads,
        )
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        _write_eval_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "mode": args.mode,
                "label": progress_label,
                "gamesPerSide": games_per_side,
                "workers": 1,
                "completedShards": 1,
                "totalShards": 1,
                "elapsedMinutes": round(elapsed_minutes, 3),
                "candidateWins": summary["candidateWins"],
                "opponentWins": summary["opponentWins"],
                "draws": summary["draws"],
                "totalGames": summary["totalGames"],
                "candidateWinRate": summary["candidateWinRate"],
                "sideGap": summary["sideGap"],
            },
        )
        return summary

    worker_count = min(workers, games_per_side)
    if worker_count == 1:
        summary = _run_eval_shard(
            games_per_side=games_per_side,
            seed_prefix=seed_prefix,
            seed_start_index=seed_start_index,
            candidate_policy=args.candidate_policy,
            opponent_policy=args.opponent_policy,
            search_worlds=args.search_worlds,
            search_rollouts=args.search_rollouts,
            search_depth=args.search_depth,
            search_max_root_actions=args.search_max_root_actions,
            search_rollout_epsilon=args.search_rollout_epsilon,
            td_value_checkpoint=args.td_value_checkpoint,
            td_worlds=args.td_worlds,
            td_search_value_checkpoint=args.td_search_value_checkpoint,
            td_search_opponent_checkpoint=args.td_search_opponent_checkpoint,
            td_search_opponent_temperature=args.td_search_opponent_temperature,
            td_search_sample_opponent_actions=args.td_search_sample_opponent_actions,
            progress_every_games=args.progress_every_games,
            progress_log_minutes=args.progress_log_minutes,
            worker_torch_threads=args.worker_torch_threads,
            worker_torch_interop_threads=args.worker_torch_interop_threads,
            worker_blas_threads=args.worker_blas_threads,
        )
        elapsed_minutes = (time.perf_counter() - started) / 60.0
        _write_eval_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "mode": args.mode,
                "label": progress_label,
                "gamesPerSide": games_per_side,
                "workers": 1,
                "completedShards": 1,
                "totalShards": 1,
                "elapsedMinutes": round(elapsed_minutes, 3),
                "candidateWins": summary["candidateWins"],
                "opponentWins": summary["opponentWins"],
                "draws": summary["draws"],
                "totalGames": summary["totalGames"],
                "candidateWinRate": summary["candidateWinRate"],
                "sideGap": summary["sideGap"],
            },
        )
        return summary

    shard_sizes = _split_games(games_per_side, worker_count)

    payloads: List[Dict[str, object]] = []
    shard_seed_start_index = seed_start_index
    for shard_games in shard_sizes:
        payloads.append(
            {
                "gamesPerSide": shard_games,
                "seedPrefix": seed_prefix,
                "seedStartIndex": shard_seed_start_index,
                "candidatePolicy": args.candidate_policy,
                "opponentPolicy": args.opponent_policy,
                "searchWorlds": args.search_worlds,
                "searchRollouts": args.search_rollouts,
                "searchDepth": args.search_depth,
                "searchMaxRootActions": args.search_max_root_actions,
                "searchRolloutEpsilon": args.search_rollout_epsilon,
                "tdValueCheckpoint": (
                    str(args.td_value_checkpoint)
                    if args.td_value_checkpoint is not None
                    else None
                ),
                "tdWorlds": args.td_worlds,
                "tdSearchValueCheckpoint": (
                    str(args.td_search_value_checkpoint)
                    if args.td_search_value_checkpoint is not None
                    else None
                ),
                "tdSearchOpponentCheckpoint": (
                    str(args.td_search_opponent_checkpoint)
                    if args.td_search_opponent_checkpoint is not None
                    else None
                ),
                "tdSearchOpponentTemperature": args.td_search_opponent_temperature,
                "tdSearchSampleOpponentActions": args.td_search_sample_opponent_actions,
                "progressEveryGames": args.progress_every_games,
                "progressLogMinutes": args.progress_log_minutes,
                "workerTorchThreads": args.worker_torch_threads,
                "workerTorchInteropThreads": args.worker_torch_interop_threads,
                "workerBlasThreads": args.worker_blas_threads,
            }
        )
        shard_seed_start_index += shard_games

    shard_results: List[Dict[str, object]] = []
    shard_done = 0
    heartbeat_seconds = max(0.0, args.progress_log_minutes * 60.0)
    next_heartbeat = started + heartbeat_seconds if heartbeat_seconds > 0 else float("inf")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        pending = {executor.submit(_evaluate_shard, payload) for payload in payloads}
        while pending:
            done, pending = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                shard_results.append(result)
                shard_done += 1
                print(
                    json.dumps(
                        {
                            "event": "evalShardComplete",
                            "completedShards": shard_done,
                            "totalShards": worker_count,
                            "gamesPerSide": result["gamesPerSide"],
                            "candidateWins": result["candidateWins"],
                            "opponentWins": result["opponentWins"],
                            "draws": result["draws"],
                            "candidateWinRate": result["candidateWinRate"],
                        }
                    ),
                    flush=True,
                )
                elapsed_minutes = (time.perf_counter() - started) / 60.0
                _write_eval_progress(
                    progress_path,
                    {
                        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                        "status": "running",
                        "mode": args.mode,
                        "label": progress_label,
                        "gamesPerSide": games_per_side,
                        "workers": worker_count,
                        "completedShards": shard_done,
                        "totalShards": worker_count,
                        "elapsedMinutes": round(elapsed_minutes, 3),
                    },
                )

            now = time.perf_counter()
            if now >= next_heartbeat:
                elapsed_minutes = (now - started) / 60.0
                print(
                    f"[eval-suite] heartbeat label={progress_label} "
                    f"shards={shard_done}/{worker_count} elapsedMin={elapsed_minutes:.1f}",
                    flush=True,
                )
                _write_eval_progress(
                    progress_path,
                    {
                        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                        "status": "running",
                        "mode": args.mode,
                        "label": progress_label,
                        "gamesPerSide": games_per_side,
                        "workers": worker_count,
                        "completedShards": shard_done,
                        "totalShards": worker_count,
                        "elapsedMinutes": round(elapsed_minutes, 3),
                    },
                )
                next_heartbeat = now + heartbeat_seconds
    summary = _merge_shard_results(shard_results)
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    _write_eval_progress(
        progress_path,
        {
            "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "mode": args.mode,
            "label": progress_label,
            "gamesPerSide": games_per_side,
            "workers": worker_count,
            "completedShards": worker_count,
            "totalShards": worker_count,
            "elapsedMinutes": round(elapsed_minutes, 3),
            "candidateWins": summary["candidateWins"],
            "opponentWins": summary["opponentWins"],
            "draws": summary["draws"],
            "totalGames": summary["totalGames"],
            "candidateWinRate": summary["candidateWinRate"],
            "sideGap": summary["sideGap"],
        },
    )
    return summary


def _run_eval_shard(
    *,
    games_per_side: int,
    seed_prefix: str,
    seed_start_index: int,
    candidate_policy: str,
    opponent_policy: str,
    search_worlds: int,
    search_rollouts: int,
    search_depth: int,
    search_max_root_actions: int,
    search_rollout_epsilon: float,
    td_value_checkpoint: Path | None,
    td_worlds: int,
    td_search_value_checkpoint: Path | None,
    td_search_opponent_checkpoint: Path | None,
    td_search_opponent_temperature: float,
    td_search_sample_opponent_actions: bool,
    progress_every_games: int,
    progress_log_minutes: float,
    worker_torch_threads: int,
    worker_torch_interop_threads: int,
    worker_blas_threads: int,
) -> Dict[str, object]:
    _configure_worker_threads(
        torch_threads=worker_torch_threads,
        torch_interop_threads=worker_torch_interop_threads,
        blas_threads=worker_blas_threads,
    )
    policies = {candidate_policy.strip().lower(), opponent_policy.strip().lower()}
    search_config = SearchConfig(
        worlds=search_worlds,
        rollouts=search_rollouts,
        depth=search_depth,
        max_root_actions=search_max_root_actions,
        rollout_epsilon=search_rollout_epsilon,
    )
    td_value_config = (
        TDValuePolicyConfig(
            checkpoint_path=td_value_checkpoint,
            worlds=td_worlds,
        )
        if td_value_checkpoint is not None
        else None
    )
    td_search_config = (
        TDSearchPolicyConfig(
            value_checkpoint_path=td_search_value_checkpoint,
            opponent_checkpoint_path=td_search_opponent_checkpoint,
            worlds=search_worlds,
            rollouts=search_rollouts,
            depth=search_depth,
            max_root_actions=search_max_root_actions,
            rollout_epsilon=search_rollout_epsilon,
            opponent_temperature=td_search_opponent_temperature,
            sample_opponent_actions=td_search_sample_opponent_actions,
        )
        if "td-search" in policies
        else None
    )

    candidate = policy_from_name(
        candidate_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config,
    )
    opponent = policy_from_name(
        opponent_policy,
        search_config=search_config,
        td_value_config=td_value_config,
        td_search_config=td_search_config,
    )

    shard_started = time.perf_counter()

    try:
        with BridgeClient() as client:
            env = MagnateBridgeEnv(client=client)
            last_log_at: Dict[str, float] = {}

            def _on_progress(
                leg: str,
                completed: int,
                total: int,
                winners: Dict[str, int],
                wins_by_seat: Dict[str, int],
            ) -> None:
                now = time.perf_counter()
                elapsed = now - shard_started
                previous = last_log_at.get(leg, 0.0)
                progress_log_seconds = max(0.0, progress_log_minutes * 60.0)
                if completed < total and progress_log_seconds > 0 and (now - previous) < progress_log_seconds:
                    return
                last_log_at[leg] = now
                print(
                    json.dumps(
                        {
                            "event": "evalProgress",
                            "pid": os.getpid(),
                            "seedStartIndex": seed_start_index,
                            "gamesPerSideShard": games_per_side,
                            "leg": leg,
                            "completedGames": completed,
                            "totalGames": total,
                            "winners": winners,
                            "winsBySeat": wins_by_seat,
                            "elapsedMinutes": round(elapsed / 60.0, 2),
                        }
                    ),
                    flush=True,
                )

            from trainer.eval_suite import evaluate_side_swapped

            summary = evaluate_side_swapped(
                env=env,
                candidate_policy=candidate,
                opponent_policy=opponent,
                games_per_side=games_per_side,
                seed_prefix=seed_prefix,
                seed_start_index=seed_start_index,
                progress_every_games=progress_every_games,
                on_progress=_on_progress if progress_every_games > 0 else None,
            )
    finally:
        candidate.close()
        opponent.close()

    return summary.to_json()


def _evaluate_shard(payload: Dict[str, object]) -> Dict[str, object]:
    return _run_eval_shard(
        games_per_side=int(payload["gamesPerSide"]),
        seed_prefix=str(payload["seedPrefix"]),
        seed_start_index=int(payload["seedStartIndex"]),
        candidate_policy=str(payload["candidatePolicy"]),
        opponent_policy=str(payload["opponentPolicy"]),
        search_worlds=int(payload["searchWorlds"]),
        search_rollouts=int(payload["searchRollouts"]),
        search_depth=int(payload["searchDepth"]),
        search_max_root_actions=int(payload["searchMaxRootActions"]),
        search_rollout_epsilon=float(payload["searchRolloutEpsilon"]),
        td_value_checkpoint=(
            Path(str(payload["tdValueCheckpoint"]))
            if payload.get("tdValueCheckpoint")
            else None
        ),
        td_worlds=int(payload["tdWorlds"]),
        td_search_value_checkpoint=(
            Path(str(payload["tdSearchValueCheckpoint"]))
            if payload.get("tdSearchValueCheckpoint")
            else None
        ),
        td_search_opponent_checkpoint=(
            Path(str(payload["tdSearchOpponentCheckpoint"]))
            if payload.get("tdSearchOpponentCheckpoint")
            else None
        ),
        td_search_opponent_temperature=float(payload["tdSearchOpponentTemperature"]),
        td_search_sample_opponent_actions=bool(payload["tdSearchSampleOpponentActions"]),
        progress_every_games=int(payload["progressEveryGames"]),
        progress_log_minutes=float(payload["progressLogMinutes"]),
        worker_torch_threads=int(payload["workerTorchThreads"]),
        worker_torch_interop_threads=int(payload["workerTorchInteropThreads"]),
        worker_blas_threads=int(payload["workerBlasThreads"]),
    )


def _split_games(total_games_per_side: int, workers: int) -> List[int]:
    base = total_games_per_side // workers
    remainder = total_games_per_side % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _merge_shard_results(shard_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not shard_results:
        raise RuntimeError("No shard results to merge.")

    candidate_name = str(shard_results[0]["candidate"])
    opponent_name = str(shard_results[0]["opponent"])
    for result in shard_results[1:]:
        if str(result["candidate"]) != candidate_name or str(result["opponent"]) != opponent_name:
            raise RuntimeError("Shard results use different candidate/opponent policies.")

    games_per_side = sum(int(result["gamesPerSide"]) for result in shard_results)
    total_games = sum(int(result["totalGames"]) for result in shard_results)
    candidate_wins = sum(int(result["candidateWins"]) for result in shard_results)
    opponent_wins = sum(int(result["opponentWins"]) for result in shard_results)
    draws = sum(int(result["draws"]) for result in shard_results)

    winners = {
        "PlayerA": sum(int(result["winners"]["PlayerA"]) for result in shard_results),
        "PlayerB": sum(int(result["winners"]["PlayerB"]) for result in shard_results),
        "Draw": draws,
    }

    candidate_wins_as_a = 0
    candidate_wins_as_b = 0
    leg_a_games = 0
    leg_b_games = 0
    leg_a_turn_total = 0.0
    leg_b_turn_total = 0.0

    wins_by_seat_a: Dict[str, int] = {"PlayerA": 0, "PlayerB": 0}
    wins_by_seat_b: Dict[str, int] = {"PlayerA": 0, "PlayerB": 0}
    policy_by_seat_a: Dict[str, str] | None = None
    policy_by_seat_b: Dict[str, str] | None = None
    leg_a_winners: Dict[str, int] = defaultdict(int)
    leg_b_winners: Dict[str, int] = defaultdict(int)

    for result in shard_results:
        leg_a = result["legs"]["candidateAsPlayerA"]
        leg_b = result["legs"]["candidateAsPlayerB"]

        leg_a_games += int(leg_a["games"])
        leg_b_games += int(leg_b["games"])

        leg_a_turn_total += float(leg_a["averageTurn"]) * int(leg_a["games"])
        leg_b_turn_total += float(leg_b["averageTurn"]) * int(leg_b["games"])

        candidate_wins_as_a += int(leg_a["winners"]["PlayerA"])
        candidate_wins_as_b += int(leg_b["winners"]["PlayerB"])

        candidate_policy_by_seat_a = {
            "PlayerA": str(leg_a["policyBySeat"]["PlayerA"]),
            "PlayerB": str(leg_a["policyBySeat"]["PlayerB"]),
        }
        candidate_policy_by_seat_b = {
            "PlayerA": str(leg_b["policyBySeat"]["PlayerA"]),
            "PlayerB": str(leg_b["policyBySeat"]["PlayerB"]),
        }

        if policy_by_seat_a is None:
            policy_by_seat_a = candidate_policy_by_seat_a
        elif policy_by_seat_a != candidate_policy_by_seat_a:
            raise RuntimeError("Shard results use inconsistent policyBySeat for candidateAsPlayerA.")

        if policy_by_seat_b is None:
            policy_by_seat_b = candidate_policy_by_seat_b
        elif policy_by_seat_b != candidate_policy_by_seat_b:
            raise RuntimeError("Shard results use inconsistent policyBySeat for candidateAsPlayerB.")

        wins_by_seat_a["PlayerA"] += int(leg_a["winsBySeat"]["PlayerA"])
        wins_by_seat_a["PlayerB"] += int(leg_a["winsBySeat"]["PlayerB"])
        wins_by_seat_b["PlayerA"] += int(leg_b["winsBySeat"]["PlayerA"])
        wins_by_seat_b["PlayerB"] += int(leg_b["winsBySeat"]["PlayerB"])

        for key, value in leg_a["winners"].items():
            leg_a_winners[str(key)] += int(value)
        for key, value in leg_b["winners"].items():
            leg_b_winners[str(key)] += int(value)

    candidate_win_rate = candidate_wins / float(total_games)
    ci_low, ci_high = wilson_interval(candidate_wins, total_games)

    candidate_win_rate_as_a = candidate_wins_as_a / float(max(1, leg_a_games))
    candidate_win_rate_as_b = candidate_wins_as_b / float(max(1, leg_b_games))
    side_gap = abs(candidate_win_rate_as_a - candidate_win_rate_as_b)

    average_turn = (
        (leg_a_turn_total + leg_b_turn_total) / float(max(1, leg_a_games + leg_b_games))
    )

    return {
        "gamesPerSide": games_per_side,
        "totalGames": total_games,
        "candidate": candidate_name,
        "opponent": opponent_name,
        "winners": winners,
        "candidateWins": candidate_wins,
        "opponentWins": opponent_wins,
        "draws": draws,
        "candidateWinRate": candidate_win_rate,
        "candidateWinRateCi95": {
            "low": ci_low,
            "high": ci_high,
        },
        "candidateWinRateAsPlayerA": candidate_win_rate_as_a,
        "candidateWinRateAsPlayerB": candidate_win_rate_as_b,
        "sideGap": side_gap,
        "averageTurn": average_turn,
        "legs": {
            "candidateAsPlayerA": {
                "games": leg_a_games,
                "winners": dict(leg_a_winners),
                "winsBySeat": dict(wins_by_seat_a),
                "policyBySeat": dict(policy_by_seat_a or {}),
                "averageTurn": (leg_a_turn_total / float(max(1, leg_a_games))),
            },
            "candidateAsPlayerB": {
                "games": leg_b_games,
                "winners": dict(leg_b_winners),
                "winsBySeat": dict(wins_by_seat_b),
                "policyBySeat": dict(policy_by_seat_b or {}),
                "averageTurn": (leg_b_turn_total / float(max(1, leg_b_games))),
            },
        },
    }


def _configure_worker_threads(
    *,
    torch_threads: int,
    torch_interop_threads: int,
    blas_threads: int,
) -> None:
    thread_count = str(max(1, int(blas_threads)))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = thread_count
    try:
        import torch

        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(max(1, int(torch_interop_threads)))
    except Exception:
        # Keep eval runnable even if torch thread controls are unavailable.
        pass


def _resolve_progress_path(*, args: argparse.Namespace, output_path: Path) -> Path:
    if args.progress_out is not None:
        return args.progress_out
    return output_path.with_name("eval.progress.json")


def _write_eval_progress(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _default_output_path(seed_prefix: str, mode: str, candidate_policy: str, opponent_policy: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_seed = _slug(seed_prefix)
    safe_mode = _slug(mode)
    safe_candidate = _slug(candidate_policy)
    safe_opponent = _slug(opponent_policy)
    return Path("artifacts/evals") / (
        f"{stamp}-{safe_seed}-{safe_mode}-suite-{safe_candidate}-vs-{safe_opponent}.json"
    )


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_json_object(path: Path, *, label: str) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "unknown"


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


def _validate_policy_args(args: argparse.Namespace) -> None:
    policies = {args.candidate_policy.strip().lower(), args.opponent_policy.strip().lower()}
    if "td-value" in policies and args.td_value_checkpoint is None:
        raise SystemExit("--td-value-checkpoint is required when using td-value policy.")
    if "td-search" in policies:
        if args.td_search_value_checkpoint is None:
            raise SystemExit("--td-search-value-checkpoint is required when using td-search policy.")
        if args.td_search_opponent_checkpoint is None:
            raise SystemExit("--td-search-opponent-checkpoint is required when using td-search policy.")


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.seed_start_index < 0:
        raise SystemExit("--seed-start-index must be >= 0.")
    if args.progress_every_games < 0:
        raise SystemExit("--progress-every-games must be >= 0.")
    if args.progress_log_minutes < 0:
        raise SystemExit("--progress-log-minutes must be >= 0.")
    if args.worker_torch_threads <= 0:
        raise SystemExit("--worker-torch-threads must be > 0.")
    if args.worker_torch_interop_threads <= 0:
        raise SystemExit("--worker-torch-interop-threads must be > 0.")
    if args.worker_blas_threads <= 0:
        raise SystemExit("--worker-blas-threads must be > 0.")

    if args.mode == "certify":
        if args.games_per_side <= 0:
            raise SystemExit("--games-per-side must be > 0.")
        if args.resume_from_artifact is not None:
            raise SystemExit("--resume-from-artifact is supported only in --mode gate.")
        return

    if args.gate_h0_win_rate < 0.0 or args.gate_h0_win_rate > 1.0:
        raise SystemExit("--gate-h0-win-rate must be in [0, 1].")
    if args.gate_h1_win_rate < 0.0 or args.gate_h1_win_rate > 1.0:
        raise SystemExit("--gate-h1-win-rate must be in [0, 1].")
    if args.gate_h0_win_rate >= args.gate_h1_win_rate:
        raise SystemExit("--gate-h0-win-rate must be strictly less than --gate-h1-win-rate.")
    if args.gate_alpha <= 0.0 or args.gate_alpha >= 1.0:
        raise SystemExit("--gate-alpha must be in (0, 1).")
    if args.gate_beta <= 0.0 or args.gate_beta >= 1.0:
        raise SystemExit("--gate-beta must be in (0, 1).")
    if args.gate_batch_games_per_side <= 0:
        raise SystemExit("--gate-batch-games-per-side must be > 0.")
    if args.gate_max_games_per_side <= 0:
        raise SystemExit("--gate-max-games-per-side must be > 0.")
    if args.gate_batch_games_per_side > args.gate_max_games_per_side:
        raise SystemExit("--gate-batch-games-per-side must be <= --gate-max-games-per-side.")
    if args.gate_max_side_gap < 0.0 or args.gate_max_side_gap > 1.0:
        raise SystemExit("--gate-max-side-gap must be in [0, 1].")


if __name__ == "__main__":
    raise SystemExit(main())
