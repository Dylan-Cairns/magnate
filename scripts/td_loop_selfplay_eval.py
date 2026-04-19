from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

from scripts.opponent_pool import PoolCheckpoint
from scripts.td_loop_common import LoopCheckpoint
from scripts.td_loop_eval_common import (
    EvalRow,
    PromotionThresholds,
    evaluate_promotion_gate,
)


def _build_eval_command_vs_search(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    checkpoint: LoopCheckpoint,
    out_path: Path,
    seed_prefix: str,
    seed_start_index: int,
    workers: int,
    games_per_side: int,
) -> list[str]:
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
        "search",
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


def _build_eval_command_vs_incumbent(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    checkpoint: LoopCheckpoint,
    incumbent: PoolCheckpoint,
    out_path: Path,
    seed_prefix: str,
    seed_start_index: int,
    workers: int,
    games_per_side: int,
) -> list[str]:
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
        "td-search",
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
        "--candidate-td-search-value-checkpoint",
        str(checkpoint.value_path),
        "--candidate-td-search-opponent-checkpoint",
        str(checkpoint.opponent_path),
        "--opponent-td-search-value-checkpoint",
        str(incumbent.value_path),
        "--opponent-td-search-opponent-checkpoint",
        str(incumbent.opponent_path),
        "--td-search-opponent-temperature",
        str(args.eval_td_search_opponent_temperature),
    ]
    if args.eval_td_search_sample_opponent_actions:
        command.append("--td-search-sample-opponent-actions")
    return command


def _promotion_decision(
    *,
    baseline_eval: EvalRow,
    baseline_windows: Sequence[EvalRow],
    incumbent_eval: EvalRow,
    incumbent_windows: Sequence[EvalRow],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    baseline_gate = evaluate_promotion_gate(
        eval_row=baseline_eval,
        eval_windows=baseline_windows,
        thresholds=PromotionThresholds(
            min_win_rate=args.promotion_min_win_rate,
            max_side_gap=args.promotion_max_side_gap,
            min_ci_low=args.promotion_min_ci_low,
            max_window_side_gap=args.promotion_max_window_side_gap,
        ),
    )
    incumbent_gate = evaluate_promotion_gate(
        eval_row=incumbent_eval,
        eval_windows=incumbent_windows,
        thresholds=PromotionThresholds(
            min_win_rate=args.promotion_incumbent_min_win_rate,
            max_side_gap=args.promotion_incumbent_max_side_gap,
            min_ci_low=args.promotion_incumbent_min_ci_low,
            max_window_side_gap=args.promotion_incumbent_max_window_side_gap,
        ),
    )
    promoted = bool(baseline_gate["passed"] and incumbent_gate["passed"])
    return {
        "promoted": promoted,
        "checks": {
            "baselineVsSearch": baseline_gate["checks"],
            "candidateVsIncumbent": incumbent_gate["checks"],
        },
        "windowChecks": {
            "baselineVsSearch": baseline_gate["windowChecks"],
            "candidateVsIncumbent": incumbent_gate["windowChecks"],
        },
        "reason": "dual_gate_passed" if promoted else "dual_gate_failed",
    }
