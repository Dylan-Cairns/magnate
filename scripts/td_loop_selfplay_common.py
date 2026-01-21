from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Sequence

from scripts.opponent_pool import PoolCheckpoint
from scripts.td_loop_common import LoopCheckpoint, read_json


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


def _read_eval_row(path: Path, *, opponent_policy: str) -> EvalRow:
    payload = read_json(path, label=f"eval artifact {path.name}")
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


def _promotion_decision(
    *,
    baseline_eval: EvalRow,
    baseline_windows: Sequence[EvalRow],
    incumbent_eval: EvalRow,
    incumbent_windows: Sequence[EvalRow],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    baseline_window_checks = [
        {
            "artifact": str(row.artifact),
            "sideGap": row.side_gap,
            "maxWindowSideGap": row.side_gap <= args.promotion_max_window_side_gap,
        }
        for row in baseline_windows
    ]
    baseline_checks = {
        "minWinRate": baseline_eval.candidate_win_rate >= args.promotion_min_win_rate,
        "maxSideGap": baseline_eval.side_gap <= args.promotion_max_side_gap,
        "minCiLow": baseline_eval.ci_low >= args.promotion_min_ci_low,
        "maxWindowSideGap": all(
            window_check["maxWindowSideGap"] for window_check in baseline_window_checks
        ),
    }

    incumbent_window_checks = [
        {
            "artifact": str(row.artifact),
            "sideGap": row.side_gap,
            "maxWindowSideGap": row.side_gap <= args.promotion_incumbent_max_window_side_gap,
        }
        for row in incumbent_windows
    ]
    incumbent_checks = {
        "minWinRate": incumbent_eval.candidate_win_rate >= args.promotion_incumbent_min_win_rate,
        "maxSideGap": incumbent_eval.side_gap <= args.promotion_incumbent_max_side_gap,
        "minCiLow": incumbent_eval.ci_low >= args.promotion_incumbent_min_ci_low,
        "maxWindowSideGap": all(
            window_check["maxWindowSideGap"] for window_check in incumbent_window_checks
        ),
    }
    promoted = bool(all(baseline_checks.values()) and all(incumbent_checks.values()))
    return {
        "promoted": promoted,
        "checks": {
            "baselineVsSearch": baseline_checks,
            "candidateVsIncumbent": incumbent_checks,
        },
        "windowChecks": {
            "baselineVsSearch": baseline_window_checks,
            "candidateVsIncumbent": incumbent_window_checks,
        },
        "reason": "dual_gate_passed" if promoted else "dual_gate_failed",
    }


def _eval_payload(
    seed_indices: Sequence[int], rows: Sequence[EvalRow], pooled: EvalRow
) -> Dict[str, Any]:
    return {
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
            for seed_start_index, row in zip(seed_indices, rows)
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
    }


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
