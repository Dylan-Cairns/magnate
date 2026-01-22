from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Sequence

from scripts.td_loop_common import read_json


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
class PromotionThresholds:
    min_win_rate: float
    max_side_gap: float
    min_ci_low: float
    max_window_side_gap: float


def read_eval_row(path: Path, *, opponent_policy: str) -> EvalRow:
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


def pool_eval_rows(*, eval_rows: Sequence[EvalRow], opponent_policy: str) -> EvalRow:
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


def build_eval_payload(
    seed_indices: Sequence[int],
    rows: Sequence[EvalRow],
    pooled: EvalRow,
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
            for seed_start_index, row in zip(seed_indices, rows, strict=False)
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


def evaluate_promotion_gate(
    *,
    eval_row: EvalRow,
    eval_windows: Sequence[EvalRow],
    thresholds: PromotionThresholds,
) -> Dict[str, Any]:
    window_checks = [
        {
            "artifact": str(row.artifact),
            "sideGap": row.side_gap,
            "maxWindowSideGap": row.side_gap <= thresholds.max_window_side_gap,
        }
        for row in eval_windows
    ]
    checks = {
        "minWinRate": eval_row.candidate_win_rate >= thresholds.min_win_rate,
        "maxSideGap": eval_row.side_gap <= thresholds.max_side_gap,
        "minCiLow": eval_row.ci_low >= thresholds.min_ci_low,
        "maxWindowSideGap": all(window_check["maxWindowSideGap"] for window_check in window_checks),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "windowChecks": window_checks,
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
