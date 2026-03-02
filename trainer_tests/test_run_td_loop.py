from __future__ import annotations

from pathlib import Path
import unittest

from scripts.run_td_loop import (
    EvalRow,
    _evaluate_benchmark_comparison,
    _intervals_overlap,
)


class RunTdLoopBenchmarkTests(unittest.TestCase):
    def test_intervals_overlap_detects_overlap_and_separation(self) -> None:
        self.assertTrue(
            _intervals_overlap(begin_low=0.40, begin_high=0.60, end_low=0.55, end_high=0.70)
        )
        self.assertFalse(
            _intervals_overlap(begin_low=0.40, begin_high=0.50, end_low=0.51, end_high=0.70)
        )

    def test_benchmark_comparison_passes_when_all_checks_pass(self) -> None:
        begin = EvalRow(
            step=200,
            artifact=Path("begin.json"),
            candidate_win_rate=0.50,
            ci_low=0.45,
            ci_high=0.55,
            side_gap=0.10,
            candidate_wins=100,
            opponent_wins=100,
            draws=0,
            total_games=200,
        )
        end = EvalRow(
            step=2000,
            artifact=Path("end.json"),
            candidate_win_rate=0.60,
            ci_low=0.56,
            ci_high=0.64,
            side_gap=0.04,
            candidate_wins=120,
            opponent_wins=80,
            draws=0,
            total_games=200,
        )
        comparison = _evaluate_benchmark_comparison(
            opponent_policy="search",
            begin=begin,
            end=end,
            min_delta_win_rate=0.05,
            min_end_win_rate=0.55,
            max_end_side_gap=0.08,
            require_ci_separation=True,
        )
        self.assertTrue(comparison.passed)
        self.assertTrue(all(comparison.checks.values()))
        self.assertAlmostEqual(comparison.delta_candidate_win_rate, 0.10)

    def test_benchmark_comparison_fails_ci_separation_when_required(self) -> None:
        begin = EvalRow(
            step=200,
            artifact=Path("begin.json"),
            candidate_win_rate=0.52,
            ci_low=0.45,
            ci_high=0.58,
            side_gap=0.05,
            candidate_wins=104,
            opponent_wins=96,
            draws=0,
            total_games=200,
        )
        end = EvalRow(
            step=2000,
            artifact=Path("end.json"),
            candidate_win_rate=0.56,
            ci_low=0.50,
            ci_high=0.62,
            side_gap=0.04,
            candidate_wins=112,
            opponent_wins=88,
            draws=0,
            total_games=200,
        )
        comparison = _evaluate_benchmark_comparison(
            opponent_policy="heuristic",
            begin=begin,
            end=end,
            min_delta_win_rate=0.02,
            min_end_win_rate=0.55,
            max_end_side_gap=0.08,
            require_ci_separation=True,
        )
        self.assertFalse(comparison.passed)
        self.assertFalse(comparison.checks["ciSeparation"])
        self.assertTrue(comparison.ci_overlap)


if __name__ == "__main__":
    unittest.main()
