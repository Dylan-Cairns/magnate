from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import unittest

from scripts.run_td_loop import (
    EvalRow,
    _recommended_cloud_worker_count,
    _evaluate_certify_result,
    _promotion_decision,
)


class RunTdLoopPromotionTests(unittest.TestCase):
    def _args(self) -> Namespace:
        return Namespace(
            certify_min_win_rate=0.55,
            certify_max_side_gap=0.08,
            certify_min_ci_low=0.50,
        )

    def test_certify_result_passes_when_all_thresholds_pass(self) -> None:
        row = EvalRow(
            artifact=Path("certify-search.json"),
            opponent_policy="search",
            candidate_win_rate=0.60,
            ci_low=0.52,
            ci_high=0.67,
            side_gap=0.05,
            candidate_wins=480,
            opponent_wins=320,
            draws=0,
            total_games=800,
        )
        comparison = _evaluate_certify_result(row=row, args=self._args())
        self.assertTrue(comparison.passed)
        self.assertTrue(all(comparison.checks.values()))

    def test_certify_result_fails_when_side_gap_too_high(self) -> None:
        row = EvalRow(
            artifact=Path("certify-search.json"),
            opponent_policy="search",
            candidate_win_rate=0.60,
            ci_low=0.52,
            ci_high=0.67,
            side_gap=0.11,
            candidate_wins=480,
            opponent_wins=320,
            draws=0,
            total_games=800,
        )
        comparison = _evaluate_certify_result(row=row, args=self._args())
        self.assertFalse(comparison.passed)
        self.assertFalse(comparison.checks["maxSideGap"])

    def test_promotion_requires_gate_accept(self) -> None:
        result = _promotion_decision(
            gate_decision="rejected",
            gate_reason="sprt_reject",
            certify={"ran": False, "overallPassed": False, "comparisons": []},
        )
        self.assertFalse(result["promoted"])
        self.assertIn("gate_rejected", str(result["reason"]))

    def test_promotion_requires_certify_pass(self) -> None:
        result = _promotion_decision(
            gate_decision="accepted",
            gate_reason="sprt_accept",
            certify={
                "ran": True,
                "overallPassed": False,
                "comparisons": [
                    {"opponentPolicy": "search", "passed": False},
                    {"opponentPolicy": "heuristic", "passed": True},
                ],
            },
        )
        self.assertFalse(result["promoted"])
        self.assertEqual(result["reason"], "certify_failed")
        self.assertEqual(result["failedOpponents"], ["search"])

    def test_promotion_passes_when_gate_and_certify_pass(self) -> None:
        result = _promotion_decision(
            gate_decision="accepted",
            gate_reason="sprt_accept",
            certify={
                "ran": True,
                "overallPassed": True,
                "comparisons": [
                    {"opponentPolicy": "search", "passed": True},
                    {"opponentPolicy": "heuristic", "passed": True},
                ],
            },
        )
        self.assertTrue(result["promoted"])
        self.assertEqual(result["reason"], "gate_and_certify_passed")

    def test_recommended_cloud_worker_count_scales_by_profile(self) -> None:
        self.assertEqual(_recommended_cloud_worker_count(8), 6)
        self.assertEqual(_recommended_cloud_worker_count(16), 12)
        self.assertEqual(_recommended_cloud_worker_count(32), 24)


if __name__ == "__main__":
    unittest.main()
