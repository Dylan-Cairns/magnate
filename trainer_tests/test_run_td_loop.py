from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import unittest

from scripts.run_td_loop import (
    EvalRow,
    _recommended_cloud_worker_count,
    _promotion_decision,
)


class RunTdLoopPromotionTests(unittest.TestCase):
    def _args(self) -> Namespace:
        return Namespace(
            promotion_min_win_rate=0.55,
            promotion_max_side_gap=0.08,
            promotion_min_ci_low=0.50,
            promotion_max_window_side_gap=0.10,
        )

    def test_promotion_passes_when_all_thresholds_pass(self) -> None:
        row = EvalRow(
            artifact=Path("promotion_eval.json"),
            opponent_policy="search",
            candidate_win_rate=0.60,
            ci_low=0.52,
            ci_high=0.67,
            side_gap=0.05,
            candidate_wins=480,
            opponent_wins=320,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.61,
            candidate_win_rate_as_player_b=0.59,
        )
        result = _promotion_decision(eval_row=row, eval_windows=[row], args=self._args())
        self.assertTrue(result["promoted"])
        self.assertTrue(all(result["checks"].values()))

    def test_promotion_fails_when_side_gap_too_high(self) -> None:
        row = EvalRow(
            artifact=Path("promotion_eval.json"),
            opponent_policy="search",
            candidate_win_rate=0.60,
            ci_low=0.52,
            ci_high=0.67,
            side_gap=0.11,
            candidate_wins=480,
            opponent_wins=320,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.70,
            candidate_win_rate_as_player_b=0.49,
        )
        result = _promotion_decision(eval_row=row, eval_windows=[row], args=self._args())
        self.assertFalse(result["promoted"])
        self.assertFalse(result["checks"]["maxSideGap"])
        self.assertFalse(result["checks"]["maxWindowSideGap"])

    def test_promotion_fails_when_ci_low_too_small(self) -> None:
        row = EvalRow(
            artifact=Path("promotion_eval.json"),
            opponent_policy="search",
            candidate_win_rate=0.58,
            ci_low=0.40,
            ci_high=0.67,
            side_gap=0.04,
            candidate_wins=464,
            opponent_wins=336,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.60,
            candidate_win_rate_as_player_b=0.56,
        )
        result = _promotion_decision(eval_row=row, eval_windows=[row], args=self._args())
        self.assertFalse(result["promoted"])
        self.assertFalse(result["checks"]["minCiLow"])

    def test_promotion_reason_reflects_pooled_eval_flow(self) -> None:
        row = EvalRow(
            artifact=Path("promotion_eval.json"),
            opponent_policy="search",
            candidate_win_rate=0.50,
            ci_low=0.46,
            ci_high=0.54,
            side_gap=0.03,
            candidate_wins=400,
            opponent_wins=400,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.51,
            candidate_win_rate_as_player_b=0.49,
        )
        result = _promotion_decision(eval_row=row, eval_windows=[row], args=self._args())
        self.assertEqual(result["reason"], "pooled_eval_failed")

    def test_recommended_cloud_worker_count_scales_by_profile(self) -> None:
        self.assertEqual(_recommended_cloud_worker_count(8), 4)
        self.assertEqual(_recommended_cloud_worker_count(16), 8)
        self.assertEqual(_recommended_cloud_worker_count(32), 16)


if __name__ == "__main__":
    unittest.main()
