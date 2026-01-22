from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.resume_td_loop_run import (
    _latest_checkpoint_from_train_summary,
    _promotion_decision,
)
from scripts.td_loop_eval_common import EvalRow


class ResumeTdLoopRunTests(unittest.TestCase):
    def _args(self) -> Namespace:
        return Namespace(
            promotion_min_win_rate=0.55,
            promotion_max_side_gap=0.08,
            promotion_min_ci_low=0.50,
            promotion_max_window_side_gap=0.10,
        )

    def test_latest_checkpoint_from_train_summary_uses_latest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path = root / "step-2000.value.pt"
            opponent_path = root / "step-2000.opponent.pt"
            value_path.write_text("value", encoding="utf-8")
            opponent_path.write_text("opponent", encoding="utf-8")
            payload = {
                "results": {
                    "checkpoints": [
                        {
                            "step": 1000,
                            "value": str(root / "step-1000.value.pt"),
                        },
                        {
                            "step": 2000,
                            "value": str(value_path),
                            "opponent": str(opponent_path),
                        },
                    ]
                }
            }

            checkpoint = _latest_checkpoint_from_train_summary(payload)

        self.assertEqual(checkpoint.step, 2000)
        self.assertEqual(checkpoint.value_path, value_path)
        self.assertEqual(checkpoint.opponent_path, opponent_path)

    def test_promotion_decision_uses_shared_gate_checks(self) -> None:
        row = EvalRow(
            artifact=Path("promotion_eval.json"),
            opponent_policy="search",
            candidate_win_rate=0.54,
            ci_low=0.49,
            ci_high=0.59,
            side_gap=0.09,
            candidate_wins=216,
            opponent_wins=184,
            draws=0,
            total_games=400,
            candidate_win_rate_as_player_a=0.585,
            candidate_win_rate_as_player_b=0.495,
        )

        result = _promotion_decision(eval_row=row, eval_windows=[row], args=self._args())

        self.assertFalse(result["promoted"])
        self.assertFalse(result["checks"]["minWinRate"])
        self.assertFalse(result["checks"]["maxSideGap"])
        self.assertFalse(result["checks"]["minCiLow"])
        self.assertTrue(result["checks"]["maxWindowSideGap"])
        self.assertEqual(result["reason"], "pooled_eval_failed")


if __name__ == "__main__":
    unittest.main()
