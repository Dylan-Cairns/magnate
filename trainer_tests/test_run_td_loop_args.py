from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from scripts.run_td_loop import parse_args


class RunTdLoopArgSurfaceTests(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["run_td_loop.py", *args]):
            return parse_args()

    def test_overnight_command_arg_surface_parses(self) -> None:
        parsed = self._parse(
            "--cloud",
            "--cloud-vcpus",
            "16",
            "--run-label",
            "td-loop-r2-overnight",
            "--chunks-per-loop",
            "3",
            "--collect-games",
            "800",
            "--train-steps",
            "30000",
            "--eval-games-per-side",
            "200",
            "--eval-opponent-policy",
            "search",
            "--promotion-min-ci-low",
            "0.5",
            "--progress-heartbeat-minutes",
            "30",
            "--eval-progress-log-minutes",
            "30",
            "--promotion-manifest-key",
            "td-loop-r2-overnight-promoted",
        )
        self.assertTrue(parsed.cloud)
        self.assertEqual(parsed.cloud_vcpus, 16)
        self.assertEqual(parsed.chunks_per_loop, 3)
        self.assertEqual(parsed.collect_games, 800)
        self.assertEqual(parsed.train_steps, 30000)
        self.assertEqual(parsed.eval_games_per_side, 200)
        self.assertEqual(parsed.eval_opponent_policy, "search")
        self.assertAlmostEqual(parsed.promotion_min_ci_low, 0.5)
        self.assertEqual(parsed.promotion_manifest_key, "td-loop-r2-overnight-promoted")

    def test_smoke_command_arg_surface_parses(self) -> None:
        parsed = self._parse(
            "--run-label",
            "td-loop-smoke",
            "--chunks-per-loop",
            "1",
            "--collect-games",
            "12",
            "--collect-search-worlds",
            "2",
            "--collect-search-depth",
            "8",
            "--collect-search-max-root-actions",
            "4",
            "--train-steps",
            "30",
            "--train-save-every-steps",
            "15",
            "--train-hidden-dim",
            "64",
            "--train-value-batch-size",
            "32",
            "--train-opponent-batch-size",
            "16",
            "--eval-games-per-side",
            "10",
            "--eval-opponent-policy",
            "search",
            "--eval-workers",
            "1",
            "--eval-search-worlds",
            "2",
            "--eval-search-depth",
            "8",
            "--eval-search-max-root-actions",
            "4",
            "--promotion-min-ci-low",
            "0.5",
        )
        self.assertEqual(parsed.run_label, "td-loop-smoke")
        self.assertEqual(parsed.chunks_per_loop, 1)
        self.assertEqual(parsed.collect_games, 12)
        self.assertEqual(parsed.eval_games_per_side, 10)
        self.assertEqual(parsed.eval_opponent_policy, "search")


if __name__ == "__main__":
    unittest.main()

