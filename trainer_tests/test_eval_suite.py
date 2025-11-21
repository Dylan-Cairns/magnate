from __future__ import annotations

import unittest

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.eval_suite import evaluate_side_swapped
from trainer.policies import HeuristicPolicy, RandomLegalPolicy


class EvalSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_side_swapped_summary_contains_ci_and_side_gap(self) -> None:
        summary = evaluate_side_swapped(
            env=self.env,
            candidate_policy=HeuristicPolicy(),
            opponent_policy=RandomLegalPolicy(),
            games_per_side=2,
            seed_prefix="eval-suite-test",
        )
        self.assertEqual(summary.games_per_side, 2)
        self.assertEqual(summary.total_games, 4)
        self.assertEqual(summary.candidate_wins + summary.opponent_wins + summary.draws, 4)
        self.assertGreaterEqual(summary.candidate_win_rate_ci95[0], 0.0)
        self.assertLessEqual(summary.candidate_win_rate_ci95[1], 1.0)
        self.assertGreaterEqual(summary.side_gap, 0.0)
        self.assertLessEqual(summary.side_gap, 1.0)


if __name__ == "__main__":
    unittest.main()
