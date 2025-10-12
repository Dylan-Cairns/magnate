from __future__ import annotations

import unittest

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.evaluate import evaluate_matchup
from trainer.policies import HeuristicPolicy, RandomLegalPolicy


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_evaluate_matchup_runs_games_and_summarizes(self) -> None:
        summary = evaluate_matchup(
            env=self.env,
            policy_player_a=HeuristicPolicy(),
            policy_player_b=RandomLegalPolicy(),
            games=2,
            seed_prefix="eval-test",
        )
        self.assertEqual(summary.games, 2)
        self.assertEqual(
            summary.winners["PlayerA"] + summary.winners["PlayerB"] + summary.winners["Draw"],
            2,
        )
        self.assertGreater(summary.average_turn, 0.0)


if __name__ == "__main__":
    unittest.main()

