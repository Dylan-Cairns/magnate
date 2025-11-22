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
        self.assertEqual(summary.policy_name_by_seat["PlayerA"], "heuristic")
        self.assertEqual(summary.policy_name_by_seat["PlayerB"], "random")

    def test_evaluate_matchup_tracks_seat_wins_when_policy_names_match(self) -> None:
        summary = evaluate_matchup(
            env=self.env,
            policy_player_a=RandomLegalPolicy(),
            policy_player_b=RandomLegalPolicy(),
            games=4,
            seed_prefix="eval-test-same-policy",
        )
        self.assertEqual(summary.policy_name_by_seat["PlayerA"], "random")
        self.assertEqual(summary.policy_name_by_seat["PlayerB"], "random")
        self.assertEqual(set(summary.wins_by_seat.keys()), {"PlayerA", "PlayerB"})
        self.assertEqual(
            summary.wins_by_seat["PlayerA"] + summary.wins_by_seat["PlayerB"],
            summary.winners["PlayerA"] + summary.winners["PlayerB"],
        )


if __name__ == "__main__":
    unittest.main()
