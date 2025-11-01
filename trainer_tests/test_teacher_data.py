from __future__ import annotations

import unittest

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import HeuristicPolicy, RandomLegalPolicy
from trainer.teacher_data import collect_teacher_samples


class TeacherDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_collect_teacher_samples_both_players(self) -> None:
        samples, summary = collect_teacher_samples(
            env=self.env,
            teacher_policy=HeuristicPolicy(),
            opponent_policy=None,
            games=2,
            seed_prefix="teacher-data-both",
            teacher_player_ids={"PlayerA", "PlayerB"},
            rng_seed=123,
        )

        self.assertEqual(summary.games, 2)
        self.assertGreater(summary.samples, 0)
        self.assertEqual(summary.samples, len(samples))
        self.assertGreater(summary.decisions_by_player["PlayerA"], 0)
        self.assertGreater(summary.decisions_by_player["PlayerB"], 0)

        for sample in samples:
            self.assertIn(sample.active_player_id, ("PlayerA", "PlayerB"))
            self.assertIn(sample.winner, ("PlayerA", "PlayerB", "Draw"))
            if sample.winner == "Draw":
                self.assertEqual(sample.reward, 0.0)
            elif sample.winner == sample.active_player_id:
                self.assertEqual(sample.reward, 1.0)
            else:
                self.assertEqual(sample.reward, -1.0)

    def test_collect_teacher_samples_single_player_filter(self) -> None:
        samples, summary = collect_teacher_samples(
            env=self.env,
            teacher_policy=HeuristicPolicy(),
            opponent_policy=RandomLegalPolicy(),
            games=3,
            seed_prefix="teacher-data-a-only",
            teacher_player_ids={"PlayerA"},
            rng_seed=0,
        )

        self.assertEqual(summary.games, 3)
        self.assertGreater(summary.samples, 0)
        self.assertEqual(summary.decisions_by_player["PlayerB"], 0)
        self.assertTrue(all(sample.active_player_id == "PlayerA" for sample in samples))


if __name__ == "__main__":
    unittest.main()
