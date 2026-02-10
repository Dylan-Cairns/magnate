from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.encoding import OBSERVATION_DIM
from trainer.env import MagnateBridgeEnv
from trainer.policies import TDValuePolicy, TDValuePolicyConfig
from trainer.td.checkpoint import save_value_checkpoint
from trainer.td.models import ValueNet


class TDValuePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_td_value_accumulates_cache_hits_when_limits_are_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=checkpoint_path,
            )
            policy = TDValuePolicy(
                config=TDValuePolicyConfig(
                    checkpoint_path=checkpoint_path,
                    worlds=2,
                    transition_cache_limit=256,
                    legal_actions_cache_limit=256,
                    observation_cache_limit=256,
                )
            )
            try:
                step_result = self.env.reset(seed="td-value-policy-cache-hits", first_player="PlayerA")
                legal = self.env.legal_actions()

                action_one = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(31),
                    state=step_result.state,
                )
                stats_after_first = policy._forward_model.cache_stats()

                action_two = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(31),
                    state=step_result.state,
                )
                stats_after_second = policy._forward_model.cache_stats()

                legal_keys = {entry.action_key for entry in legal.actions}
                self.assertIn(action_one, legal_keys)
                self.assertEqual(action_one, action_two)
                self.assertGreater(stats_after_first.transition_misses, 0)
                self.assertGreater(stats_after_first.observation_misses, 0)
                self.assertGreater(stats_after_second.transition_hits, stats_after_first.transition_hits)
                self.assertGreater(
                    stats_after_second.observation_hits,
                    stats_after_first.observation_hits,
                )
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
