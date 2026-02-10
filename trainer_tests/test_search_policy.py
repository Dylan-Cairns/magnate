from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.env import MagnateBridgeEnv
from trainer.policies import (
    DeterminizedSearchPolicy,
    SearchConfig,
    TDDeterminizedSearchPolicy,
    TDSearchPolicyConfig,
)
from trainer.td.checkpoint import save_opponent_checkpoint, save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet


class SearchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_deterministic_choice_for_fixed_rng(self) -> None:
        policy = DeterminizedSearchPolicy(
            config=SearchConfig(
                worlds=2,
                rollouts=1,
                depth=4,
                max_root_actions=4,
                rollout_epsilon=0.0,
            )
        )
        try:
            step_result = self.env.reset(seed="search-policy-deterministic", first_player="PlayerA")
            legal = self.env.legal_actions()

            action_one = policy.choose_action_key(
                step_result.view,
                legal.actions,
                random.Random(777),
                state=step_result.state,
            )
            action_two = policy.choose_action_key(
                step_result.view,
                legal.actions,
                random.Random(777),
                state=step_result.state,
            )

            legal_keys = {entry.action_key for entry in legal.actions}
            self.assertIn(action_one, legal_keys)
            self.assertEqual(action_one, action_two)
        finally:
            policy.close()

    def test_search_accumulates_cache_hits_when_limits_are_enabled(self) -> None:
        policy = DeterminizedSearchPolicy(
            config=SearchConfig(
                worlds=2,
                rollouts=1,
                depth=4,
                max_root_actions=4,
                rollout_epsilon=0.0,
                transition_cache_limit=256,
                legal_actions_cache_limit=256,
                observation_cache_limit=256,
            )
        )
        try:
            step_result = self.env.reset(seed="search-policy-cache-hits", first_player="PlayerA")
            legal = self.env.legal_actions()

            action_one = policy.choose_action_key(
                step_result.view,
                legal.actions,
                random.Random(777),
                state=step_result.state,
            )
            stats_after_first = policy._forward_model.cache_stats()

            action_two = policy.choose_action_key(
                step_result.view,
                legal.actions,
                random.Random(777),
                state=step_result.state,
            )
            stats_after_second = policy._forward_model.cache_stats()

            legal_keys = {entry.action_key for entry in legal.actions}
            self.assertIn(action_one, legal_keys)
            self.assertEqual(action_one, action_two)
            self.assertGreater(stats_after_first.transition_misses, 0)
            self.assertGreater(stats_after_first.legal_actions_misses, 0)
            self.assertGreater(stats_after_first.observation_misses, 0)
            self.assertGreater(stats_after_second.transition_hits, stats_after_first.transition_hits)
            self.assertGreater(
                stats_after_second.legal_actions_hits,
                stats_after_first.legal_actions_hits,
            )
            self.assertGreater(
                stats_after_second.observation_hits,
                stats_after_first.observation_hits,
            )
        finally:
            policy.close()

    def test_search_requires_state_payload(self) -> None:
        policy = DeterminizedSearchPolicy(
            config=SearchConfig(
                worlds=1,
                rollouts=1,
                depth=2,
                max_root_actions=3,
                rollout_epsilon=0.0,
            )
        )
        try:
            step_result = self.env.reset(seed="search-policy-requires-state", first_player="PlayerA")
            legal = self.env.legal_actions()
            with self.assertRaises(ValueError):
                policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(0),
                )
        finally:
            policy.close()

    def test_td_search_is_deterministic_for_fixed_rng(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=32,
                ),
                output_path=opponent_path,
            )
            policy = TDDeterminizedSearchPolicy(
                config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=2,
                    max_root_actions=3,
                    rollout_epsilon=0.0,
                )
            )
            try:
                step_result = self.env.reset(seed="td-search-policy-deterministic", first_player="PlayerA")
                legal = self.env.legal_actions()

                action_one = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(19),
                    state=step_result.state,
                )
                action_two = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(19),
                    state=step_result.state,
                )

                legal_keys = {entry.action_key for entry in legal.actions}
                self.assertIn(action_one, legal_keys)
                self.assertEqual(action_one, action_two)
            finally:
                policy.close()

    def test_td_search_accumulates_cache_hits_when_limits_are_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=32,
                ),
                output_path=opponent_path,
            )
            policy = TDDeterminizedSearchPolicy(
                config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=2,
                    max_root_actions=3,
                    rollout_epsilon=0.0,
                    transition_cache_limit=256,
                    legal_actions_cache_limit=256,
                    observation_cache_limit=256,
                )
            )
            try:
                step_result = self.env.reset(seed="td-search-policy-cache-hits", first_player="PlayerA")
                legal = self.env.legal_actions()

                action_one = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(19),
                    state=step_result.state,
                )
                stats_after_first = policy._forward_model.cache_stats()

                action_two = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(19),
                    state=step_result.state,
                )
                stats_after_second = policy._forward_model.cache_stats()

                legal_keys = {entry.action_key for entry in legal.actions}
                self.assertIn(action_one, legal_keys)
                self.assertEqual(action_one, action_two)
                self.assertGreater(stats_after_first.transition_misses, 0)
                self.assertGreater(stats_after_first.legal_actions_misses, 0)
                self.assertGreater(stats_after_second.transition_hits, stats_after_first.transition_hits)
                self.assertGreater(
                    stats_after_second.legal_actions_hits,
                    stats_after_first.legal_actions_hits,
                )
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
