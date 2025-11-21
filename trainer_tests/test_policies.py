from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.encoding import OBSERVATION_DIM
from trainer.policies import (
    DeterminizedSearchPolicy,
    SearchConfig,
    TDDeterminizedSearchPolicy,
    TDSearchPolicyConfig,
    TDValuePolicy,
    TDValuePolicyConfig,
    policy_from_name,
)
from trainer.td.checkpoint import save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet
from trainer.encoding import ACTION_FEATURE_DIM
from trainer.td.checkpoint import save_opponent_checkpoint


class PolicyFactoryTests(unittest.TestCase):
    def test_policy_factory_creates_random_policy(self) -> None:
        policy = policy_from_name("random")
        self.assertEqual(policy.name, "random")

    def test_policy_factory_creates_heuristic_policy(self) -> None:
        policy = policy_from_name("heuristic")
        self.assertEqual(policy.name, "heuristic")

    def test_policy_factory_creates_search_policy(self) -> None:
        config = SearchConfig(worlds=1, rollouts=1, depth=1, max_root_actions=1, rollout_epsilon=0.0)
        policy = policy_from_name("search", search_config=config)
        try:
            self.assertIsInstance(policy, DeterminizedSearchPolicy)
        finally:
            policy.close()

    def test_policy_factory_rejects_unknown_policy(self) -> None:
        with self.assertRaises(ValueError):
            policy_from_name("legacy")

    def test_policy_factory_creates_td_value_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=checkpoint_path,
            )
            policy = policy_from_name(
                "td-value",
                td_value_config=TDValuePolicyConfig(
                    checkpoint_path=checkpoint_path,
                    worlds=2,
                ),
            )
            try:
                self.assertIsInstance(policy, TDValuePolicy)
            finally:
                policy.close()

    def test_policy_factory_creates_td_search_policy(self) -> None:
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
            policy = policy_from_name(
                "td-search",
                td_search_config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                ),
            )
            try:
                self.assertIsInstance(policy, TDDeterminizedSearchPolicy)
            finally:
                policy.close()

    def test_td_search_config_requires_opponent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            with self.assertRaises(TypeError):
                TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                )

    def test_td_search_config_accepts_opponent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=16,
                ),
                output_path=opponent_path,
            )
            policy = policy_from_name(
                "td-search",
                td_search_config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                ),
            )
            try:
                self.assertIsInstance(policy, TDDeterminizedSearchPolicy)
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
