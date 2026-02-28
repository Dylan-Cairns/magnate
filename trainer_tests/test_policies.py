from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.policies import (
    DeterminizedMctsPolicy,
    DeterminizedSearchPolicy,
    MctsConfig,
    SearchConfig,
    policy_from_name,
)
from trainer.ppo_model import CandidateActorCritic, save_ppo_checkpoint


class PolicyFactoryTests(unittest.TestCase):
    def test_ppo_policy_requires_checkpoint_path(self) -> None:
        with self.assertRaises(ValueError):
            policy_from_name("ppo")

    def test_policy_factory_loads_ppo_checkpoint(self) -> None:
        model = CandidateActorCritic(observation_dim=2, action_feature_dim=2, hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ppo.pt"
            save_ppo_checkpoint(model, path)
            policy = policy_from_name("ppo", checkpoint_path=path)
        self.assertEqual(policy.name, f"ppo:{path.name}")

    def test_policy_factory_creates_search_policy(self) -> None:
        config = SearchConfig(worlds=1, rollouts=1, depth=1, max_root_actions=1, rollout_epsilon=0.0)
        policy = policy_from_name("search", search_config=config)
        try:
            self.assertIsInstance(policy, DeterminizedSearchPolicy)
        finally:
            policy.close()

    def test_policy_factory_creates_search_policy_with_guidance(self) -> None:
        config = SearchConfig(worlds=1, rollouts=1, depth=1, max_root_actions=1, rollout_epsilon=0.0)
        model = CandidateActorCritic(observation_dim=2, action_feature_dim=2, hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "guidance.pt"
            save_ppo_checkpoint(model, path)
            policy = policy_from_name(
                "search",
                search_config=config,
                search_guidance_checkpoint=path,
            )
        try:
            self.assertIsInstance(policy, DeterminizedSearchPolicy)
            self.assertIsNotNone(policy.guidance_model)
        finally:
            policy.close()

    def test_policy_factory_creates_mcts_policy(self) -> None:
        config = MctsConfig(worlds=1, simulations=4, depth=2, max_root_actions=2, c_puct=1.0)
        policy = policy_from_name("mcts", mcts_config=config)
        try:
            self.assertIsInstance(policy, DeterminizedMctsPolicy)
        finally:
            policy.close()

    def test_policy_factory_creates_mcts_policy_with_guidance(self) -> None:
        config = MctsConfig(worlds=1, simulations=4, depth=2, max_root_actions=2, c_puct=1.0)
        model = CandidateActorCritic(observation_dim=2, action_feature_dim=2, hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "guidance.pt"
            save_ppo_checkpoint(model, path)
            policy = policy_from_name(
                "mcts",
                mcts_config=config,
                mcts_guidance_checkpoint=path,
            )
        try:
            self.assertIsInstance(policy, DeterminizedMctsPolicy)
            self.assertIsNotNone(policy.guidance_model)
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
