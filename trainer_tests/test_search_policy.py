from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from trainer.bridge_client import BridgeClient
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.env import MagnateBridgeEnv
from trainer.policies import DeterminizedSearchPolicy, SearchConfig, policy_from_name
from trainer.ppo_model import CandidateActorCritic, save_ppo_checkpoint


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

    def test_search_guidance_checkpoint_path_is_usable(self) -> None:
        config = SearchConfig(
            worlds=1,
            rollouts=1,
            depth=2,
            max_root_actions=3,
            rollout_epsilon=0.0,
        )
        model = CandidateActorCritic(
            observation_dim=OBSERVATION_DIM,
            action_feature_dim=ACTION_FEATURE_DIM,
            hidden_dim=32,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "guidance.pt"
            save_ppo_checkpoint(model, checkpoint)
            policy = policy_from_name(
                "search",
                search_config=config,
                search_guidance_checkpoint=checkpoint,
            )
            try:
                step_result = self.env.reset(seed="search-guidance-checkpoint", first_player="PlayerA")
                legal = self.env.legal_actions()
                action_key = policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(17),
                    state=step_result.state,
                )
                self.assertIn(action_key, {entry.action_key for entry in legal.actions})
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
