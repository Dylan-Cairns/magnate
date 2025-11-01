from __future__ import annotations

import random
import unittest

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv
from trainer.policies import DeterminizedMctsPolicy, MctsConfig


class MctsPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_deterministic_choice_for_fixed_rng(self) -> None:
        policy = DeterminizedMctsPolicy(
            config=MctsConfig(
                worlds=2,
                simulations=8,
                depth=6,
                max_root_actions=4,
                c_puct=1.0,
            )
        )
        try:
            step_result = self.env.reset(seed="mcts-policy-deterministic", first_player="PlayerA")
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

    def test_mcts_requires_state_payload(self) -> None:
        policy = DeterminizedMctsPolicy(
            config=MctsConfig(
                worlds=1,
                simulations=4,
                depth=3,
                max_root_actions=3,
                c_puct=1.0,
            )
        )
        try:
            step_result = self.env.reset(seed="mcts-policy-requires-state", first_player="PlayerA")
            legal = self.env.legal_actions()
            with self.assertRaises(ValueError):
                policy.choose_action_key(
                    step_result.view,
                    legal.actions,
                    random.Random(0),
                )
        finally:
            policy.close()

    def test_root_ranking_keeps_all_legal_actions(self) -> None:
        policy = DeterminizedMctsPolicy(
            config=MctsConfig(
                worlds=1,
                simulations=6,
                depth=3,
                max_root_actions=1,
                c_puct=1.0,
            )
        )
        try:
            step_result = self.env.reset(seed="mcts-policy-root-ranking", first_player="PlayerA")
            legal = self.env.legal_actions()

            ranked = policy._ranked_root_actions(legal.actions)
            self.assertEqual(len(ranked), len(legal.actions))
            self.assertEqual(
                {action.action_key for action in ranked},
                {action.action_key for action in legal.actions},
            )
        finally:
            policy.close()

    def test_step_cache_reuses_transition(self) -> None:
        policy = DeterminizedMctsPolicy(
            config=MctsConfig(
                worlds=1,
                simulations=6,
                depth=3,
                max_root_actions=2,
                c_puct=1.0,
            )
        )
        try:
            step_result = self.env.reset(seed="mcts-policy-step-cache", first_player="PlayerA")
            legal = self.env.legal_actions()
            action_key = legal.actions[0].action_key

            first = policy._step_state(step_result.state, action_key)
            self.assertEqual(len(policy._step_cache), 1)

            second = policy._step_state(step_result.state, action_key)
            self.assertEqual(len(policy._step_cache), 1)
            self.assertEqual(first, second)
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
