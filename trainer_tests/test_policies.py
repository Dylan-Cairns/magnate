from __future__ import annotations

import unittest

from trainer.policies import DeterminizedSearchPolicy, SearchConfig, policy_from_name


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


if __name__ == "__main__":
    unittest.main()
