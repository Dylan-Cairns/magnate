from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.behavior_cloning import BehaviorCloningModel, save_behavior_cloning_checkpoint
from trainer.policies import BehaviorCloningPolicy, policy_from_name


class PolicyFactoryTests(unittest.TestCase):
    def test_bc_policy_requires_checkpoint_path(self) -> None:
        with self.assertRaises(ValueError):
            policy_from_name("bc")

    def test_policy_factory_loads_bc_checkpoint(self) -> None:
        model = BehaviorCloningModel.zeros(observation_dim=2, action_feature_dim=2)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bc.json"
            save_behavior_cloning_checkpoint(model, path)
            policy = policy_from_name("bc", checkpoint_path=path)

        self.assertIsInstance(policy, BehaviorCloningPolicy)


if __name__ == "__main__":
    unittest.main()
