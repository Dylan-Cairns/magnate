from __future__ import annotations

import unittest

import torch
from trainer.td.models import OpponentModel, ValueNet


class TDModelTests(unittest.TestCase):
    def test_value_net_forward_shapes(self) -> None:
        model = ValueNet(observation_dim=6, hidden_dim=12)
        batch = torch.randn(4, 6)
        output = model(batch)
        self.assertEqual(list(output.shape), [4])

        single = torch.randn(6)
        output_single = model(single)
        self.assertEqual(list(output_single.shape), [1])

    def test_opponent_model_distribution_matches_action_count(self) -> None:
        model = OpponentModel(observation_dim=6, action_feature_dim=4, hidden_dim=12)
        observation = [0.1] * 6
        action_features = [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        distribution = model.action_distribution(
            observation=observation,
            action_features=action_features,
        )
        self.assertEqual(int(distribution.probs.numel()), 3)
        self.assertAlmostEqual(float(distribution.probs.sum().item()), 1.0, places=6)

    def test_invalid_dimensions_raise(self) -> None:
        with self.assertRaises(ValueError):
            ValueNet(observation_dim=0)
        with self.assertRaises(ValueError):
            OpponentModel(observation_dim=4, action_feature_dim=0)


if __name__ == "__main__":
    unittest.main()
