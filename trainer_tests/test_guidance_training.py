from __future__ import annotations

import unittest

from trainer.guidance_training import GuidanceConfig, train_guidance_model
from trainer.types import DecisionSample


class GuidanceTrainingTests(unittest.TestCase):
    def test_trains_model_from_decision_samples(self) -> None:
        samples = [
            DecisionSample(
                seed="s-0",
                turn=1,
                phase="ActionWindow",
                active_player_id="PlayerA",
                action_key="a0",
                action_id="develop-outright",
                action_index=0,
                observation=[1.0, 0.0, 0.0],
                action_features=[[1.0, 0.0], [0.0, 1.0]],
                winner="PlayerA",
                reward=1.0,
            ),
            DecisionSample(
                seed="s-1",
                turn=1,
                phase="ActionWindow",
                active_player_id="PlayerB",
                action_key="a1",
                action_id="sell-card",
                action_index=1,
                observation=[0.0, 1.0, 0.0],
                action_features=[[1.0, 0.0], [0.0, 1.0]],
                winner="PlayerB",
                reward=-1.0,
            ),
            DecisionSample(
                seed="s-2",
                turn=2,
                phase="ActionWindow",
                active_player_id="PlayerA",
                action_key="a0",
                action_id="develop-deed",
                action_index=0,
                observation=[1.0, 1.0, 0.0],
                action_features=[[1.0, 0.0], [0.0, 1.0]],
                winner="PlayerA",
                reward=1.0,
            ),
            DecisionSample(
                seed="s-3",
                turn=2,
                phase="ActionWindow",
                active_player_id="PlayerB",
                action_key="a1",
                action_id="trade",
                action_index=1,
                observation=[0.0, 0.0, 1.0],
                action_features=[[1.0, 0.0], [0.0, 1.0]],
                winner="PlayerB",
                reward=-1.0,
            ),
        ]
        config = GuidanceConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            hidden_dim=8,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            seed=11,
        )
        model, summary = train_guidance_model(samples=samples, config=config)
        self.assertEqual(summary.sample_count, len(samples))
        self.assertEqual(summary.observation_dim, 3)
        self.assertEqual(summary.action_feature_dim, 2)
        self.assertEqual(len(summary.history), 2)
        self.assertIsNotNone(summary.final)
        self.assertEqual(model.observation_dim, 3)
        self.assertEqual(model.action_feature_dim, 2)


if __name__ == "__main__":
    unittest.main()
