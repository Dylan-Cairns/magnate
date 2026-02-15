from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.behavior_cloning import (
    BehaviorCloningConfig,
    load_behavior_cloning_checkpoint,
    save_behavior_cloning_checkpoint,
    train_behavior_cloning,
)
from trainer.types import DecisionSample


def _sample(
    observation: list[float],
    action_features: list[list[float]],
    action_index: int,
) -> DecisionSample:
    return DecisionSample(
        seed="synthetic",
        turn=1,
        phase="ActionWindow",
        active_player_id="PlayerA",
        action_key=f"a{action_index}",
        action_id="develop-outright",
        action_index=action_index,
        observation=observation,
        action_features=action_features,
        winner="Draw",
        reward=0.0,
    )


class BehaviorCloningTests(unittest.TestCase):
    def test_training_improves_accuracy_on_synthetic_data(self) -> None:
        base_actions = [[1.0, 0.0], [0.0, 1.0]]
        samples = [
            _sample([1.0, 0.0], base_actions, 0),
            _sample([0.0, 1.0], base_actions, 1),
        ] * 20

        model, summary = train_behavior_cloning(
            samples=samples,
            config=BehaviorCloningConfig(
                epochs=40,
                learning_rate=0.5,
                l2=0.0,
                seed=0,
            ),
        )

        self.assertLess(summary.initial.accuracy, summary.final.accuracy)
        self.assertGreaterEqual(summary.final.accuracy, 0.99)
        self.assertEqual(model.choose_action_index([1.0, 0.0], base_actions), 0)
        self.assertEqual(model.choose_action_index([0.0, 1.0], base_actions), 1)

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        base_actions = [[1.0, 0.0], [0.0, 1.0]]
        samples = [
            _sample([1.0, 0.0], base_actions, 0),
            _sample([0.0, 1.0], base_actions, 1),
        ] * 10
        model, _ = train_behavior_cloning(
            samples=samples,
            config=BehaviorCloningConfig(epochs=20, learning_rate=0.3, l2=0.0, seed=1),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.json"
            save_behavior_cloning_checkpoint(model, path, metadata={"source": "unit-test"})
            loaded = load_behavior_cloning_checkpoint(path)

        self.assertEqual(loaded.observation_dim, model.observation_dim)
        self.assertEqual(loaded.action_feature_dim, model.action_feature_dim)
        self.assertEqual(loaded.choose_action_index([1.0, 0.0], base_actions), 0)
        self.assertEqual(loaded.choose_action_index([0.0, 1.0], base_actions), 1)


if __name__ == "__main__":
    unittest.main()
