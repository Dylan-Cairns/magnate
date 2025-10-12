from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from trainer.ppo_model import CandidateActorCritic, load_ppo_checkpoint, save_ppo_checkpoint
from trainer.ppo_training import PpoConfig


class PpoScaffoldTests(unittest.TestCase):
    def test_config_defaults_are_positive(self) -> None:
        config = PpoConfig()
        self.assertGreater(config.total_episodes, 0)
        self.assertGreater(config.episodes_per_update, 0)
        self.assertGreater(config.learning_rate, 0.0)

    def test_model_outputs_expected_shapes(self) -> None:
        model = CandidateActorCritic(observation_dim=3, action_feature_dim=4, hidden_dim=16)
        observation = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        action_features = torch.tensor([[0.1, 0.0, 0.2, 0.3], [0.0, 0.4, 0.1, 0.2]], dtype=torch.float32)

        logits = model.policy_logits_tensor(observation, action_features)
        value = model.value_tensor(observation)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(value.shape), ())

    def test_checkpoint_round_trip(self) -> None:
        model = CandidateActorCritic(observation_dim=3, action_feature_dim=4, hidden_dim=16)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ppo.pt"
            save_ppo_checkpoint(model, path, metadata={"suite": "ppo-scaffold"})
            loaded, payload = load_ppo_checkpoint(path)

        self.assertEqual(loaded.observation_dim, model.observation_dim)
        self.assertEqual(loaded.action_feature_dim, model.action_feature_dim)
        self.assertEqual(str(payload.get("checkpointType")), "magnate_ppo_policy_v1")


if __name__ == "__main__":
    unittest.main()
