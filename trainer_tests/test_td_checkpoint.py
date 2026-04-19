from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from trainer.td.checkpoint import (
    load_opponent_checkpoint,
    load_value_checkpoint,
    save_opponent_checkpoint,
    save_value_checkpoint,
)
from trainer.td.models import OpponentModel, ValueNet


class TDCheckpointTests(unittest.TestCase):
    def test_value_checkpoint_round_trip(self) -> None:
        model = ValueNet(observation_dim=8, hidden_dim=16)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(model=model, output_path=path, metadata={"run": "test"})
            loaded_model, payload = load_value_checkpoint(path=path)

        self.assertEqual(int(payload["observationDim"]), 8)
        self.assertEqual(int(payload["hiddenDim"]), 16)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, loaded_model.state_dict()[key]))

    def test_opponent_checkpoint_round_trip(self) -> None:
        model = OpponentModel(observation_dim=8, action_feature_dim=5, hidden_dim=16)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "opponent.pt"
            save_opponent_checkpoint(model=model, output_path=path, metadata={"run": "test"})
            loaded_model, payload = load_opponent_checkpoint(path=path)

        self.assertEqual(int(payload["observationDim"]), 8)
        self.assertEqual(int(payload["actionFeatureDim"]), 5)
        self.assertEqual(int(payload["hiddenDim"]), 16)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, loaded_model.state_dict()[key]))

    def test_loading_with_wrong_checkpoint_type_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.pt"
            torch.save(
                {
                    "checkpointType": "unknown",
                    "encodingVersion": 2,
                    "observationDim": 8,
                    "hiddenDim": 16,
                    "stateDict": {},
                },
                path,
            )
            with self.assertRaises(ValueError):
                load_value_checkpoint(path=path)


if __name__ == "__main__":
    unittest.main()
