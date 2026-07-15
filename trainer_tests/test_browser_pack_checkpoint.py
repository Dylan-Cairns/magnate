from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from scripts.export_browser_td_root_pack import export_td_root_checkpoint_pack
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td.browser_pack_checkpoint import (
    BrowserPackCheckpointError,
    reconstruct_browser_td_root_checkpoints,
)
from trainer.td.checkpoint import (
    load_opponent_checkpoint,
    load_value_checkpoint,
    save_opponent_checkpoint,
    save_value_checkpoint,
)
from trainer.td.models import OpponentModel, ValueNet


class BrowserPackCheckpointTests(unittest.TestCase):
    def test_reconstructs_exact_trainer_checkpoints_with_metadata_and_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            value_model, opponent_model, manifest_path = self._export_test_pack(root)

            result = reconstruct_browser_td_root_checkpoints(
                manifest_path=manifest_path,
                output_dir=root / "reconstructed",
            )

            reconstructed_value, value_payload = load_value_checkpoint(
                path=Path(result["valueCheckpoint"])
            )
            reconstructed_opponent, opponent_payload = load_opponent_checkpoint(
                path=Path(result["opponentCheckpoint"])
            )
            for key, source_tensor in value_model.state_dict().items():
                self.assertTrue(
                    torch.equal(source_tensor, reconstructed_value.state_dict()[key]),
                    key,
                )
            for key, source_tensor in opponent_model.state_dict().items():
                self.assertTrue(
                    torch.equal(source_tensor, reconstructed_opponent.state_dict()[key]),
                    key,
                )

            self.assertNotIn("optimizerStateDict", value_payload)
            self.assertNotIn("optimizerStateDict", opponent_payload)
            self.assertEqual(value_payload["metadata"]["step"], 314)
            self.assertEqual(value_payload["metadata"]["modelRole"], "value")
            self.assertEqual(opponent_payload["metadata"]["step"], 314)
            self.assertEqual(opponent_payload["metadata"]["modelRole"], "opponent")
            value_provenance = cast(
                dict[str, object],
                value_payload["metadata"]["browserPackReconstruction"],
            )
            opponent_provenance = cast(
                dict[str, object],
                opponent_payload["metadata"]["browserPackReconstruction"],
            )
            self.assertIsInstance(value_provenance, dict)
            self.assertIsInstance(opponent_provenance, dict)
            self.assertEqual(value_provenance["packId"], "round-trip-pack")
            self.assertEqual(opponent_provenance["packId"], "round-trip-pack")

            self.assertEqual(result["valueParity"]["tensorCount"], 6)
            self.assertEqual(result["opponentParity"]["tensorCount"], 10)
            self.assertLessEqual(
                result["valueParity"]["maxOutputAbsoluteDifference"],
                1e-6,
            )
            self.assertLessEqual(
                result["opponentParity"]["maxOutputAbsoluteDifference"],
                1e-6,
            )
            self.assertEqual(len(result["manifestSha256"]), 64)
            self.assertEqual(len(result["weightsSha256"]), 64)

            observation = torch.linspace(-1.0, 1.0, OBSERVATION_DIM)
            action_features = torch.stack(
                (
                    torch.linspace(-0.5, 0.5, ACTION_FEATURE_DIM),
                    torch.linspace(0.5, -0.5, ACTION_FEATURE_DIM),
                )
            )
            with torch.inference_mode():
                self.assertTrue(
                    torch.equal(value_model(observation), reconstructed_value(observation))
                )
                self.assertTrue(
                    torch.equal(
                        opponent_model.logits_tensor(observation, action_features),
                        reconstructed_opponent.logits_tensor(observation, action_features),
                    )
                )

    def test_rejects_tensor_shape_that_does_not_match_trainer_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, _, manifest_path = self._export_test_pack(root)
            weights_path = manifest_path.parent / "weights.json"
            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            weights["valueTensors"]["encoder.4.bias"]["shape"] = [2]
            weights_path.write_text(json.dumps(weights), encoding="utf-8")

            with self.assertRaisesRegex(
                BrowserPackCheckpointError,
                "encoder.4.bias.shape mismatch",
            ):
                reconstruct_browser_td_root_checkpoints(
                    manifest_path=manifest_path,
                    output_dir=root / "reconstructed",
                )
            self.assertFalse((root / "reconstructed").exists())

    def test_refuses_to_replace_existing_checkpoints_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            _, _, manifest_path = self._export_test_pack(root)
            output_dir = root / "reconstructed"
            first = reconstruct_browser_td_root_checkpoints(
                manifest_path=manifest_path,
                output_dir=output_dir,
            )
            value_before = Path(first["valueCheckpoint"]).read_bytes()
            opponent_before = Path(first["opponentCheckpoint"]).read_bytes()

            with self.assertRaisesRegex(BrowserPackCheckpointError, "already exists"):
                reconstruct_browser_td_root_checkpoints(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                )

            self.assertEqual(Path(first["valueCheckpoint"]).read_bytes(), value_before)
            self.assertEqual(Path(first["opponentCheckpoint"]).read_bytes(), opponent_before)

    @staticmethod
    def _export_test_pack(root: Path) -> tuple[ValueNet, OpponentModel, Path]:
        torch.manual_seed(20260714)
        value_model = ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=8)
        opponent_model = OpponentModel(
            observation_dim=OBSERVATION_DIM,
            action_feature_dim=ACTION_FEATURE_DIM,
            hidden_dim=8,
        )
        value_checkpoint = root / "source" / "value.pt"
        opponent_checkpoint = root / "source" / "opponent.pt"
        save_value_checkpoint(
            model=value_model,
            output_path=value_checkpoint,
            metadata={"step": 314, "modelRole": "value"},
        )
        save_opponent_checkpoint(
            model=opponent_model,
            output_path=opponent_checkpoint,
            metadata={"step": 314, "modelRole": "opponent"},
        )
        export_result = export_td_root_checkpoint_pack(
            value_checkpoint_path=value_checkpoint,
            opponent_checkpoint_path=opponent_checkpoint,
            output_root=root / "model-packs",
            pack_id="round-trip-pack",
            label="Round Trip Pack",
            set_default=False,
            source_run_id="round-trip-run",
        )
        return value_model, opponent_model, Path(export_result["manifest"])


if __name__ == "__main__":
    unittest.main()
