from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_browser_model_pack import export_value_checkpoint_pack, resolve_checkpoint
from trainer.encoding import ENCODING_VERSION, OBSERVATION_DIM
from trainer.td.checkpoint import save_value_checkpoint
from trainer.td.models import ValueNet


class ExportBrowserModelPackTests(unittest.TestCase):
    def test_exports_pack_manifest_weights_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "value.pt"
            model = ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16)
            save_value_checkpoint(model=model, output_path=checkpoint_path, metadata={"step": 12})

            output_root = root / "public" / "model-packs"
            result = export_value_checkpoint_pack(
                checkpoint_path=checkpoint_path,
                output_root=output_root,
                pack_id="td-value-test-pack",
                label="TD Value Test",
                set_default=True,
            )

            manifest_path = Path(result["manifest"])
            weights_path = Path(result["weights"])
            index_path = Path(result["index"])

            self.assertTrue(manifest_path.exists())
            self.assertTrue(weights_path.exists())
            self.assertTrue(index_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["packId"], "td-value-test-pack")
            self.assertEqual(manifest["model"]["modelType"], "td-value-v1")
            self.assertEqual(manifest["model"]["encodingVersion"], ENCODING_VERSION)
            self.assertEqual(manifest["model"]["observationDim"], OBSERVATION_DIM)
            self.assertEqual(manifest["model"]["weightsPath"], "weights.json")

            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["defaultPackId"], "td-value-test-pack")
            self.assertTrue(any(pack["id"] == "td-value-test-pack" for pack in index["packs"]))

            weights = json.loads(weights_path.read_text(encoding="utf-8"))
            self.assertEqual(weights["schemaVersion"], 1)
            self.assertIn("encoder.0.weight", weights["tensors"])
            self.assertIn("encoder.4.bias", weights["tensors"])

    def test_resolve_latest_promoted_checkpoint_from_loop_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_root = root / "artifacts" / "td_loops"
            run_dir = artifact_root / "20260305-100000Z-sample-run"
            run_dir.mkdir(parents=True)

            checkpoint_path = run_dir / "value-step-0000123.pt"
            model = ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=8)
            save_value_checkpoint(model=model, output_path=checkpoint_path, metadata={"step": 123})

            summary_payload = {
                "promotion": {"promoted": True},
                "chunks": [
                    {
                        "latestCheckpoint": {
                            "step": 123,
                            "value": str(checkpoint_path),
                            "opponent": None,
                        }
                    }
                ],
            }
            (run_dir / "loop.summary.json").write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )

            resolved = resolve_checkpoint(
                explicit_checkpoint=None,
                latest_promoted=True,
                artifact_root=artifact_root,
            )
            self.assertEqual(Path(resolved["checkpointPath"]), checkpoint_path)
            self.assertEqual(resolved["sourceRunId"], "20260305-100000Z-sample-run")


if __name__ == "__main__":
    unittest.main()
