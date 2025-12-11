from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_browser_td_search_pack import (
    export_td_search_checkpoint_pack,
    resolve_checkpoints,
)
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td.checkpoint import save_opponent_checkpoint, save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet


class ExportBrowserTdSearchPackTests(unittest.TestCase):
    def test_exports_pack_manifest_weights_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            value_checkpoint = root / "value.pt"
            opponent_checkpoint = root / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_checkpoint,
                metadata={"step": 22},
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=16,
                ),
                output_path=opponent_checkpoint,
                metadata={"step": 22},
            )

            output_root = root / "public" / "model-packs"
            result = export_td_search_checkpoint_pack(
                value_checkpoint_path=value_checkpoint,
                opponent_checkpoint_path=opponent_checkpoint,
                output_root=output_root,
                pack_id="td-search-test-pack",
                label="TD Search Test",
                set_default=False,
            )

            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["packId"], "td-search-test-pack")
            self.assertEqual(manifest["model"]["modelType"], "td-search-v1")
            self.assertIn("value", manifest["model"])
            self.assertIn("opponent", manifest["model"])

            weights = json.loads(Path(result["weights"]).read_text(encoding="utf-8"))
            self.assertEqual(weights["schemaVersion"], 1)
            self.assertIn("encoder.0.weight", weights["valueTensors"])
            self.assertIn("obs_encoder.0.weight", weights["opponentTensors"])

            index = json.loads(Path(result["index"]).read_text(encoding="utf-8"))
            self.assertTrue(any(pack["id"] == "td-search-test-pack" for pack in index["packs"]))

    def test_resolve_latest_promoted_pair_from_loop_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            artifact_root = root / "artifacts" / "td_loops"
            run_dir = artifact_root / "20260305-100000Z-sample-run"
            run_dir.mkdir(parents=True)

            value_checkpoint = run_dir / "value-step-0000123.pt"
            opponent_checkpoint = run_dir / "opponent-step-0000123.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=8),
                output_path=value_checkpoint,
                metadata={"step": 123},
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=8,
                ),
                output_path=opponent_checkpoint,
                metadata={"step": 123},
            )

            summary_payload = {
                "promotion": {"promoted": True},
                "chunks": [
                    {
                        "latestCheckpoint": {
                            "step": 123,
                            "value": str(value_checkpoint),
                            "opponent": str(opponent_checkpoint),
                        }
                    }
                ],
            }
            (run_dir / "loop.summary.json").write_text(
                json.dumps(summary_payload),
                encoding="utf-8",
            )

            resolved = resolve_checkpoints(
                value_checkpoint=None,
                opponent_checkpoint=None,
                latest_promoted=True,
                artifact_root=artifact_root,
            )
            self.assertEqual(Path(resolved["valueCheckpointPath"]), value_checkpoint)
            self.assertEqual(Path(resolved["opponentCheckpointPath"]), opponent_checkpoint)
            self.assertEqual(resolved["sourceRunId"], "20260305-100000Z-sample-run")


if __name__ == "__main__":
    unittest.main()
