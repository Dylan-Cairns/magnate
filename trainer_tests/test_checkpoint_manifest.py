from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from scripts.checkpoint_manifest import (
    load_default_warm_start,
    load_manifest_opponent_pool,
    update_manifest_promoted_checkpoint,
)
from scripts.promote_td_checkpoint import promote_checkpoint_pair


class CheckpointManifestTests(unittest.TestCase):
    def test_load_default_warm_start_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, value_path, opponent_path = self._write_manifest_checkpoint(root)

            checkpoint = load_default_warm_start(
                manifest_path=manifest_path,
                require_paths=True,
            )

            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(checkpoint.key, "promoted")
            self.assertEqual(checkpoint.status, "promoted")
            self.assertEqual(checkpoint.value_path, value_path)
            self.assertEqual(checkpoint.opponent_path, opponent_path)

    def test_opponent_pool_falls_back_to_promoted_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _, _ = self._write_manifest_checkpoint(
                root,
                include_opponent_pool=False,
            )

            pool = load_manifest_opponent_pool(
                manifest_path=manifest_path,
                require_paths=True,
            )

            self.assertEqual([row.key for row in pool], ["promoted"])

    def test_update_manifest_promoted_checkpoint_prepends_pool_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _, _ = self._write_manifest_checkpoint(root)
            value_path = root / "artifacts" / "td_loops" / "run-2" / "value.pt"
            opponent_path = root / "artifacts" / "td_loops" / "run-2" / "opponent.pt"
            value_path.parent.mkdir(parents=True, exist_ok=True)
            value_path.write_text("value", encoding="utf-8")
            opponent_path.write_text("opponent", encoding="utf-8")

            payload = update_manifest_promoted_checkpoint(
                manifest_path=manifest_path,
                key="Run 2 Promoted",
                value_path=value_path,
                opponent_path=opponent_path,
                source_run_id="run-2",
                source_loop_summary=root / "artifacts" / "td_loops" / "run-2" / "loop.summary.json",
                source_chunk="chunk-009",
                source_eval_artifacts=[root / "artifacts" / "td_loops" / "run-2" / "eval.json"],
                step=10000,
                generated_at_utc="2026-04-22T00:00:00+00:00",
                set_default=True,
                add_to_opponent_pool=True,
            )

            self.assertEqual(payload["defaultWarmStart"], "run-2-promoted")
            self.assertEqual(payload["opponentPool"][:2], ["run-2-promoted", "promoted"])
            entry = payload["checkpoints"]["run-2-promoted"]
            self.assertEqual(entry["value"], "artifacts/td_loops/run-2/value.pt")
            self.assertEqual(entry["opponent"], "artifacts/td_loops/run-2/opponent.pt")
            self.assertEqual(entry["sourceChunk"], "chunk-009")

    def test_promote_checkpoint_pair_copies_and_registers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "models" / "td_checkpoints" / "manifest.json"
            source_dir = root / "artifacts" / "td_loops" / "run-3" / "chunks" / "chunk-009"
            source_dir.mkdir(parents=True, exist_ok=True)
            value_path = source_dir / "value-step-0010000.pt"
            opponent_path = source_dir / "opponent-step-0010000.pt"
            value_path.write_text("value", encoding="utf-8")
            opponent_path.write_text("opponent", encoding="utf-8")

            result = promote_checkpoint_pair(
                manifest_path=manifest_path,
                checkpoint_root=root / "models" / "td_checkpoints",
                key="run-3-promoted",
                value_checkpoint=value_path,
                opponent_checkpoint=opponent_path,
                source_run_id="run-3",
                source_chunk="chunk-009",
                step=10000,
                set_default=True,
                add_to_opponent_pool=True,
            )

            copied_value = cast(Path, result["value"])
            copied_opponent = cast(Path, result["opponent"])
            self.assertTrue(copied_value.exists())
            self.assertTrue(copied_opponent.exists())
            self.assertEqual(copied_value.parent.name, "run-3-promoted")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["defaultWarmStart"], "run-3-promoted")
            self.assertEqual(payload["opponentPool"], ["run-3-promoted"])

    def _write_manifest_checkpoint(
        self,
        root: Path,
        *,
        include_opponent_pool: bool = True,
    ) -> tuple[Path, Path, Path]:
        checkpoint_dir = root / "models" / "td_checkpoints" / "promoted"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        value_path = checkpoint_dir / "value.pt"
        opponent_path = checkpoint_dir / "opponent.pt"
        value_path.write_text("value", encoding="utf-8")
        opponent_path.write_text("opponent", encoding="utf-8")
        manifest_path = checkpoint_dir.parent / "manifest.json"
        payload = {
            "schemaVersion": 2,
            "defaultWarmStart": "promoted",
            "checkpoints": {
                "promoted": {
                    "status": "promoted",
                    "sourceRunId": "run-1",
                    "value": "models/td_checkpoints/promoted/value.pt",
                    "opponent": "models/td_checkpoints/promoted/opponent.pt",
                }
            },
        }
        if include_opponent_pool:
            payload["opponentPool"] = ["promoted"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path, value_path, opponent_path


if __name__ == "__main__":
    unittest.main()
