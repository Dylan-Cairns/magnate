from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.opponent_pool import (
    filter_pool_excluding_checkpoint,
    load_promoted_checkpoints,
    split_evenly,
    weighted_game_split,
)


class OpponentPoolTests(unittest.TestCase):
    def test_weighted_game_split_assigns_all_games(self) -> None:
        split = weighted_game_split(
            total_games=17,
            weights={"selfplay": 0.6, "pool": 0.25, "search": 0.15},
        )
        self.assertEqual(sum(split.values()), 17)
        self.assertGreater(split["selfplay"], split["pool"])
        self.assertGreater(split["pool"], split["search"])

    def test_split_evenly_distributes_remainder(self) -> None:
        split = split_evenly(10, ["a", "b", "c"])
        self.assertEqual(sum(split.values()), 10)
        self.assertEqual(split["a"], 4)
        self.assertEqual(split["b"], 3)
        self.assertEqual(split["c"], 3)

    def test_load_promoted_checkpoints_sorted_and_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "td_loops"
            root.mkdir(parents=True, exist_ok=True)

            # older promoted run
            older = root / "20260101-old"
            older.mkdir(parents=True, exist_ok=True)
            older_value = older / "value.pt"
            older_opp = older / "opp.pt"
            older_value.write_text("v", encoding="utf-8")
            older_opp.write_text("o", encoding="utf-8")
            older_summary = {
                "generatedAtUtc": "2026-01-01T00:00:00+00:00",
                "runId": "20260101-old",
                "promotion": {"promoted": True},
                "chunks": [
                    {
                        "latestCheckpoint": {
                            "value": str(older_value),
                            "opponent": str(older_opp),
                        }
                    }
                ],
            }
            (older / "loop.summary.json").write_text(
                json.dumps(older_summary), encoding="utf-8"
            )

            # newer promoted run
            newer = root / "20260102-new"
            newer.mkdir(parents=True, exist_ok=True)
            newer_value = newer / "value.pt"
            newer_opp = newer / "opp.pt"
            newer_value.write_text("v", encoding="utf-8")
            newer_opp.write_text("o", encoding="utf-8")
            newer_summary = {
                "generatedAtUtc": "2026-01-02T00:00:00+00:00",
                "runId": "20260102-new",
                "promotion": {"promoted": True},
                "chunks": [
                    {
                        "latestCheckpoint": {
                            "value": str(newer_value),
                            "opponent": str(newer_opp),
                        }
                    }
                ],
            }
            (newer / "loop.summary.json").write_text(
                json.dumps(newer_summary), encoding="utf-8"
            )

            # non-promoted run should be skipped.
            skipped = root / "20260103-skip"
            skipped.mkdir(parents=True, exist_ok=True)
            (skipped / "loop.summary.json").write_text(
                json.dumps({"promotion": {"promoted": False}, "chunks": []}),
                encoding="utf-8",
            )

            pool = load_promoted_checkpoints(artifact_dir=root, max_entries=10, require_paths=True)
            self.assertEqual([entry.run_id for entry in pool], ["20260102-new", "20260101-old"])

            filtered = filter_pool_excluding_checkpoint(
                checkpoints=pool,
                value_path=newer_value,
                opponent_path=newer_opp,
            )
            self.assertEqual([entry.run_id for entry in filtered], ["20260101-old"])

    def test_load_promoted_checkpoints_handles_mixed_datetime_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "td_loops"
            root.mkdir(parents=True, exist_ok=True)

            aware = root / "aware"
            aware.mkdir(parents=True, exist_ok=True)
            (aware / "v.pt").write_text("v", encoding="utf-8")
            (aware / "o.pt").write_text("o", encoding="utf-8")
            (aware / "loop.summary.json").write_text(
                json.dumps(
                    {
                        "generatedAtUtc": "2026-01-03T00:00:00+00:00",
                        "runId": "aware",
                        "promotion": {"promoted": True},
                        "chunks": [
                            {
                                "latestCheckpoint": {
                                    "value": str(aware / "v.pt"),
                                    "opponent": str(aware / "o.pt"),
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            naive = root / "naive"
            naive.mkdir(parents=True, exist_ok=True)
            (naive / "v.pt").write_text("v", encoding="utf-8")
            (naive / "o.pt").write_text("o", encoding="utf-8")
            (naive / "loop.summary.json").write_text(
                json.dumps(
                    {
                        "generatedAtUtc": "2026-01-02T00:00:00",
                        "runId": "naive",
                        "promotion": {"promoted": True},
                        "chunks": [
                            {
                                "latestCheckpoint": {
                                    "value": str(naive / "v.pt"),
                                    "opponent": str(naive / "o.pt"),
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            missing = root / "missing"
            missing.mkdir(parents=True, exist_ok=True)
            (missing / "v.pt").write_text("v", encoding="utf-8")
            (missing / "o.pt").write_text("o", encoding="utf-8")
            (missing / "loop.summary.json").write_text(
                json.dumps(
                    {
                        "runId": "missing",
                        "promotion": {"promoted": True},
                        "chunks": [
                            {
                                "latestCheckpoint": {
                                    "value": str(missing / "v.pt"),
                                    "opponent": str(missing / "o.pt"),
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = load_promoted_checkpoints(artifact_dir=root, max_entries=10, require_paths=True)
            self.assertEqual([row.run_id for row in rows], ["aware", "naive", "missing"])

    def test_load_promoted_checkpoints_prefers_manifest_pool_and_dedupes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_root = root / "artifacts" / "td_loops"
            artifact_root.mkdir(parents=True, exist_ok=True)

            manifest_dir = root / "models" / "td_checkpoints" / "promoted"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_value = manifest_dir / "value.pt"
            manifest_opponent = manifest_dir / "opponent.pt"
            manifest_value.write_text("value", encoding="utf-8")
            manifest_opponent.write_text("opponent", encoding="utf-8")
            manifest = {
                "schemaVersion": 2,
                "defaultWarmStart": "promoted",
                "opponentPool": ["promoted"],
                "checkpoints": {
                    "promoted": {
                        "status": "promoted",
                        "sourceRunId": "manifest-run",
                        "generatedAtUtc": "2026-01-03T00:00:00+00:00",
                        "value": "models/td_checkpoints/promoted/value.pt",
                        "opponent": "models/td_checkpoints/promoted/opponent.pt",
                    }
                },
            }
            (root / "models" / "td_checkpoints" / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            duplicate = artifact_root / "duplicate-run"
            duplicate.mkdir()
            (duplicate / "loop.summary.json").write_text(
                json.dumps(
                    {
                        "generatedAtUtc": "2026-01-04T00:00:00+00:00",
                        "runId": "duplicate-run",
                        "promotion": {"promoted": True},
                        "chunks": [
                            {
                                "latestCheckpoint": {
                                    "value": str(manifest_value),
                                    "opponent": str(manifest_opponent),
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifact_only = artifact_root / "artifact-run"
            artifact_only.mkdir()
            artifact_value = artifact_only / "value.pt"
            artifact_opponent = artifact_only / "opponent.pt"
            artifact_value.write_text("value", encoding="utf-8")
            artifact_opponent.write_text("opponent", encoding="utf-8")
            (artifact_only / "loop.summary.json").write_text(
                json.dumps(
                    {
                        "generatedAtUtc": "2026-01-02T00:00:00+00:00",
                        "runId": "artifact-run",
                        "promotion": {"promoted": True},
                        "chunks": [
                            {
                                "latestCheckpoint": {
                                    "value": str(artifact_value),
                                    "opponent": str(artifact_opponent),
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = load_promoted_checkpoints(
                artifact_dir=artifact_root,
                max_entries=10,
                require_paths=True,
            )

            self.assertEqual([row.run_id for row in rows], ["manifest-run", "artifact-run"])


if __name__ == "__main__":
    unittest.main()
