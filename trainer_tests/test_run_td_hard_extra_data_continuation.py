from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_td_hard_extra_data_continuation as runner


class RunTDHardExtraDataContinuationTests(unittest.TestCase):
    def test_completed_value_summary_validates_all_frozen_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_root = root / "checkpoints"
            output_root = checkpoint_root / "primary" / "continued-control" / "value"
            run_root = output_root / "run"
            run_root.mkdir(parents=True)
            checkpoints = []
            for step in runner.EXPECTED_CHECKPOINT_STEPS:
                checkpoint = run_root / f"value-step-{step:07d}.pt"
                checkpoint.write_bytes(b"checkpoint")
                checkpoints.append({"step": step, "value": str(checkpoint)})

            summary_path = checkpoint_root / "primary" / "continued-control" / "value.summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "config": {
                    "steps": 10_000,
                    "seed": 2026072801,
                    "districtAugmentation": "none",
                    "replayKeyMode": "run-qualified-canonical-v1",
                    "numThreads": 4,
                    "numInteropThreads": 1,
                    "saveEverySteps": 1_000,
                    "progressEverySteps": 50,
                    "trainValue": True,
                    "trainOpponent": False,
                    "valueReplayFiles": [
                        f"run/shards/shard-{index:03d}.value.jsonl" for index in range(900)
                    ],
                },
                "provenance": {
                    "replayKeyMode": "run-qualified-canonical-v1",
                    "experimentManifestSha256": "1" * 64,
                    "implementationSha256": "2" * 64,
                    "valueReplayContentSha256": "3" * 64,
                    "warmStartValueSha256": "4" * 64,
                },
                "results": {
                    "valueReplaySize": 163_194,
                    "opponentReplaySize": 0,
                    "valueSamplingTraceSha256": "5" * 64,
                    "latestValue": {"step": 10_000},
                    "checkpoints": checkpoints,
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            row = {
                "id": "primary-continued-control-value",
                "seedId": "primary",
                "armId": "continued-control",
                "model": "value",
                "command": [
                    "python",
                    "-m",
                    "scripts.train_td",
                    "--summary-out",
                    str(summary_path),
                    "--out-dir",
                    str(output_root),
                    "--expected-value-replay-content-sha256",
                    "3" * 64,
                    "--expected-warm-start-value-sha256",
                    "4" * 64,
                ],
            }
            plan = {
                "sourceManifestSha256": "1" * 64,
                "verification": {
                    "implementationSha256": "2" * 64,
                    "originalControl": {
                        "games": 900,
                        "rows": 163_194,
                    },
                },
            }

            result = runner._validate_completed_summary(
                row=row,
                plan=plan,
                checkpoint_root=checkpoint_root,
            )

            self.assertEqual(result["id"], row["id"])
            self.assertEqual(result["completedSteps"], 10_000)
            self.assertEqual(result["finalCheckpoint"], str(Path(checkpoints[-1]["value"])))

    def test_completed_summary_rejects_holdout_replay_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_root = root / "checkpoints"
            output_root = checkpoint_root / "primary" / "continued-control" / "opponent"
            run_root = output_root / "run"
            run_root.mkdir(parents=True)
            checkpoints = []
            for step in runner.EXPECTED_CHECKPOINT_STEPS:
                checkpoint = run_root / f"opponent-step-{step:07d}.pt"
                checkpoint.write_bytes(b"checkpoint")
                checkpoints.append({"step": step, "opponent": str(checkpoint)})
            summary_path = (
                checkpoint_root / "primary" / "continued-control" / "opponent.summary.json"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "config": {
                            "steps": 10_000,
                            "seed": 2026072801,
                            "districtAugmentation": "none",
                            "replayKeyMode": "run-qualified-canonical-v1",
                            "numThreads": 4,
                            "numInteropThreads": 1,
                            "saveEverySteps": 1_000,
                            "progressEverySteps": 50,
                            "trainValue": False,
                            "trainOpponent": True,
                            "opponentReplayFiles": [
                                *[
                                    f"run/shards/shard-{index:03d}.opponent.jsonl"
                                    for index in range(899)
                                ],
                                "development.opponent.jsonl",
                            ],
                        },
                        "provenance": {
                            "replayKeyMode": "run-qualified-canonical-v1",
                            "experimentManifestSha256": "1" * 64,
                            "implementationSha256": "2" * 64,
                            "opponentReplayContentSha256": "3" * 64,
                            "warmStartOpponentSha256": "4" * 64,
                        },
                        "results": {
                            "valueReplaySize": 0,
                            "opponentReplaySize": 163_194,
                            "opponentSamplingTraceSha256": "5" * 64,
                            "latestOpponent": {"step": 10_000},
                            "checkpoints": checkpoints,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "id": "primary-continued-control-opponent",
                "seedId": "primary",
                "armId": "continued-control",
                "model": "opponent",
                "command": [
                    "python",
                    "-m",
                    "scripts.train_td",
                    "--summary-out",
                    str(summary_path),
                    "--out-dir",
                    str(output_root),
                    "--expected-opponent-replay-content-sha256",
                    "3" * 64,
                    "--expected-warm-start-opponent-sha256",
                    "4" * 64,
                ],
            }
            plan = {
                "sourceManifestSha256": "1" * 64,
                "verification": {
                    "implementationSha256": "2" * 64,
                    "originalControl": {
                        "games": 900,
                        "rows": 163_194,
                    },
                },
            }

            with self.assertRaisesRegex(SystemExit, "development or test"):
                runner._validate_completed_summary(
                    row=row,
                    plan=plan,
                    checkpoint_root=checkpoint_root,
                )

    def test_flag_value_rejects_duplicate_flags(self) -> None:
        with self.assertRaisesRegex(SystemExit, "exactly once"):
            runner._flag_value(
                ["python", "--steps", "1", "--steps", "2"],
                "--steps",
            )

    def test_completion_marker_rejects_pending_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "training.complete.json"
            marker.write_text(
                json.dumps(
                    {
                        "experimentId": runner.EXPERIMENT_ID,
                        "sourceManifestSha256": "1" * 64,
                        "commands": 8,
                        "completedCommandIds": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "summaries are missing"):
                runner._validate_existing_completion_marker(
                    completion_path=marker,
                    plan={"sourceManifestSha256": "1" * 64},
                    completed=[],
                    pending=[{"id": "primary-continued-control-value"}],
                )


if __name__ == "__main__":
    unittest.main()
