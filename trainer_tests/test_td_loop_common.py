from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import td_loop_common


class TdLoopCommonTests(unittest.TestCase):
    def test_checkpoints_from_train_summary_sorts_and_parses_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_value = root / "value-1000.pt"
            second_value = root / "value-2000.pt"
            second_opponent = root / "opponent-2000.pt"
            payload = {
                "results": {
                    "checkpoints": [
                        {"step": 2000, "value": str(second_value), "opponent": str(second_opponent)},
                        {"step": 1000, "value": str(first_value)},
                    ]
                }
            }

            checkpoints = td_loop_common.checkpoints_from_train_summary(payload)

        self.assertEqual([checkpoint.step for checkpoint in checkpoints], [1000, 2000])
        self.assertEqual(checkpoints[0].value_path, first_value)
        self.assertIsNone(checkpoints[0].opponent_path)
        self.assertEqual(checkpoints[1].value_path, second_value)
        self.assertEqual(checkpoints[1].opponent_path, second_opponent)

    def test_checkpoints_from_train_summary_rejects_invalid_step(self) -> None:
        payload = {"results": {"checkpoints": [{"step": True, "value": "value.pt"}]}}

        with self.assertRaises(SystemExit):
            td_loop_common.checkpoints_from_train_summary(payload)

    def test_select_latest_checkpoint_filters_by_candidate_policy(self) -> None:
        checkpoints = [
            td_loop_common.LoopCheckpoint(step=1000, value_path=Path("value-1000.pt"), opponent_path=None),
            td_loop_common.LoopCheckpoint(
                step=2000,
                value_path=Path("value-2000.pt"),
                opponent_path=Path("opponent-2000.pt"),
            ),
            td_loop_common.LoopCheckpoint(step=3000, value_path=Path("value-3000.pt"), opponent_path=None),
        ]

        td_value_latest = td_loop_common.select_latest_checkpoint(
            checkpoints=checkpoints,
            candidate_policy="td-value",
        )
        td_search_latest = td_loop_common.select_latest_checkpoint(
            checkpoints=checkpoints,
            candidate_policy="td-search",
        )

        self.assertEqual(td_value_latest.step, 3000)
        self.assertEqual(td_search_latest.step, 2000)

    def test_select_latest_checkpoint_rejects_missing_eligible_checkpoint(self) -> None:
        checkpoints = [td_loop_common.LoopCheckpoint(step=1000, value_path=None, opponent_path=None)]

        with self.assertRaises(SystemExit):
            td_loop_common.select_latest_checkpoint(
                checkpoints=checkpoints,
                candidate_policy="td-search",
            )

    def test_concat_jsonl_files_merges_inputs_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "a.jsonl"
            shard_b = root / "b.jsonl"
            merged = root / "merged" / "self_play.jsonl"
            shard_a.write_text('{"id":"a"}\n', encoding="utf-8")
            shard_b.write_text('{"id":"b"}\n', encoding="utf-8")

            td_loop_common.concat_jsonl_files(inputs=[shard_a, shard_b], output=merged)

            self.assertTrue(shard_a.exists())
            self.assertTrue(shard_b.exists())
            self.assertEqual(
                merged.read_text(encoding="utf-8"),
                '{"id":"a"}\n{"id":"b"}\n',
            )

    def test_concat_jsonl_files_delete_inputs_after_merge_reuses_first_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = root / "a.jsonl"
            shard_b = root / "b.jsonl"
            merged = root / "merged.jsonl"
            shard_a.write_text('{"id":"a"}\n', encoding="utf-8")
            shard_b.write_text('{"id":"b"}\n', encoding="utf-8")

            td_loop_common.concat_jsonl_files(
                inputs=[shard_a, shard_b],
                output=merged,
                delete_inputs_after_merge=True,
            )

            self.assertFalse(shard_a.exists())
            self.assertFalse(shard_b.exists())
            self.assertEqual(
                merged.read_text(encoding="utf-8"),
                '{"id":"a"}\n{"id":"b"}\n',
            )

    def test_write_progress_and_read_json_round_trip_without_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress_path = root / "progress" / "status.json"
            payload = {"status": "running", "step": "collect"}

            td_loop_common.write_progress(progress_path, payload)
            loaded = td_loop_common.read_json(progress_path, label="progress")

            self.assertEqual(loaded, payload)
            self.assertFalse(progress_path.with_name(f".{progress_path.name}.tmp").exists())

    def test_read_json_rejects_missing_or_invalid_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            invalid = root / "invalid.json"
            invalid.write_text("{", encoding="utf-8")

            with self.assertRaises(SystemExit):
                td_loop_common.read_json(missing, label="missing payload")
            with self.assertRaises(SystemExit):
                td_loop_common.read_json(invalid, label="invalid payload")

    def test_run_step_marks_progress_completed_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress_path = root / "progress.json"

            td_loop_common.run_step(
                name="success-step",
                command=[sys.executable, "-c", "print('ok')"],
                heartbeat_minutes=0.0,
                progress_path=progress_path,
                log_prefix="[td-loop-test]",
            )

            payload = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["step"], "success-step")
        self.assertEqual(payload["returnCode"], 0)

    def test_run_step_marks_progress_failed_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress_path = root / "progress.json"

            with self.assertRaises(SystemExit) as context:
                td_loop_common.run_step(
                    name="failure-step",
                    command=[sys.executable, "-c", "import sys; sys.exit(3)"],
                    heartbeat_minutes=0.0,
                    progress_path=progress_path,
                    log_prefix="[td-loop-test]",
                )

            payload = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertIn("returnCode=3", str(context.exception))
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["step"], "failure-step")
        self.assertEqual(payload["returnCode"], 3)


if __name__ == "__main__":
    unittest.main()
