from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_td
from trainer.td.ablation_manifest import (
    replay_content_sha256,
    resolve_frozen_replay_split,
    write_replay_path_lists,
)


class TDAblationManifestTests(unittest.TestCase):
    def test_replay_split_is_deterministic_paired_and_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards = root / "shards"
            shards.mkdir()
            for index in range(5):
                key = f"shard-{index:03d}"
                (shards / f"{key}.value.jsonl").write_text(
                    json.dumps({"index": index}) + "\n", encoding="utf-8"
                )
                (shards / f"{key}.opponent.jsonl").write_text(
                    json.dumps({"index": index}) + "\n", encoding="utf-8"
                )

            first = resolve_frozen_replay_split(
                shards_dir=shards,
                salt="fixed-test-salt",
                validation_shards=2,
            )
            second = resolve_frozen_replay_split(
                shards_dir=shards,
                salt="fixed-test-salt",
                validation_shards=2,
            )
            self.assertEqual(first.membership_sha256, second.membership_sha256)
            self.assertEqual(first.inventory_sha256, second.inventory_sha256)
            self.assertEqual(
                first.value_content_sha256,
                second.value_content_sha256,
            )
            self.assertEqual(len(first.training_keys), 3)
            self.assertEqual(len(first.validation_keys), 2)
            self.assertEqual(
                set(first.training_keys) | set(first.validation_keys),
                {f"shard-{index:03d}" for index in range(5)},
            )
            self.assertFalse(set(first.training_keys) & set(first.validation_keys))

            outputs = write_replay_path_lists(split=first, output_dir=root / "lists")
            training_value = outputs["trainingValue"].read_text(encoding="utf-8").splitlines()
            training_opponent = outputs["trainingOpponent"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(training_value), 3)
            self.assertEqual(len(training_opponent), 3)
            self.assertEqual(
                [Path(path).name.removesuffix(".value.jsonl") for path in training_value],
                list(first.training_keys),
            )

            ordered_value_paths = [first.value_paths[key] for key in first.training_keys]
            self.assertEqual(
                replay_content_sha256(ordered_value_paths),
                first.training_value_content_sha256,
            )
            before = first.value_content_sha256
            changed_path = first.value_paths[sorted(first.value_paths)[0]]
            changed_path.write_text('{"changed": true}\n', encoding="utf-8")
            after = resolve_frozen_replay_split(
                shards_dir=shards,
                salt="fixed-test-salt",
                validation_shards=2,
            )
            self.assertNotEqual(before, after.value_content_sha256)
            self.assertEqual(
                [Path(path).name.removesuffix(".opponent.jsonl") for path in training_opponent],
                list(first.training_keys),
            )

    def test_replay_split_rejects_unpaired_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            shards = Path(temp_dir)
            (shards / "shard-000.value.jsonl").write_text("{}\n", encoding="utf-8")
            (shards / "shard-001.opponent.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not paired"):
                resolve_frozen_replay_split(
                    shards_dir=shards,
                    salt="salt",
                    validation_shards=1,
                )

    def test_train_td_replay_list_resolves_relative_paths_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            replay = root / "one.value.jsonl"
            replay.write_text("{}\n", encoding="utf-8")
            replay_list = root / "paths.txt"
            replay_list.write_text("# frozen split\none.value.jsonl\n\n", encoding="utf-8")
            paths = train_td._merge_replay_paths(
                inline_paths=None,
                list_path=replay_list,
                label="value",
            )
            self.assertEqual(paths, [replay.resolve()])

    def test_train_td_s4_mode_requires_explicit_augmentation_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = Path(temp_dir) / "one.opponent.jsonl"
            replay.write_text("{}\n", encoding="utf-8")
            argv = [
                "train_td",
                "--disable-value",
                "--opponent-replay",
                str(replay),
                "--district-augmentation",
                "s4",
            ]
            with patch("sys.argv", argv):
                args = train_td.parse_args()
            with self.assertRaisesRegex(SystemExit, "augmentation-seed"):
                train_td._validate_args(args)

    def test_train_td_rejects_mismatched_frozen_replay_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            replay = Path(temp_dir) / "one.opponent.jsonl"
            replay.write_text("{}\n", encoding="utf-8")
            argv = [
                "train_td",
                "--disable-value",
                "--opponent-replay",
                str(replay),
                "--expected-opponent-replay-content-sha256",
                "0" * 64,
            ]
            with patch("sys.argv", argv):
                args = train_td.parse_args()
            train_td._validate_args(args)
            with self.assertRaisesRegex(SystemExit, "content SHA-256 mismatch"):
                train_td._resolve_training_provenance(args)


if __name__ == "__main__":
    unittest.main()
