from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_td_hard_extra_data_continuation as preparation
from trainer.td import REPLAY_KEY_MODE_RUN_QUALIFIED


class PrepareTDHardExtraDataContinuationTests(unittest.TestCase):
    @staticmethod
    def _manifest() -> dict[str, object]:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "td-training"
            / "td-hard-extra-data-continuation-v1.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    def test_source_manifest_is_explicitly_non_launching(self) -> None:
        manifest = self._manifest()
        preparation._validate_manifest(manifest)
        self.assertEqual(manifest["status"], "review-required")
        self.assertIs(manifest["launchAuthorized"], False)

        unsafe = copy.deepcopy(manifest)
        unsafe["launchAuthorized"] = True
        with self.assertRaisesRegex(SystemExit, "launchAuthorized=false"):
            preparation._validate_manifest(unsafe)

    def test_prepared_commands_are_matched_and_exclude_holdout_paths(self) -> None:
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            value_warm = root / "value.pt"
            opponent_warm = root / "opponent.pt"
            value_warm.write_bytes(b"value")
            opponent_warm.write_bytes(b"opponent")
            path_lists = {}
            for name in (
                "controlValue",
                "controlOpponent",
                "treatmentValue",
                "treatmentOpponent",
                "developmentValue",
                "developmentOpponent",
            ):
                path = root / f"{name}.txt"
                path.write_text("placeholder\n", encoding="utf-8")
                path_lists[name] = path
            commands = preparation._training_commands(
                manifest=manifest,
                python_bin=root / "python.exe",
                manifest_path=root / "manifest.json",
                source_manifest_sha256="1" * 64,
                implementation_sha256="2" * 64,
                repo_root=root,
                checkpoint_dir=root / "checkpoints",
                path_lists=path_lists,
                replay_key_mode=REPLAY_KEY_MODE_RUN_QUALIFIED,
                value_warm=value_warm,
                opponent_warm=opponent_warm,
                original_stats={"contentSha256": {"value": "3" * 64, "opponent": "4" * 64}},
                treatment_stats={"contentSha256": {"value": "5" * 64, "opponent": "6" * 64}},
            )

        self.assertEqual(len(commands), 8)
        self.assertEqual(len({str(row["id"]) for row in commands}), 8)
        by_seed_arm: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in commands:
            command = [str(value) for value in row["command"]]
            self.assertIn("--replay-key-mode", command)
            self.assertEqual(
                command[command.index("--replay-key-mode") + 1],
                REPLAY_KEY_MODE_RUN_QUALIFIED,
            )
            self.assertNotIn("developmentValue", " ".join(command))
            self.assertNotIn("developmentOpponent", " ".join(command))
            self.assertNotIn("test", " ".join(command).lower())
            key = (str(row["seedId"]), str(row["armId"]))
            by_seed_arm.setdefault(key, []).append(row)
        self.assertEqual(
            set(by_seed_arm),
            {
                ("primary", "continued-control"),
                ("primary", "extra-data-treatment"),
                ("replication", "continued-control"),
                ("replication", "extra-data-treatment"),
            },
        )
        self.assertTrue(all(len(rows) == 2 for rows in by_seed_arm.values()))
        smoke_commands = preparation._guardrail_smoke_commands(
            commands=commands,
            checkpoint_dir=Path("checkpoints"),
        )
        self.assertEqual(len(smoke_commands), 4)
        for row in smoke_commands:
            command = [str(value) for value in row["command"]]
            self.assertEqual(command[command.index("--steps") + 1], "1")
            self.assertEqual(command[command.index("--save-every-steps") + 1], "0")

    def test_path_list_resolves_relative_entries_from_list_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            replay = root / "run" / "shards" / "shard-000.value.jsonl"
            replay.parent.mkdir(parents=True)
            replay.write_text("{}\n", encoding="utf-8")
            path_list = root / "lists" / "value.paths.txt"
            path_list.parent.mkdir()
            path_list.write_text(
                "../run/shards/shard-000.value.jsonl\n",
                encoding="utf-8",
            )
            self.assertEqual(preparation._read_path_list(path_list), [replay.resolve()])


if __name__ == "__main__":
    unittest.main()
