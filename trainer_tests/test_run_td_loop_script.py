from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_td_loop import (
    ChunkExecutionResult,
    EvalRow,
    PromotionStageResult,
    RunContext,
    build_td_loop_summary,
    initialize_td_loop_run,
    parse_args,
    run_td_loop,
    run_td_loop_chunk,
    run_td_loop_promotion_stage,
)
from scripts.td_loop_common import LoopCheckpoint
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td.checkpoint import save_opponent_checkpoint, save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet


class RunTdLoopScriptTests(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["run_td_loop.py", *args]):
            return parse_args()

    def _make_checkpoints(self, root: Path, *, hidden_dim: int = 32) -> tuple[Path, Path]:
        value_path = root / "warm.value.pt"
        opponent_path = root / "warm.opponent.pt"
        save_value_checkpoint(
            model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=hidden_dim),
            output_path=value_path,
        )
        save_opponent_checkpoint(
            model=OpponentModel(
                observation_dim=OBSERVATION_DIM,
                action_feature_dim=ACTION_FEATURE_DIM,
                hidden_dim=hidden_dim,
            ),
            output_path=opponent_path,
        )
        return value_path, opponent_path

    def _base_args(self, root: Path, *, value_path: Path | None = None, opponent_path: Path | None = None):
        args = [
            "--artifact-dir",
            str(root / "artifacts"),
            "--run-label",
            "unit-bootstrap",
            "--chunks-per-loop",
            "1",
            "--eval-seed-start-indices",
            "0",
            "100",
            "--disable-manifest-promotion",
        ]
        if value_path is not None:
            args.extend(["--train-warm-start-value-checkpoint", str(value_path)])
        if opponent_path is not None:
            args.extend(["--train-warm-start-opponent-checkpoint", str(opponent_path)])
        return self._parse(*args)

    def _eval_row(
        self,
        *,
        artifact_name: str,
        opponent_policy: str,
        win_rate: float,
        ci_low: float,
        ci_high: float,
        side_gap: float,
        candidate_wins: int,
        opponent_wins: int,
        total_games: int,
    ) -> EvalRow:
        return EvalRow(
            artifact=Path(artifact_name),
            opponent_policy=opponent_policy,
            candidate_win_rate=win_rate,
            ci_low=ci_low,
            ci_high=ci_high,
            side_gap=side_gap,
            candidate_wins=candidate_wins,
            opponent_wins=opponent_wins,
            draws=total_games - candidate_wins - opponent_wins,
            total_games=total_games,
            candidate_win_rate_as_player_a=win_rate,
            candidate_win_rate_as_player_b=win_rate,
        )

    def test_initialize_td_loop_run_builds_paths_and_initial_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)

            context = initialize_td_loop_run(args)

            self.assertTrue(context.run_id.endswith("-unit-bootstrap"))
            self.assertTrue(context.run_dir.exists())
            self.assertTrue(context.chunks_dir.exists())
            self.assertTrue(context.eval_dir.exists())
            self.assertEqual(context.warm_value, value_path)
            self.assertEqual(context.warm_opponent, opponent_path)

    def test_run_td_loop_chunk_records_collect_train_and_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm_value, warm_opponent = self._make_checkpoints(root)
            next_value, next_opponent = self._make_checkpoints(root / "next")
            args = self._base_args(root, value_path=warm_value, opponent_path=warm_opponent)
            next_checkpoint = LoopCheckpoint(
                step=1000,
                value_path=next_value,
                opponent_path=next_opponent,
            )

            with patch(
                "scripts.run_td_loop._run_collect_stage",
                return_value={"mode": "single", "commands": [["collect"]]},
            ) as mocked_collect, patch(
                "scripts.run_td_loop._build_train_command",
                return_value=["python", "-m", "scripts.train_td"],
            ) as mocked_train_command, patch(
                "scripts.run_td_loop._run_step",
                return_value=None,
            ), patch(
                "scripts.run_td_loop._read_json",
                return_value={"results": {"checkpoints": []}},
            ), patch(
                "scripts.run_td_loop._checkpoints_from_train_summary",
                return_value=[next_checkpoint],
            ), patch(
                "scripts.run_td_loop._select_latest_checkpoint",
                return_value=next_checkpoint,
            ):
                result = run_td_loop_chunk(
                    args=args,
                    run_id="run-123",
                    chunk_index=1,
                    chunks_dir=root / "artifacts" / "run-123" / "chunks",
                    warm_value=warm_value,
                    warm_opponent=warm_opponent,
                    progress_path=root / "artifacts" / "run-123" / "progress.json",
                )

        self.assertEqual(result.chunk_label, "chunk-001")
        self.assertEqual(result.latest_checkpoint, next_checkpoint)
        self.assertEqual(result.command_row["collect"]["mode"], "single")
        self.assertEqual(result.command_row["train"], ["python", "-m", "scripts.train_td"])
        self.assertEqual(result.chunk_row["latestCheckpoint"]["step"], 1000)
        self.assertEqual(result.chunk_row["latestCheckpoint"]["value"], str(next_value))
        self.assertEqual(result.chunk_row["latestCheckpoint"]["opponent"], str(next_opponent))
        collect_kwargs = mocked_collect.call_args.kwargs
        self.assertEqual(collect_kwargs["run_id"], "run-123-chunk-001")
        train_kwargs = mocked_train_command.call_args.kwargs
        self.assertTrue(
            str(train_kwargs["value_replay"]).endswith("chunk-001\\replay\\self_play.value.jsonl")
        )
        self.assertTrue(
            str(train_kwargs["opponent_replay"]).endswith("chunk-001\\replay\\self_play.opponent.jsonl")
        )

    def test_run_promotion_stage_records_eval_commands_and_pooled_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            latest_checkpoint = LoopCheckpoint(
                step=1000,
                value_path=value_path,
                opponent_path=opponent_path,
            )
            eval_rows = [
                self._eval_row(
                    artifact_name="eval-a.json",
                    opponent_policy="search",
                    win_rate=0.60,
                    ci_low=0.53,
                    ci_high=0.68,
                    side_gap=0.02,
                    candidate_wins=120,
                    opponent_wins=80,
                    total_games=200,
                ),
                self._eval_row(
                    artifact_name="eval-b.json",
                    opponent_policy="search",
                    win_rate=0.58,
                    ci_low=0.51,
                    ci_high=0.65,
                    side_gap=0.03,
                    candidate_wins=116,
                    opponent_wins=84,
                    total_games=200,
                ),
            ]

            with patch(
                "scripts.run_td_loop._build_eval_command",
                side_effect=[
                    ["python", "-m", "scripts.eval_suite", "--seed-start-index", "0"],
                    ["python", "-m", "scripts.eval_suite", "--seed-start-index", "100"],
                ],
            ), patch("scripts.run_td_loop._run_step", return_value=None), patch(
                "scripts.run_td_loop._read_eval_row",
                side_effect=eval_rows,
            ):
                result = run_td_loop_promotion_stage(
                    args=args,
                    eval_dir=root / "artifacts" / "run-123" / "evals",
                    latest_checkpoint=latest_checkpoint,
                    progress_path=root / "artifacts" / "run-123" / "progress.json",
                )

        self.assertEqual(len(result.eval_rows), 2)
        self.assertEqual(result.promotion_eval_artifacts[0], str(root / "artifacts" / "run-123" / "evals" / "promotion_eval.seed-000000.json"))
        self.assertEqual(
            result.promotion_eval_artifacts[1],
            str(root / "artifacts" / "run-123" / "evals" / "promotion_eval.seed-000100.json"),
        )
        self.assertEqual(
            result.promotion_eval_commands,
            [
                ["python", "-m", "scripts.eval_suite", "--seed-start-index", "0"],
                ["python", "-m", "scripts.eval_suite", "--seed-start-index", "100"],
            ],
        )
        self.assertTrue(result.promotion["promoted"])
        self.assertGreater(result.pooled_eval_row.candidate_win_rate, 0.58)

    def test_build_td_loop_summary_contains_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            context = RunContext(
                run_id="run-123",
                run_dir=root / "artifacts" / "run-123",
                chunks_dir=root / "artifacts" / "run-123" / "chunks",
                eval_dir=root / "artifacts" / "run-123" / "evals",
                loop_summary_path=root / "artifacts" / "run-123" / "loop.summary.json",
                progress_path=root / "artifacts" / "run-123" / "progress.json",
                loop_started=0.0,
                warm_value=value_path,
                warm_opponent=opponent_path,
            )
            eval_rows = [
                self._eval_row(
                    artifact_name="eval-a.json",
                    opponent_policy="search",
                    win_rate=0.55,
                    ci_low=0.50,
                    ci_high=0.60,
                    side_gap=0.01,
                    candidate_wins=110,
                    opponent_wins=90,
                    total_games=200,
                ),
                self._eval_row(
                    artifact_name="eval-b.json",
                    opponent_policy="search",
                    win_rate=0.56,
                    ci_low=0.51,
                    ci_high=0.61,
                    side_gap=0.02,
                    candidate_wins=112,
                    opponent_wins=88,
                    total_games=200,
                ),
            ]
            pooled_eval_row = self._eval_row(
                artifact_name="pooled.json",
                opponent_policy="search",
                win_rate=0.555,
                ci_low=0.52,
                ci_high=0.59,
                side_gap=0.015,
                candidate_wins=222,
                opponent_wins=178,
                total_games=400,
            )
            promotion_stage = PromotionStageResult(
                eval_rows=eval_rows,
                promotion_eval_commands=[["cmd-a"], ["cmd-b"]],
                promotion_eval_artifacts=["eval-a.json", "eval-b.json"],
                pooled_eval_row=pooled_eval_row,
                promotion={"promoted": True, "reason": "pooled_eval_passed"},
            )

            payload = build_td_loop_summary(
                args=args,
                context=context,
                commands={"chunks": [{"chunk": "chunk-001"}], "promotionEvals": [["cmd-a"], ["cmd-b"]]},
                chunk_rows=[{"chunk": "chunk-001"}],
                promotion_stage=promotion_stage,
                elapsed_minutes=1.234,
            )

        self.assertEqual(payload["runId"], "run-123")
        self.assertIn("config", payload)
        self.assertIn("artifacts", payload)
        self.assertEqual(payload["commands"]["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(payload["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(payload["evaluation"]["windows"][0]["seedStartIndex"], 0)
        self.assertEqual(payload["evaluation"]["windows"][1]["seedStartIndex"], 100)
        self.assertTrue(payload["promotion"]["promoted"])

    def test_run_td_loop_writes_summary_and_prints_terminal_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            next_value, next_opponent = self._make_checkpoints(root / "next")
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            run_dir = root / "artifacts" / "run-123"
            chunks_dir = run_dir / "chunks"
            eval_dir = run_dir / "evals"
            for path in (run_dir, chunks_dir, eval_dir):
                path.mkdir(parents=True, exist_ok=True)
            context = RunContext(
                run_id="run-123",
                run_dir=run_dir,
                chunks_dir=chunks_dir,
                eval_dir=eval_dir,
                loop_summary_path=run_dir / "loop.summary.json",
                progress_path=run_dir / "progress.json",
                loop_started=time.perf_counter(),
                warm_value=value_path,
                warm_opponent=opponent_path,
            )
            chunk_result = ChunkExecutionResult(
                chunk_label="chunk-001",
                latest_checkpoint=LoopCheckpoint(
                    step=1000,
                    value_path=next_value,
                    opponent_path=next_opponent,
                ),
                command_row={
                    "chunk": "chunk-001",
                    "collect": {"mode": "single", "commands": [["collect"]]},
                    "train": ["python", "-m", "scripts.train_td"],
                },
                chunk_row={
                    "chunk": "chunk-001",
                    "replayRegime": "chunk-local",
                    "collectSummary": str(run_dir / "chunks" / "chunk-001" / "replay" / "self_play.summary.json"),
                    "trainSummary": str(run_dir / "chunks" / "chunk-001" / "train" / "summary.json"),
                    "latestCheckpoint": {
                        "step": 1000,
                        "value": str(next_value),
                        "opponent": str(next_opponent),
                    },
                },
            )
            pooled_eval_row = self._eval_row(
                artifact_name="pooled.json",
                opponent_policy="search",
                win_rate=0.55,
                ci_low=0.51,
                ci_high=0.59,
                side_gap=0.02,
                candidate_wins=220,
                opponent_wins=180,
                total_games=400,
            )
            promotion_stage = PromotionStageResult(
                eval_rows=[
                    self._eval_row(
                        artifact_name="eval-a.json",
                        opponent_policy="search",
                        win_rate=0.54,
                        ci_low=0.50,
                        ci_high=0.58,
                        side_gap=0.02,
                        candidate_wins=108,
                        opponent_wins=92,
                        total_games=200,
                    ),
                    self._eval_row(
                        artifact_name="eval-b.json",
                        opponent_policy="search",
                        win_rate=0.56,
                        ci_low=0.52,
                        ci_high=0.60,
                        side_gap=0.02,
                        candidate_wins=112,
                        opponent_wins=88,
                        total_games=200,
                    ),
                ],
                promotion_eval_commands=[["cmd-a"], ["cmd-b"]],
                promotion_eval_artifacts=["eval-a.json", "eval-b.json"],
                pooled_eval_row=pooled_eval_row,
                promotion={"promoted": True, "reason": "pooled_eval_passed"},
            )

            with patch("scripts.run_td_loop.initialize_td_loop_run", return_value=context), patch(
                "scripts.run_td_loop.run_td_loop_chunk",
                return_value=chunk_result,
            ) as mocked_chunk, patch(
                "scripts.run_td_loop.run_td_loop_promotion_stage",
                return_value=promotion_stage,
            ), patch("builtins.print") as mocked_print:
                result = run_td_loop(args)

            payload = json.loads(context.loop_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(mocked_chunk.call_count, 1)
        self.assertEqual(payload["runId"], "run-123")
        self.assertEqual(payload["commands"]["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(payload["promotion"]["reason"], "pooled_eval_passed")
        printed = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(printed["runId"], "run-123")
        self.assertTrue(printed["promoted"])

    def test_run_td_loop_raises_without_writing_summary_when_chunk_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            run_dir = root / "artifacts" / "run-123"
            chunks_dir = run_dir / "chunks"
            eval_dir = run_dir / "evals"
            for path in (run_dir, chunks_dir, eval_dir):
                path.mkdir(parents=True, exist_ok=True)
            context = RunContext(
                run_id="run-123",
                run_dir=run_dir,
                chunks_dir=chunks_dir,
                eval_dir=eval_dir,
                loop_summary_path=run_dir / "loop.summary.json",
                progress_path=run_dir / "progress.json",
                loop_started=time.perf_counter(),
                warm_value=value_path,
                warm_opponent=opponent_path,
            )

            with patch("scripts.run_td_loop.initialize_td_loop_run", return_value=context), patch(
                "scripts.run_td_loop.run_td_loop_chunk",
                side_effect=SystemExit("chunk failed"),
            ), patch("builtins.print") as mocked_print:
                with self.assertRaises(SystemExit):
                    run_td_loop(args)

        self.assertFalse(context.loop_summary_path.exists())
        mocked_print.assert_not_called()


class RunTdLoopSmokeTests(unittest.TestCase):
    def test_script_smoke_runs_small_td_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            command = [
                sys.executable,
                "-m",
                "scripts.run_td_loop",
                "--artifact-dir",
                str(artifact_dir),
                "--run-label",
                "bootstrap-smoke",
                "--chunks-per-loop",
                "1",
                "--collect-games",
                "2",
                "--collect-workers",
                "1",
                "--collect-progress-every-games",
                "0",
                "--collect-search-worlds",
                "1",
                "--collect-search-rollouts",
                "1",
                "--collect-search-depth",
                "1",
                "--collect-search-max-root-actions",
                "1",
                "--collect-td-worlds",
                "1",
                "--train-steps",
                "1",
                "--train-hidden-dim",
                "32",
                "--train-value-batch-size",
                "4",
                "--train-opponent-batch-size",
                "4",
                "--train-save-every-steps",
                "1",
                "--train-progress-every-steps",
                "0",
                "--train-target-sync-interval",
                "1",
                "--train-num-threads",
                "1",
                "--train-num-interop-threads",
                "1",
                "--eval-games-per-side",
                "1",
                "--eval-workers",
                "1",
                "--eval-seed-start-indices",
                "0",
                "--eval-progress-every-games",
                "0",
                "--eval-progress-log-minutes",
                "0",
                "--eval-search-worlds",
                "1",
                "--eval-search-rollouts",
                "1",
                "--eval-search-depth",
                "1",
                "--eval-search-max-root-actions",
                "1",
                "--eval-td-worlds",
                "1",
                "--progress-heartbeat-minutes",
                "0",
                "--promotion-min-ci-low",
                "0.0",
                "--disable-manifest-promotion",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                combined_output = f"{completed.stdout}\n{completed.stderr}"
                if "spawn EPERM" in combined_output:
                    self.skipTest(
                        "Node bridge child-process spawn is blocked in the sandboxed test environment."
                    )
                raise AssertionError(
                    "bootstrap loop smoke run failed.\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            run_dirs = list(artifact_dir.iterdir())
            self.assertEqual(len(run_dirs), 1)
            loop_summary_path = run_dirs[0] / "loop.summary.json"
            self.assertTrue(loop_summary_path.exists())
            payload = json.loads(loop_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runId"], run_dirs[0].name)
            self.assertEqual(len(payload["chunks"]), 1)


if __name__ == "__main__":
    unittest.main()
