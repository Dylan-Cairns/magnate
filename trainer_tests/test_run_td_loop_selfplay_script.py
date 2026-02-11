from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.opponent_pool import PoolCheckpoint
from scripts.run_td_loop_selfplay import (
    ChunkExecutionResult,
    ChunkGateResult,
    EvalRow,
    PromotionStageResult,
    ReplayChunk,
    RunContext,
    _validate_args,
    build_replay_window,
    build_selfplay_loop_summary,
    initialize_selfplay_run,
    parse_args,
    run_checkpoint_selection,
    run_chunk_gate,
    run_promotion_stage,
    run_selfplay_chunk,
    run_selfplay_loop,
)
from scripts.td_loop_common import LoopCheckpoint
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.td.checkpoint import save_opponent_checkpoint, save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet


class RunTdLoopSelfplayScriptTests(unittest.TestCase):
    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["run_td_loop_selfplay.py", *args]):
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

    def _base_args(self, root: Path, *, value_path: Path, opponent_path: Path):
        return self._parse(
            "--artifact-dir",
            str(root / "artifacts"),
            "--run-label",
            "unit-selfplay",
            "--chunks-per-loop",
            "1",
            "--eval-seed-start-indices",
            "0",
            "--incumbent-eval-seed-start-indices",
            "100",
            "--train-warm-start-value-checkpoint",
            str(value_path),
            "--train-warm-start-opponent-checkpoint",
            str(opponent_path),
            "--disable-manifest-promotion",
            "--disable-chunk-gate",
        )

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

    def _write_mock_collect_outputs(self, **kwargs):
        Path(kwargs["collect_value_path"]).write_text('{"v": 1}\n', encoding="utf-8")
        Path(kwargs["collect_opponent_path"]).write_text('{"o": 1}\n', encoding="utf-8")
        Path(kwargs["collect_summary_path"]).write_text("{}", encoding="utf-8")
        return [{"profile": "selfplay"}]

    def _write_replay_chunk(
        self,
        root: Path,
        label: str,
        *,
        value_lines: list[str],
        opponent_lines: list[str] | None = None,
    ) -> ReplayChunk:
        replay_dir = root / label / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        value_path = replay_dir / "self_play.value.jsonl"
        opponent_path = replay_dir / "self_play.opponent.jsonl"
        value_path.write_text("".join(value_lines), encoding="utf-8")
        default_opponent_lines = [
            line.replace("old", "opp").replace("new", "opp-new") for line in value_lines
        ]
        opponent_path.write_text(
            "".join(opponent_lines or default_opponent_lines),
            encoding="utf-8",
        )
        return ReplayChunk(label=label, value_path=value_path, opponent_path=opponent_path)

    def test_initialize_selfplay_run_uses_explicit_warm_start_and_builds_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)

            self.assertEqual(args.train_replay_window_chunks, 3)
            with patch("scripts.run_td_loop_selfplay.load_promoted_checkpoints", return_value=[]):
                context = initialize_selfplay_run(args)
                self.assertTrue(context.run_id.endswith("-unit-selfplay"))
                self.assertTrue(context.run_dir.exists())
                self.assertTrue(context.chunks_dir.exists())
                self.assertTrue(context.eval_dir.exists())
                self.assertEqual(context.latest_checkpoint.step, 0)
                self.assertEqual(context.latest_checkpoint.value_path, value_path)
                self.assertEqual(context.latest_checkpoint.opponent_path, opponent_path)
                self.assertEqual(context.incumbent_checkpoint.run_id, "incumbent")
                self.assertEqual(context.incumbent_checkpoint.value_path, value_path)
                self.assertEqual(context.incumbent_checkpoint.opponent_path, opponent_path)
                self.assertEqual(args.train_value_target_mode, "td-lambda")

    def test_validate_args_rejects_td_lambda_with_capped_value_replay_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            args.train_replay_window_max_value_lines = 10

            with self.assertRaises(SystemExit) as raised:
                _validate_args(args)

        self.assertIn("raw line caps can split episode trajectories", str(raised.exception))

    def test_run_selfplay_chunk_records_commands_and_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            next_value, next_opponent = self._make_checkpoints(root / "next")
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            latest_checkpoint = LoopCheckpoint(
                step=0,
                value_path=value_path,
                opponent_path=opponent_path,
            )
            next_checkpoint = LoopCheckpoint(
                step=1000,
                value_path=next_value,
                opponent_path=next_opponent,
            )
            candidate = PoolCheckpoint(
                run_id="candidate",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=value_path,
                opponent_path=opponent_path,
            )

            with patch("scripts.run_td_loop_selfplay.load_promoted_checkpoints", return_value=[]), patch(
                "scripts.run_td_loop_selfplay._build_collect_profiles",
                return_value=[candidate],
            ), patch(
                "scripts.run_td_loop_selfplay._run_collect_profiles",
                side_effect=self._write_mock_collect_outputs,
            ) as mocked_collect, patch(
                "scripts.run_td_loop_selfplay.build_train_command",
                return_value=["python", "-m", "scripts.train_td"],
            ) as mocked_train_command, patch(
                "scripts.run_td_loop_selfplay.run_step",
                return_value=None,
            ), patch(
                "scripts.run_td_loop_selfplay.read_json",
                return_value={"results": {"checkpoints": []}},
            ), patch(
                "scripts.run_td_loop_selfplay.checkpoints_from_train_summary",
                return_value=[next_checkpoint],
            ), patch(
                "scripts.run_td_loop_selfplay.select_latest_checkpoint",
                return_value=next_checkpoint,
            ):
                result = run_selfplay_chunk(
                    args=args,
                    run_id="run-123",
                    chunk_index=1,
                    chunks_dir=root / "artifacts" / "run-123" / "chunks",
                    latest_checkpoint=latest_checkpoint,
                    accepted_replay_chunks=[],
                    progress_path=root / "artifacts" / "run-123" / "progress.json",
                )

        self.assertEqual(result.chunk_label, "chunk-001")
        self.assertEqual(result.latest_checkpoint, next_checkpoint)
        self.assertEqual(result.command_row["collectProfiles"], [{"profile": "selfplay"}])
        self.assertEqual(result.command_row["train"], ["python", "-m", "scripts.train_td"])
        self.assertEqual(result.chunk_row["latestCheckpoint"]["step"], 1000)
        self.assertEqual(result.chunk_row["latestCheckpoint"]["value"], str(next_value))
        self.assertEqual(result.chunk_row["latestCheckpoint"]["opponent"], str(next_opponent))
        collect_kwargs = mocked_collect.call_args.kwargs
        self.assertEqual(collect_kwargs["run_id"], "run-123-chunk-001")
        train_kwargs = mocked_train_command.call_args.kwargs
        self.assertTrue(str(train_kwargs["value_replay"]).endswith("chunk-001\\train\\replay_window\\window.value.jsonl"))
        self.assertTrue(str(train_kwargs["opponent_replay"]).endswith("chunk-001\\train\\replay_window\\window.opponent.jsonl"))
        self.assertEqual(result.chunk_row["replayWindow"]["chunks"][0]["chunk"], "chunk-001")
        assert result.replay_chunk is not None
        self.assertEqual(result.replay_chunk.value_path.name, "self_play.value.jsonl")

    def test_build_replay_window_uses_recent_accepted_chunks_and_current_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root / "warm")
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            args.train_replay_window_chunks = 2
            args.train_replay_window_max_value_lines = 0
            args.train_replay_window_max_opponent_lines = 0
            previous_one = self._write_replay_chunk(root, "chunk-001", value_lines=["old-1\n"])
            previous_two = self._write_replay_chunk(root, "chunk-002", value_lines=["old-2\n"])
            current = self._write_replay_chunk(root, "chunk-003", value_lines=["new\n"])

            result = build_replay_window(
                args=args,
                chunk_label="chunk-003",
                train_dir=root / "chunk-003" / "train",
                accepted_replay_chunks=[previous_one, previous_two],
                current_replay=current,
            )

            self.assertEqual([chunk.label for chunk in result.chunks], ["chunk-002", "chunk-003"])
            self.assertEqual(result.value_path.read_text(encoding="utf-8"), "old-2\nnew\n")
            self.assertEqual(result.value_lines, 2)
            self.assertTrue(result.summary_path.exists())

    def test_run_promotion_stage_records_commands_and_pools_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root)
            incumbent_value, incumbent_opponent = self._make_checkpoints(root / "incumbent")
            args = self._base_args(root, value_path=value_path, opponent_path=opponent_path)
            latest_checkpoint = LoopCheckpoint(
                step=1000,
                value_path=value_path,
                opponent_path=opponent_path,
            )
            incumbent_checkpoint = PoolCheckpoint(
                run_id="incumbent",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=incumbent_value,
                opponent_path=incumbent_opponent,
            )
            baseline_row = self._eval_row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.60,
                ci_low=0.53,
                ci_high=0.68,
                side_gap=0.02,
                candidate_wins=120,
                opponent_wins=80,
                total_games=200,
            )
            incumbent_row = self._eval_row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.60,
                ci_low=0.53,
                ci_high=0.67,
                side_gap=0.01,
                candidate_wins=120,
                opponent_wins=80,
                total_games=200,
            )

            with patch(
                "scripts.run_td_loop_selfplay._run_eval_windows_vs_search",
                return_value=[{"command": ["baseline-cmd"], "row": baseline_row}],
            ), patch(
                "scripts.run_td_loop_selfplay._run_eval_windows_vs_incumbent",
                return_value=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
            ):
                result = run_promotion_stage(
                    args=args,
                    eval_dir=root / "artifacts" / "evals",
                    latest_checkpoint=latest_checkpoint,
                    incumbent_checkpoint=incumbent_checkpoint,
                    progress_path=root / "artifacts" / "progress.json",
                )

        self.assertEqual(result.commands["baselineVsSearch"], [["baseline-cmd"]])
        self.assertEqual(result.commands["candidateVsIncumbent"], [["incumbent-cmd"]])
        self.assertAlmostEqual(result.pooled_baseline.candidate_win_rate, 0.60)
        self.assertAlmostEqual(result.pooled_incumbent.candidate_win_rate, 0.60)
        self.assertTrue(result.promotion["promoted"])

    def test_run_chunk_gate_accepts_candidate_when_small_eval_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_value, candidate_opponent = self._make_checkpoints(root / "candidate")
            accepted_value, accepted_opponent = self._make_checkpoints(root / "accepted")
            args = self._parse(
                "--chunk-gate-games-per-side",
                "5",
                "--chunk-gate-seed-start-indices",
                "40000",
            )
            candidate = LoopCheckpoint(
                step=1000,
                value_path=candidate_value,
                opponent_path=candidate_opponent,
            )
            accepted = LoopCheckpoint(
                step=0,
                value_path=accepted_value,
                opponent_path=accepted_opponent,
            )
            row = self._eval_row(
                artifact_name="gate.json",
                opponent_policy="td-search",
                win_rate=0.60,
                ci_low=0.10,
                ci_high=0.90,
                side_gap=0.02,
                candidate_wins=6,
                opponent_wins=4,
                total_games=10,
            )

            with patch("scripts.run_td_loop_selfplay.run_step") as mocked_run_step, patch(
                "scripts.run_td_loop_selfplay.read_eval_row",
                return_value=row,
            ):
                result = run_chunk_gate(
                    args=args,
                    python_bin=Path(sys.executable),
                    chunk_label="chunk-001",
                    eval_dir=root / "eval",
                    candidate_checkpoint=candidate,
                    accepted_checkpoint=accepted,
                    progress_path=root / "progress.json",
                    log_prefix="[test]",
                )

        self.assertTrue(result.enabled)
        self.assertTrue(result.accepted)
        self.assertEqual(result.accepted_after, candidate)
        self.assertEqual(len(result.commands), 1)
        mocked_run_step.assert_called_once()

    def test_run_checkpoint_selection_selects_best_eval_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_value, first_opponent = self._make_checkpoints(root / "first")
            second_value, second_opponent = self._make_checkpoints(root / "second")
            accepted_value, accepted_opponent = self._make_checkpoints(root / "accepted")
            args = self._parse(
                "--checkpoint-selection-games-per-side",
                "5",
                "--checkpoint-selection-seed-start-indices",
                "45000",
            )
            first = LoopCheckpoint(
                step=1000,
                value_path=first_value,
                opponent_path=first_opponent,
            )
            second = LoopCheckpoint(
                step=2000,
                value_path=second_value,
                opponent_path=second_opponent,
            )
            accepted = LoopCheckpoint(
                step=0,
                value_path=accepted_value,
                opponent_path=accepted_opponent,
            )
            first_row = self._eval_row(
                artifact_name="first.json",
                opponent_policy="td-search",
                win_rate=0.40,
                ci_low=0.10,
                ci_high=0.70,
                side_gap=0.02,
                candidate_wins=4,
                opponent_wins=6,
                total_games=10,
            )
            second_row = self._eval_row(
                artifact_name="second.json",
                opponent_policy="td-search",
                win_rate=0.60,
                ci_low=0.20,
                ci_high=0.90,
                side_gap=0.03,
                candidate_wins=6,
                opponent_wins=4,
                total_games=10,
            )

            with patch("scripts.run_td_loop_selfplay.run_step") as mocked_run_step, patch(
                "scripts.run_td_loop_selfplay.read_eval_row",
                side_effect=[first_row, second_row],
            ):
                result = run_checkpoint_selection(
                    args=args,
                    python_bin=Path(sys.executable),
                    chunk_label="chunk-001",
                    eval_dir=root / "eval" / "checkpoint_selection",
                    checkpoints=[first, second],
                    accepted_checkpoint=accepted,
                    progress_path=root / "progress.json",
                    log_prefix="[test]",
                )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result.selected_checkpoint, second)
        self.assertEqual(result.trained_latest_checkpoint, second)
        self.assertEqual(len(result.commands), 2)
        self.assertEqual(mocked_run_step.call_count, 2)
        self.assertEqual(
            summary["checkpointSelection"]["selectedCheckpoint"]["value"],
            str(second_value),
        )
        self.assertFalse(summary["checkpointSelection"]["candidates"][0]["selected"])
        self.assertTrue(summary["checkpointSelection"]["candidates"][1]["selected"])

    def test_run_checkpoint_selection_records_single_candidate_without_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value_path, opponent_path = self._make_checkpoints(root / "candidate")
            accepted_value, accepted_opponent = self._make_checkpoints(root / "accepted")
            args = self._parse()
            candidate = LoopCheckpoint(
                step=1000,
                value_path=value_path,
                opponent_path=opponent_path,
            )
            accepted = LoopCheckpoint(
                step=0,
                value_path=accepted_value,
                opponent_path=accepted_opponent,
            )

            with patch("scripts.run_td_loop_selfplay.run_step") as mocked_run_step:
                result = run_checkpoint_selection(
                    args=args,
                    python_bin=Path(sys.executable),
                    chunk_label="chunk-001",
                    eval_dir=root / "eval" / "checkpoint_selection",
                    checkpoints=[candidate],
                    accepted_checkpoint=accepted,
                    progress_path=root / "progress.json",
                    log_prefix="[test]",
                )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result.reason, "single_candidate")
        self.assertEqual(result.selected_checkpoint, candidate)
        self.assertEqual(result.commands, [])
        mocked_run_step.assert_not_called()
        self.assertEqual(summary["checkpointSelection"]["candidates"][0]["evaluation"], None)

    def test_run_selfplay_loop_uses_previous_accepted_checkpoint_after_rejected_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm_value, warm_opponent = self._make_checkpoints(root / "warm")
            rejected_value, rejected_opponent = self._make_checkpoints(root / "rejected")
            accepted_value, accepted_opponent = self._make_checkpoints(root / "accepted")
            args = self._parse(
                "--artifact-dir",
                str(root / "artifacts"),
                "--run-label",
                "unit-selfplay",
                "--chunks-per-loop",
                "2",
                "--disable-manifest-promotion",
            )
            run_dir = root / "artifacts" / "run-123"
            chunks_dir = run_dir / "chunks"
            eval_dir = run_dir / "evals"
            for path in (run_dir, chunks_dir, eval_dir):
                path.mkdir(parents=True, exist_ok=True)
            initial = LoopCheckpoint(
                step=0,
                value_path=warm_value,
                opponent_path=warm_opponent,
            )
            rejected = LoopCheckpoint(
                step=1000,
                value_path=rejected_value,
                opponent_path=rejected_opponent,
            )
            accepted = LoopCheckpoint(
                step=2000,
                value_path=accepted_value,
                opponent_path=accepted_opponent,
            )
            context = RunContext(
                run_id="run-123",
                run_dir=run_dir,
                chunks_dir=chunks_dir,
                eval_dir=eval_dir,
                loop_summary_path=run_dir / "loop.summary.json",
                progress_path=run_dir / "progress.json",
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=warm_value,
                    opponent_path=warm_opponent,
                ),
                latest_checkpoint=initial,
                loop_started=time.perf_counter(),
            )
            chunk_one = ChunkExecutionResult(
                chunk_label="chunk-001",
                latest_checkpoint=rejected,
                command_row={"chunk": "chunk-001"},
                chunk_row={"chunk": "chunk-001"},
            )
            chunk_two = ChunkExecutionResult(
                chunk_label="chunk-002",
                latest_checkpoint=accepted,
                command_row={"chunk": "chunk-002"},
                chunk_row={"chunk": "chunk-002"},
            )
            gate_one = ChunkGateResult(
                enabled=True,
                accepted=False,
                reason="chunk_gate_failed",
                candidate_checkpoint=rejected,
                accepted_before=initial,
                accepted_after=initial,
                seed_start_indices=[],
                windows=[],
                pooled=None,
                checks={},
                window_checks=[],
                commands=[["gate-one"]],
            )
            gate_two = ChunkGateResult(
                enabled=True,
                accepted=True,
                reason="chunk_gate_passed",
                candidate_checkpoint=accepted,
                accepted_before=initial,
                accepted_after=accepted,
                seed_start_indices=[],
                windows=[],
                pooled=None,
                checks={},
                window_checks=[],
                commands=[["gate-two"]],
            )
            baseline_row = self._eval_row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.55,
                ci_low=0.51,
                ci_high=0.59,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            incumbent_row = self._eval_row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.53,
                ci_low=0.50,
                ci_high=0.56,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            promotion_stage = PromotionStageResult(
                baseline_windows=[{"command": ["baseline-cmd"], "row": baseline_row}],
                incumbent_windows=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
                pooled_baseline=baseline_row,
                pooled_incumbent=incumbent_row,
                promotion={"promoted": True, "reason": "dual_gate_passed"},
                commands={
                    "baselineVsSearch": [["baseline-cmd"]],
                    "candidateVsIncumbent": [["incumbent-cmd"]],
                },
            )

            with patch(
                "scripts.run_td_loop_selfplay.initialize_selfplay_run",
                return_value=context,
            ), patch(
                "scripts.run_td_loop_selfplay.run_selfplay_chunk",
                side_effect=[chunk_one, chunk_two],
            ) as mocked_chunk, patch(
                "scripts.run_td_loop_selfplay.run_chunk_gate",
                side_effect=[gate_one, gate_two],
            ), patch(
                "scripts.run_td_loop_selfplay.run_promotion_stage",
                return_value=promotion_stage,
            ) as mocked_promotion, patch("builtins.print"):
                result = run_selfplay_loop(args)

            payload = json.loads(context.loop_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(mocked_chunk.call_args_list[0].kwargs["latest_checkpoint"], initial)
        self.assertEqual(mocked_chunk.call_args_list[1].kwargs["latest_checkpoint"], initial)
        self.assertEqual(mocked_promotion.call_args.kwargs["latest_checkpoint"], accepted)
        self.assertFalse(payload["chunks"][0]["generatorGate"]["accepted"])
        self.assertEqual(payload["chunks"][0]["latestCheckpoint"]["value"], str(warm_value))
        self.assertTrue(payload["chunks"][1]["generatorGate"]["accepted"])

    def test_build_selfplay_loop_summary_contains_expected_sections(self) -> None:
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
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                latest_checkpoint=LoopCheckpoint(
                    step=0,
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                loop_started=0.0,
            )
            baseline_row = self._eval_row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.55,
                ci_low=0.50,
                ci_high=0.60,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            incumbent_row = self._eval_row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.53,
                ci_low=0.50,
                ci_high=0.56,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            promotion_stage = PromotionStageResult(
                baseline_windows=[{"command": ["baseline-cmd"], "row": baseline_row}],
                incumbent_windows=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
                pooled_baseline=baseline_row,
                pooled_incumbent=incumbent_row,
                promotion={"promoted": True, "reason": "dual_gate_passed"},
                commands={
                    "baselineVsSearch": [["baseline-cmd"]],
                    "candidateVsIncumbent": [["incumbent-cmd"]],
                },
            )

            payload = build_selfplay_loop_summary(
                args=args,
                context=context,
                commands={
                    "chunks": [{"chunk": "chunk-001"}],
                    "promotionEvals": promotion_stage.commands,
                },
                chunk_rows=[{"chunk": "chunk-001"}],
                promotion_stage=promotion_stage,
                elapsed_minutes=1.234,
            )

        self.assertEqual(payload["runId"], "run-123")
        self.assertIn("config", payload)
        self.assertIn("artifacts", payload)
        self.assertEqual(payload["commands"]["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(payload["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(
            payload["evaluation"]["baselineVsSearch"]["windows"][0]["seedStartIndex"],
            0,
        )
        self.assertEqual(
            payload["evaluation"]["candidateVsIncumbent"]["windows"][0]["seedStartIndex"],
            100,
        )
        self.assertTrue(payload["promotion"]["promoted"])

    def test_run_selfplay_loop_writes_loop_summary_and_prints_terminal_report(self) -> None:
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
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                latest_checkpoint=LoopCheckpoint(
                    step=0,
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                loop_started=time.perf_counter(),
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
                    "collectProfiles": [{"profile": "selfplay"}],
                    "train": ["python", "-m", "scripts.train_td"],
                },
                chunk_row={
                    "chunk": "chunk-001",
                    "replayRegime": "chunk-local-selfplay-mixed",
                    "collectSummary": str(run_dir / "chunks" / "chunk-001" / "replay" / "self_play.summary.json"),
                    "trainSummary": str(run_dir / "chunks" / "chunk-001" / "train" / "summary.json"),
                    "latestCheckpoint": {
                        "step": 1000,
                        "value": str(next_value),
                        "opponent": str(next_opponent),
                    },
                },
            )
            baseline_row = self._eval_row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.55,
                ci_low=0.51,
                ci_high=0.59,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            incumbent_row = self._eval_row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.53,
                ci_low=0.50,
                ci_high=0.56,
                side_gap=0.01,
                candidate_wins=11,
                opponent_wins=9,
                total_games=20,
            )
            promotion_stage = PromotionStageResult(
                baseline_windows=[{"command": ["baseline-cmd"], "row": baseline_row}],
                incumbent_windows=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
                pooled_baseline=baseline_row,
                pooled_incumbent=incumbent_row,
                promotion={"promoted": True, "reason": "dual_gate_passed"},
                commands={
                    "baselineVsSearch": [["baseline-cmd"]],
                    "candidateVsIncumbent": [["incumbent-cmd"]],
                },
            )

            with patch(
                "scripts.run_td_loop_selfplay.initialize_selfplay_run",
                return_value=context,
            ), patch(
                "scripts.run_td_loop_selfplay.run_selfplay_chunk",
                return_value=chunk_result,
            ) as mocked_chunk, patch(
                "scripts.run_td_loop_selfplay.run_promotion_stage",
                return_value=promotion_stage,
            ), patch("builtins.print") as mocked_print:
                result = run_selfplay_loop(args)

            payload = json.loads(context.loop_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(mocked_chunk.call_count, 1)
        self.assertEqual(payload["runId"], "run-123")
        self.assertEqual(payload["commands"]["chunks"][0]["chunk"], "chunk-001")
        self.assertEqual(payload["promotion"]["reason"], "dual_gate_passed")
        printed = json.loads(mocked_print.call_args.args[0])
        self.assertEqual(printed["runId"], "run-123")
        self.assertTrue(printed["promoted"])

    def test_run_selfplay_loop_raises_without_writing_summary_when_promotion_fails(self) -> None:
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
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                latest_checkpoint=LoopCheckpoint(
                    step=0,
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
                loop_started=time.perf_counter(),
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
                    "collectProfiles": [{"profile": "selfplay"}],
                    "train": ["python", "-m", "scripts.train_td"],
                },
                chunk_row={"chunk": "chunk-001"},
            )

            with patch(
                "scripts.run_td_loop_selfplay.initialize_selfplay_run",
                return_value=context,
            ), patch(
                "scripts.run_td_loop_selfplay.run_selfplay_chunk",
                return_value=chunk_result,
            ), patch(
                "scripts.run_td_loop_selfplay.run_promotion_stage",
                side_effect=SystemExit("promotion failed"),
            ), patch("builtins.print") as mocked_print:
                with self.assertRaises(SystemExit):
                    run_selfplay_loop(args)

        self.assertFalse(context.loop_summary_path.exists())
        mocked_print.assert_not_called()


class RunTdLoopSelfplaySmokeTests(unittest.TestCase):
    def test_script_smoke_runs_small_selfplay_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts"
            value_path = root / "warm.value.pt"
            opponent_path = root / "warm.opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=32,
                ),
                output_path=opponent_path,
            )

            command = [
                sys.executable,
                "-m",
                "scripts.run_td_loop_selfplay",
                "--artifact-dir",
                str(artifact_dir),
                "--run-label",
                "selfplay-smoke",
                "--chunks-per-loop",
                "1",
                "--collect-games",
                "2",
                "--collect-workers",
                "1",
                "--collect-selfplay-share",
                "1.0",
                "--collect-pool-share",
                "0.0",
                "--collect-search-anchor-share",
                "0.0",
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
                "--incumbent-eval-games-per-side",
                "1",
                "--incumbent-eval-workers",
                "1",
                "--incumbent-eval-seed-start-indices",
                "100",
                "--progress-heartbeat-minutes",
                "0",
                "--train-warm-start-value-checkpoint",
                str(value_path),
                "--train-warm-start-opponent-checkpoint",
                str(opponent_path),
                "--disable-manifest-promotion",
                "--disable-chunk-gate",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                combined_output = f"{completed.stdout}\n{completed.stderr}"
                if "spawn EPERM" in combined_output:
                    self.skipTest("Node bridge child-process spawn is blocked in the sandboxed test environment.")
                raise AssertionError(
                    "self-play loop smoke run failed.\n"
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
