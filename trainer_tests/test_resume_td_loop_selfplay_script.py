from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts.opponent_pool import PoolCheckpoint
from scripts.resume_td_loop_selfplay import (
    CompletedChunk,
    ResumeCollectTemplate,
    ResumeState,
    _run_or_load_eval_windows_vs_search,
    main,
)
from scripts.td_loop_common import LoopCheckpoint
from scripts.td_loop_eval_common import EvalRow


class ResumeTdLoopSelfplayScriptTests(unittest.TestCase):
    def _row(
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

    def test_main_resumes_next_chunk_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "artifacts" / "run-123"
            chunks_dir = run_dir / "chunks"
            eval_dir = run_dir / "evals"
            for path in (chunks_dir, eval_dir):
                path.mkdir(parents=True, exist_ok=True)

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            next_value = root / "next.value.pt"
            next_opponent = root / "next.opponent.pt"
            for path in (warm_value, warm_opponent, next_value, next_opponent):
                path.write_text("x", encoding="utf-8")

            completed_chunk = CompletedChunk(
                index=1,
                label="chunk-001",
                chunk_dir=chunks_dir / "chunk-001",
                collect_summary=chunks_dir / "chunk-001" / "replay" / "self_play.summary.json",
                train_summary=chunks_dir / "chunk-001" / "train" / "summary.json",
                latest_checkpoint=LoopCheckpoint(
                    step=1000,
                    value_path=warm_value,
                    opponent_path=warm_opponent,
                ),
            )
            state = ResumeState(
                run_id="run-123",
                run_dir=run_dir,
                chunks_dir=chunks_dir,
                eval_dir=eval_dir,
                loop_summary_path=run_dir / "loop.summary.json",
                progress_path=run_dir / "progress.json",
                completed_chunks=[completed_chunk],
                latest_checkpoint=completed_chunk.latest_checkpoint,
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=warm_value,
                    opponent_path=warm_opponent,
                ),
                collect_templates=[
                    ResumeCollectTemplate(
                        label="selfplay",
                        games=600,
                        player_a_policy="td-search",
                        player_b_policy="td-search",
                        player_b_fixed_td_search=None,
                        player_b_uses_candidate=True,
                    )
                ],
                collect_games_per_chunk=600,
                collect_workers=2,
                train_config={},
                partial_chunk_label="chunk-002",
                highest_existing_chunk_index=2,
            )
            resolved_args = Namespace(
                chunks_per_loop=2,
                python_bin=Path(sys.executable),
                progress_heartbeat_minutes=0.0,
                eval_seed_start_indices=[0],
                incumbent_eval_seed_start_indices=[100],
            )
            baseline_row = self._row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.56,
                ci_low=0.52,
                ci_high=0.60,
                side_gap=0.02,
                candidate_wins=112,
                opponent_wins=88,
                total_games=200,
            )
            incumbent_row = self._row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.54,
                ci_low=0.50,
                ci_high=0.58,
                side_gap=0.01,
                candidate_wins=108,
                opponent_wins=92,
                total_games=200,
            )
            next_checkpoint = LoopCheckpoint(
                step=2000,
                value_path=next_value,
                opponent_path=next_opponent,
            )

            with patch(
                "scripts.resume_td_loop_selfplay.parse_args",
                return_value=Namespace(
                    run_id="run-123",
                    artifact_dir=root / "artifacts",
                    python_bin=Path(sys.executable),
                    force_summary=False,
                    force_eval=False,
                ),
            ), patch(
                "scripts.resume_td_loop_selfplay._require_supported_runtime",
                return_value=None,
            ), patch(
                "scripts.resume_td_loop_selfplay._discover_resume_state",
                return_value=state,
            ), patch(
                "scripts.resume_td_loop_selfplay._resolve_resume_args",
                return_value=resolved_args,
            ), patch(
                "scripts.resume_td_loop_selfplay._validate_args",
                return_value=None,
            ), patch(
                "scripts.resume_td_loop_selfplay._build_collect_profiles_from_templates",
                return_value=["selfplay-profile"],
            ), patch(
                "scripts.resume_td_loop_selfplay._run_collect_profiles",
                return_value=[{"profile": "selfplay"}],
            ) as mocked_collect, patch(
                "scripts.resume_td_loop_selfplay.build_train_command",
                return_value=["python", "-m", "scripts.train_td"],
            ), patch(
                "scripts.resume_td_loop_selfplay.run_step",
                return_value=None,
            ), patch(
                "scripts.resume_td_loop_selfplay.read_json",
                return_value={"results": {"checkpoints": []}},
            ), patch(
                "scripts.resume_td_loop_selfplay.checkpoints_from_train_summary",
                return_value=[next_checkpoint],
            ), patch(
                "scripts.resume_td_loop_selfplay.select_latest_checkpoint",
                return_value=next_checkpoint,
            ), patch(
                "scripts.resume_td_loop_selfplay._run_or_load_eval_windows_vs_search",
                return_value=[{"command": ["baseline-cmd"], "row": baseline_row}],
            ), patch(
                "scripts.resume_td_loop_selfplay._run_or_load_eval_windows_vs_incumbent",
                return_value=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
            ), patch(
                "scripts.resume_td_loop_selfplay._promotion_decision",
                return_value={"promoted": True, "reason": "dual_gate_passed"},
            ), patch(
                "scripts.resume_td_loop_selfplay._config_payload",
                return_value={"resumed": True},
            ), patch("builtins.print") as mocked_print:
                result = main()

            payload = json.loads(state.loop_summary_path.read_text(encoding="utf-8"))
            terminal_report = json.loads(mocked_print.call_args_list[-1].args[0])

        self.assertEqual(result, 0)
        self.assertEqual(payload["runId"], "run-123")
        self.assertEqual(payload["resume"]["resumedAfterChunk"], "chunk-001")
        self.assertEqual(payload["resume"]["discardedPartialChunk"], "chunk-002")
        self.assertEqual(payload["commands"]["completedChunks"], ["chunk-001"])
        self.assertEqual(len(payload["commands"]["resumedChunks"]), 1)
        self.assertEqual(payload["promotion"]["reason"], "dual_gate_passed")
        self.assertEqual(terminal_report["runId"], "run-123")
        self.assertTrue(terminal_report["promoted"])
        collect_kwargs = mocked_collect.call_args.kwargs
        self.assertEqual(collect_kwargs["run_id"], "run-123-chunk-002")

    def test_main_runs_eval_only_when_all_chunks_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "artifacts" / "run-123"
            chunks_dir = run_dir / "chunks"
            eval_dir = run_dir / "evals"
            for path in (chunks_dir, eval_dir):
                path.mkdir(parents=True, exist_ok=True)

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            for path in (warm_value, warm_opponent):
                path.write_text("x", encoding="utf-8")

            completed_chunk = CompletedChunk(
                index=1,
                label="chunk-001",
                chunk_dir=chunks_dir / "chunk-001",
                collect_summary=chunks_dir / "chunk-001" / "replay" / "self_play.summary.json",
                train_summary=chunks_dir / "chunk-001" / "train" / "summary.json",
                latest_checkpoint=LoopCheckpoint(
                    step=1000,
                    value_path=warm_value,
                    opponent_path=warm_opponent,
                ),
            )
            state = ResumeState(
                run_id="run-123",
                run_dir=run_dir,
                chunks_dir=chunks_dir,
                eval_dir=eval_dir,
                loop_summary_path=run_dir / "loop.summary.json",
                progress_path=run_dir / "progress.json",
                completed_chunks=[completed_chunk],
                latest_checkpoint=completed_chunk.latest_checkpoint,
                incumbent_checkpoint=PoolCheckpoint(
                    run_id="incumbent",
                    generated_at_utc="2026-01-01T00:00:00+00:00",
                    value_path=warm_value,
                    opponent_path=warm_opponent,
                ),
                collect_templates=[],
                collect_games_per_chunk=600,
                collect_workers=2,
                train_config={},
                partial_chunk_label=None,
                highest_existing_chunk_index=1,
            )
            resolved_args = Namespace(
                chunks_per_loop=1,
                python_bin=Path(sys.executable),
                progress_heartbeat_minutes=0.0,
                eval_seed_start_indices=[0],
                incumbent_eval_seed_start_indices=[100],
            )
            baseline_row = self._row(
                artifact_name="baseline.json",
                opponent_policy="search",
                win_rate=0.56,
                ci_low=0.52,
                ci_high=0.60,
                side_gap=0.02,
                candidate_wins=112,
                opponent_wins=88,
                total_games=200,
            )
            incumbent_row = self._row(
                artifact_name="incumbent.json",
                opponent_policy="td-search",
                win_rate=0.54,
                ci_low=0.50,
                ci_high=0.58,
                side_gap=0.01,
                candidate_wins=108,
                opponent_wins=92,
                total_games=200,
            )

            with patch(
                "scripts.resume_td_loop_selfplay.parse_args",
                return_value=Namespace(
                    run_id="run-123",
                    artifact_dir=root / "artifacts",
                    python_bin=Path(sys.executable),
                    force_summary=False,
                    force_eval=False,
                ),
            ), patch(
                "scripts.resume_td_loop_selfplay._require_supported_runtime",
                return_value=None,
            ), patch(
                "scripts.resume_td_loop_selfplay._discover_resume_state",
                return_value=state,
            ), patch(
                "scripts.resume_td_loop_selfplay._resolve_resume_args",
                return_value=resolved_args,
            ), patch(
                "scripts.resume_td_loop_selfplay._validate_args",
                return_value=None,
            ), patch(
                "scripts.resume_td_loop_selfplay._run_collect_profiles",
                side_effect=AssertionError("collect should not rerun"),
            ), patch(
                "scripts.resume_td_loop_selfplay.build_train_command",
                side_effect=AssertionError("train should not rerun"),
            ), patch(
                "scripts.resume_td_loop_selfplay.run_step",
                side_effect=AssertionError("chunk steps should not rerun"),
            ), patch(
                "scripts.resume_td_loop_selfplay._run_or_load_eval_windows_vs_search",
                return_value=[{"command": ["baseline-cmd"], "row": baseline_row}],
            ), patch(
                "scripts.resume_td_loop_selfplay._run_or_load_eval_windows_vs_incumbent",
                return_value=[{"command": ["incumbent-cmd"], "row": incumbent_row}],
            ), patch(
                "scripts.resume_td_loop_selfplay._promotion_decision",
                return_value={"promoted": False, "reason": "eval_only"},
            ), patch(
                "scripts.resume_td_loop_selfplay._config_payload",
                return_value={"resumed": True},
            ), patch("builtins.print") as mocked_print:
                result = main()

            payload = json.loads(state.loop_summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(payload["commands"]["resumedChunks"], [])
        self.assertEqual(payload["promotion"]["reason"], "eval_only")
        self.assertIn("all collect/train chunks are already complete", mocked_print.call_args_list[0].args[0])

    def test_run_or_load_eval_windows_vs_search_reuses_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "evals"
            eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = eval_dir / "promotion_eval.baseline.seed-000000.json"
            out_path.write_text(
                json.dumps(
                    {
                        "results": {
                            "candidateWinRate": 0.58,
                            "candidateWinRateCi95": {"low": 0.52, "high": 0.64},
                            "sideGap": 0.02,
                            "candidateWins": 116,
                            "opponentWins": 84,
                            "draws": 0,
                            "totalGames": 200,
                            "candidateWinRateAsPlayerA": 0.58,
                            "candidateWinRateAsPlayerB": 0.58,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                python_bin=Path(sys.executable),
                eval_seed_start_indices=[0],
                eval_seed_prefix="baseline",
                eval_workers=1,
                eval_games_per_side=100,
                progress_heartbeat_minutes=0.0,
            )
            checkpoint = LoopCheckpoint(
                step=1000,
                value_path=root / "value.pt",
                opponent_path=root / "opponent.pt",
            )

            with patch(
                "scripts.resume_td_loop_selfplay._build_eval_command_vs_search",
                return_value=["baseline-cmd"],
            ), patch("scripts.resume_td_loop_selfplay.run_step") as mocked_run_step, patch(
                "builtins.print"
            ):
                rows = _run_or_load_eval_windows_vs_search(
                    args=args,
                    eval_dir=eval_dir,
                    checkpoint=checkpoint,
                    progress_path=root / "progress.json",
                    force_eval=False,
                )

        self.assertEqual(rows[0]["command"], ["baseline-cmd"])
        self.assertEqual(rows[0]["row"].artifact, out_path)
        mocked_run_step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
