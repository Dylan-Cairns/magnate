from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.opponent_pool import PoolCheckpoint
from scripts.resume_td_loop_selfplay import (
    _build_collect_profiles_from_templates,
    _collect_templates_from_summary,
    _discover_resume_state,
)


class ResumeTdLoopSelfplayTests(unittest.TestCase):
    def test_collect_templates_recover_candidate_and_fixed_pool_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_value = root / "candidate.value.pt"
            candidate_opponent = root / "candidate.opponent.pt"
            pool_value = root / "pool.value.pt"
            pool_opponent = root / "pool.opponent.pt"
            for path in (candidate_value, candidate_opponent, pool_value, pool_opponent):
                path.write_text("x", encoding="utf-8")

            payload = {
                "config": {
                    "games": 600,
                    "profiles": [
                        {
                            "label": "selfplay",
                            "games": 510,
                            "playerAPolicy": "td-search",
                            "playerBPolicy": "td-search",
                            "playerATdSearchCheckpoint": {
                                "runId": "run-123",
                                "value": str(candidate_value),
                                "opponent": str(candidate_opponent),
                            },
                            "playerBTdSearchCheckpoint": {
                                "runId": "run-123",
                                "value": str(candidate_value),
                                "opponent": str(candidate_opponent),
                            },
                        },
                        {
                            "label": "pool-01",
                            "games": 60,
                            "playerAPolicy": "td-search",
                            "playerBPolicy": "td-search",
                            "playerATdSearchCheckpoint": {
                                "runId": "run-123",
                                "value": str(candidate_value),
                                "opponent": str(candidate_opponent),
                            },
                            "playerBTdSearchCheckpoint": {
                                "runId": "older-run",
                                "value": str(pool_value),
                                "opponent": str(pool_opponent),
                            },
                        },
                        {
                            "label": "search-anchor",
                            "games": 30,
                            "playerAPolicy": "td-search",
                            "playerBPolicy": "search",
                            "playerATdSearchCheckpoint": {
                                "runId": "run-123",
                                "value": str(candidate_value),
                                "opponent": str(candidate_opponent),
                            },
                            "playerBTdSearchCheckpoint": None,
                        },
                    ],
                }
            }

            templates = _collect_templates_from_summary(
                payload=payload,
                current_run_id="run-123",
            )
            candidate = PoolCheckpoint(
                run_id="run-123",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=candidate_value,
                opponent_path=candidate_opponent,
            )
            profiles = _build_collect_profiles_from_templates(
                templates=templates,
                candidate=candidate,
            )

        self.assertEqual([template.label for template in templates], ["selfplay", "pool-01", "search-anchor"])
        self.assertTrue(templates[0].player_b_uses_candidate)
        self.assertFalse(templates[1].player_b_uses_candidate)
        self.assertIsNotNone(templates[1].player_b_fixed_td_search)
        self.assertIsNone(templates[2].player_b_fixed_td_search)
        self.assertEqual(profiles[0].player_b_td_search, candidate)
        self.assertEqual(profiles[1].player_b_td_search, templates[1].player_b_fixed_td_search)
        self.assertIsNone(profiles[2].player_b_td_search)

    def test_discover_resume_state_uses_latest_complete_chunk_and_collect_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            run_dir = artifact_dir / run_id
            chunks_dir = run_dir / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate1_value = root / "chunk1.value.pt"
            candidate1_opponent = root / "chunk1.opponent.pt"
            candidate2_value = root / "chunk2.value.pt"
            candidate2_opponent = root / "chunk2.opponent.pt"
            for path in (
                warm_value,
                warm_opponent,
                candidate1_value,
                candidate1_opponent,
                candidate2_value,
                candidate2_opponent,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate1_value,
                latest_opponent=candidate1_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=5,
            )
            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=2,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate2_value,
                latest_opponent=candidate2_opponent,
                current_run_id=run_id,
                candidate_value=candidate1_value,
                candidate_opponent=candidate1_opponent,
                shard_count=5,
            )

            partial_chunk_dir = chunks_dir / "chunk-003"
            (partial_chunk_dir / "replay").mkdir(parents=True, exist_ok=True)
            (partial_chunk_dir / "train").mkdir(parents=True, exist_ok=True)
            (partial_chunk_dir / "replay" / "self_play.summary.json").write_text(
                json.dumps({"config": {"games": 600, "profiles": []}, "results": {"games": 600}}),
                encoding="utf-8",
            )

            state = _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertEqual(len(state.completed_chunks), 2)
        self.assertEqual(state.completed_chunks[-1].label, "chunk-002")
        self.assertEqual(state.latest_checkpoint.step, 10000)
        self.assertEqual(state.collect_games_per_chunk, 600)
        self.assertEqual(state.collect_workers, 5)
        self.assertEqual(state.partial_chunk_label, "chunk-003")
        self.assertEqual(state.incumbent_checkpoint.value_path, warm_value)
        self.assertEqual(state.incumbent_checkpoint.opponent_path, warm_opponent)
        self.assertEqual([template.label for template in state.collect_templates], ["selfplay", "search-anchor"])
        self.assertEqual([chunk.label for chunk in state.accepted_replay_chunks], ["chunk-001", "chunk-002"])

    def test_discover_resume_state_marks_last_trained_chunk_as_pending_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            chunks_dir = artifact_dir / run_id / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate_value = root / "chunk1.value.pt"
            candidate_opponent = root / "chunk1.opponent.pt"
            for path in (warm_value, warm_opponent, candidate_value, candidate_opponent):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate_value,
                latest_opponent=candidate_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=2,
                write_chunk_summary=False,
            )

            state = _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertEqual(len(state.completed_chunks), 0)
        self.assertIsNotNone(state.pending_gate_chunk)
        assert state.pending_gate_chunk is not None
        self.assertEqual(state.pending_gate_chunk.label, "chunk-001")
        self.assertEqual(state.pending_gate_chunk.train_checkpoints[-1].value_path, candidate_value)
        assert state.pending_gate_chunk.replay_chunk is not None
        self.assertEqual(state.pending_gate_chunk.replay_chunk.label, "chunk-001")
        self.assertEqual(state.latest_checkpoint.value_path, warm_value)
        self.assertEqual(state.latest_checkpoint.opponent_path, warm_opponent)

    def test_discover_resume_state_excludes_rejected_chunk_from_replay_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            chunks_dir = artifact_dir / run_id / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate_value = root / "chunk1.value.pt"
            candidate_opponent = root / "chunk1.opponent.pt"
            for path in (warm_value, warm_opponent, candidate_value, candidate_opponent):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate_value,
                latest_opponent=candidate_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=2,
                gate_accepted=False,
                accepted_value=warm_value,
                accepted_opponent=warm_opponent,
            )

            state = _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertEqual(len(state.completed_chunks), 1)
        self.assertEqual(state.accepted_replay_chunks, [])
        self.assertIsNone(state.completed_chunks[0].replay_chunk)
        self.assertEqual(state.latest_checkpoint.value_path, warm_value)

    def test_discover_resume_state_rejects_completed_chunk_without_chunk_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            chunks_dir = artifact_dir / run_id / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate_value = root / "chunk1.value.pt"
            candidate_opponent = root / "chunk1.opponent.pt"
            for path in (warm_value, warm_opponent, candidate_value, candidate_opponent):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate_value,
                latest_opponent=candidate_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=2,
                write_chunk_summary=False,
            )
            (chunks_dir / "chunk-002" / "replay").mkdir(parents=True, exist_ok=True)

            with self.assertRaises(SystemExit) as raised:
                _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertIn("missing required chunk summary", str(raised.exception))

    def test_discover_resume_state_rejects_chunk_summary_without_replay_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            chunks_dir = artifact_dir / run_id / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate_value = root / "chunk1.value.pt"
            candidate_opponent = root / "chunk1.opponent.pt"
            for path in (warm_value, warm_opponent, candidate_value, candidate_opponent):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate_value,
                latest_opponent=candidate_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=2,
            )
            chunk_summary = chunks_dir / "chunk-001" / "chunk.summary.json"
            payload = json.loads(chunk_summary.read_text(encoding="utf-8"))
            del payload["chunk"]["replayWindow"]
            chunk_summary.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertIn("missing required keys", str(raised.exception))

    def test_discover_resume_state_rejects_chunk_summary_without_checkpoint_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifacts" / "td_loops"
            run_id = "20260416-003941Z-td-loop-selfplay-r1-laptop"
            chunks_dir = artifact_dir / run_id / "chunks"

            warm_value = root / "warm.value.pt"
            warm_opponent = root / "warm.opponent.pt"
            candidate_value = root / "chunk1.value.pt"
            candidate_opponent = root / "chunk1.opponent.pt"
            for path in (warm_value, warm_opponent, candidate_value, candidate_opponent):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            self._write_completed_chunk(
                chunks_dir=chunks_dir,
                chunk_index=1,
                warm_value=warm_value,
                warm_opponent=warm_opponent,
                latest_value=candidate_value,
                latest_opponent=candidate_opponent,
                current_run_id=run_id,
                candidate_value=warm_value,
                candidate_opponent=warm_opponent,
                shard_count=2,
            )
            chunk_summary = chunks_dir / "chunk-001" / "chunk.summary.json"
            payload = json.loads(chunk_summary.read_text(encoding="utf-8"))
            del payload["chunk"]["checkpointSelection"]
            chunk_summary.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                _discover_resume_state(run_id=run_id, artifact_dir=artifact_dir)

        self.assertIn("missing required keys", str(raised.exception))

    def _write_completed_chunk(
        self,
        *,
        chunks_dir: Path,
        chunk_index: int,
        warm_value: Path,
        warm_opponent: Path,
        latest_value: Path,
        latest_opponent: Path,
        current_run_id: str,
        candidate_value: Path,
        candidate_opponent: Path,
        shard_count: int,
        write_chunk_summary: bool = True,
        gate_accepted: bool = True,
        accepted_value: Path | None = None,
        accepted_opponent: Path | None = None,
    ) -> None:
        chunk_dir = chunks_dir / f"chunk-{chunk_index:03d}"
        replay_dir = chunk_dir / "replay"
        train_dir = chunk_dir / "train"
        profiles_dir = replay_dir / "profiles"
        shard_dir = profiles_dir / "selfplay.shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        train_dir.mkdir(parents=True, exist_ok=True)
        (replay_dir / "self_play.value.jsonl").write_text('{"v": 1}\n', encoding="utf-8")
        (replay_dir / "self_play.opponent.jsonl").write_text('{"o": 1}\n', encoding="utf-8")

        for shard_index in range(1, shard_count + 1):
            (shard_dir / f"s{shard_index:02d}.summary.json").write_text("{}", encoding="utf-8")

        collect_summary = {
            "config": {
                "games": 600,
                "profiles": [
                    {
                        "label": "selfplay",
                        "games": 510,
                        "playerAPolicy": "td-search",
                        "playerBPolicy": "td-search",
                        "playerATdSearchCheckpoint": {
                            "runId": current_run_id,
                            "value": str(candidate_value),
                            "opponent": str(candidate_opponent),
                        },
                        "playerBTdSearchCheckpoint": {
                            "runId": current_run_id,
                            "value": str(candidate_value),
                            "opponent": str(candidate_opponent),
                        },
                    },
                    {
                        "label": "search-anchor",
                        "games": 90,
                        "playerAPolicy": "td-search",
                        "playerBPolicy": "search",
                        "playerATdSearchCheckpoint": {
                            "runId": current_run_id,
                            "value": str(candidate_value),
                            "opponent": str(candidate_opponent),
                        },
                        "playerBTdSearchCheckpoint": None,
                    },
                ],
            },
            "results": {
                "games": 600,
                "valueTransitions": 1000,
                "opponentSamples": 1000,
            },
        }
        (replay_dir / "self_play.summary.json").write_text(
            json.dumps(collect_summary),
            encoding="utf-8",
        )

        train_summary = {
            "config": {
                "steps": 10000,
                "seed": 0,
                "trainValue": True,
                "trainOpponent": True,
                "valueBatchSize": 128,
                "opponentBatchSize": 64,
                "hiddenDim": 256,
                "gamma": 0.995,
                "valueLearningRate": 0.0003,
                "valueWeightDecay": 1e-05,
                "opponentLearningRate": 0.0003,
                "opponentWeightDecay": 1e-05,
                "maxGradNorm": 1.0,
                "targetSyncInterval": 200,
                "valueTargetMode": "td0",
                "tdLambda": 0.7,
                "valueLoss": "huber",
                "saveEverySteps": 1000,
                "progressEverySteps": 50,
                "warmStartValueCheckpoint": str(warm_value),
                "warmStartOpponentCheckpoint": str(warm_opponent),
                "numThreads": 5,
                "numInteropThreads": 1,
            },
            "results": {
                "checkpoints": [
                    {
                        "step": 10000,
                        "value": str(latest_value),
                        "opponent": str(latest_opponent),
                    }
                ]
            },
        }
        (train_dir / "summary.json").write_text(json.dumps(train_summary), encoding="utf-8")
        replay_window_dir = train_dir / "replay_window"
        replay_window_dir.mkdir(parents=True, exist_ok=True)
        replay_window_payload = {
            "source": "accepted",
            "windowSize": 3,
            "chunks": [
                {
                    "chunk": f"chunk-{chunk_index:03d}",
                    "valueReplay": str(replay_dir / "self_play.value.jsonl"),
                    "opponentReplay": str(replay_dir / "self_play.opponent.jsonl"),
                    "valueLines": 1,
                    "opponentLines": 1,
                }
            ],
            "valueReplayFiles": [str(replay_dir / "self_play.value.jsonl")],
            "opponentReplayFiles": [str(replay_dir / "self_play.opponent.jsonl")],
            "summary": str(replay_window_dir / "window.summary.json"),
            "valueLines": 1,
            "opponentLines": 1,
            "maxValueLines": 0,
            "maxOpponentLines": 0,
        }
        (replay_window_dir / "window.summary.json").write_text(
            json.dumps(replay_window_payload),
            encoding="utf-8",
        )
        if write_chunk_summary:
            self._write_chunk_summary(
                chunks_dir=chunks_dir,
                chunk_index=chunk_index,
                accepted=gate_accepted,
                candidate_value=latest_value,
                candidate_opponent=latest_opponent,
                accepted_value=accepted_value or latest_value,
                accepted_opponent=accepted_opponent or latest_opponent,
                replay_window=replay_window_payload,
            )

    def _write_chunk_summary(
        self,
        *,
        chunks_dir: Path,
        chunk_index: int,
        accepted: bool,
        candidate_value: Path,
        candidate_opponent: Path,
        accepted_value: Path,
        accepted_opponent: Path,
        replay_window: dict,
    ) -> None:
        chunk_dir = chunks_dir / f"chunk-{chunk_index:03d}"
        chunk_label = f"chunk-{chunk_index:03d}"
        payload = {
            "chunk": {
                "chunk": chunk_label,
                "collectSummary": str(chunk_dir / "replay" / "self_play.summary.json"),
                "trainSummary": str(chunk_dir / "train" / "summary.json"),
                "trainedLatestCheckpoint": {
                    "step": 10000,
                    "value": str(candidate_value),
                    "opponent": str(candidate_opponent),
                },
                "checkpointSelection": {
                    "enabled": True,
                    "reason": "single_candidate",
                    "summary": str(chunk_dir / "eval" / "checkpoint_selection" / "summary.json"),
                    "selectedCheckpoint": {
                        "step": 10000,
                        "value": str(candidate_value),
                        "opponent": str(candidate_opponent),
                    },
                    "trainedLatestCheckpoint": {
                        "step": 10000,
                        "value": str(candidate_value),
                        "opponent": str(candidate_opponent),
                    },
                    "acceptedBefore": {
                        "step": 0,
                        "value": str(accepted_value),
                        "opponent": str(accepted_opponent),
                    },
                    "seedStartIndices": [],
                    "candidates": [],
                },
                "candidateCheckpoint": {
                    "step": 10000,
                    "value": str(candidate_value),
                    "opponent": str(candidate_opponent),
                },
                "acceptedCheckpoint": {
                    "step": 10000 if accepted else 0,
                    "value": str(accepted_value),
                    "opponent": str(accepted_opponent),
                },
                "latestCheckpoint": {
                    "step": 0 if not accepted else 10000,
                    "value": str(accepted_value),
                    "opponent": str(accepted_opponent),
                },
                "generatorGate": {
                    "accepted": accepted,
                    "reason": "chunk_gate_passed" if accepted else "chunk_gate_failed",
                },
                "replayWindow": replay_window,
                "replayForTraining": {
                    "eligible": accepted,
                    "reason": "chunk_gate_passed" if accepted else "chunk_gate_failed",
                    "chunk": chunk_label,
                    "valueReplay": str(chunk_dir / "replay" / "self_play.value.jsonl"),
                    "opponentReplay": str(chunk_dir / "replay" / "self_play.opponent.jsonl"),
                    "valueLines": 1,
                    "opponentLines": 1,
                },
            }
        }
        (chunk_dir / "chunk.summary.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
