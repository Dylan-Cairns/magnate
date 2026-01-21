from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.resume_td_loop_selfplay import (
    _build_collect_profiles_from_templates,
    _collect_templates_from_summary,
    _discover_resume_state,
)
from scripts.opponent_pool import PoolCheckpoint


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
    ) -> None:
        chunk_dir = chunks_dir / f"chunk-{chunk_index:03d}"
        replay_dir = chunk_dir / "replay"
        train_dir = chunk_dir / "train"
        profiles_dir = replay_dir / "profiles"
        shard_dir = profiles_dir / "selfplay.shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        train_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    unittest.main()
