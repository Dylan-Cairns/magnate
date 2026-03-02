from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trainer.td.io import (
    read_opponent_samples_jsonl,
    read_value_transitions_jsonl,
    write_opponent_samples_jsonl,
    write_value_transitions_jsonl,
)
from trainer.td.types import OpponentSample, ValueTransition


class TDIOTests(unittest.TestCase):
    def test_value_transition_round_trip(self) -> None:
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2],
                reward=0.0,
                done=False,
                next_observation=[0.3, 0.4],
                player_id="PlayerA",
                episode_id="ep-1",
                timestep=0,
            ),
            ValueTransition(
                observation=[0.5, 0.6],
                reward=1.0,
                done=True,
                next_observation=None,
                player_id="PlayerB",
                episode_id="ep-1",
                timestep=0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "value.jsonl"
            write_value_transitions_jsonl(transitions, path)
            loaded = read_value_transitions_jsonl(path)
        self.assertEqual(loaded, transitions)

    def test_opponent_sample_round_trip(self) -> None:
        samples = [
            OpponentSample(
                observation=[0.1, 0.2],
                action_features=[[0.0, 1.0], [1.0, 0.0]],
                action_index=1,
                player_id="PlayerA",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "opponent.jsonl"
            write_opponent_samples_jsonl(samples, path)
            loaded = read_opponent_samples_jsonl(path)
        self.assertEqual(loaded, samples)

    def test_read_value_transition_rejects_done_with_next_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad-value.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "observation": [0.1, 0.2],
                        "reward": 1.0,
                        "done": True,
                        "nextObservation": [0.3, 0.4],
                        "playerId": "PlayerA",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_value_transitions_jsonl(path)

    def test_read_value_transition_rejects_non_terminal_without_next_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad-value.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "observation": [0.1, 0.2],
                        "reward": 0.0,
                        "done": False,
                        "nextObservation": None,
                        "playerId": "PlayerA",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_value_transitions_jsonl(path)

    def test_write_value_transitions_rejects_non_terminal_without_next_observation(self) -> None:
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2],
                reward=0.0,
                done=False,
                next_observation=None,
                player_id="PlayerA",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "value.jsonl"
            with self.assertRaises(ValueError):
                write_value_transitions_jsonl(transitions, path)

    def test_read_value_transition_rejects_episode_step_partial_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad-value.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "observation": [0.1, 0.2],
                        "reward": 0.0,
                        "done": False,
                        "nextObservation": [0.3, 0.4],
                        "playerId": "PlayerA",
                        "episodeId": "ep-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_value_transitions_jsonl(path)

    def test_write_value_transitions_rejects_episode_step_partial_presence(self) -> None:
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2],
                reward=0.0,
                done=False,
                next_observation=[0.3, 0.4],
                player_id="PlayerA",
                episode_id="ep-1",
                timestep=None,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "value.jsonl"
            with self.assertRaises(ValueError):
                write_value_transitions_jsonl(transitions, path)


if __name__ == "__main__":
    unittest.main()
