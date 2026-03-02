from __future__ import annotations

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
            ),
            ValueTransition(
                observation=[0.5, 0.6],
                reward=1.0,
                done=True,
                next_observation=None,
                player_id="PlayerB",
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


if __name__ == "__main__":
    unittest.main()
