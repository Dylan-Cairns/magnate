from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from trainer.training import read_samples_jsonl, write_samples_jsonl
from trainer.types import DecisionSample


class TrainingIoTests(unittest.TestCase):
    def test_write_then_read_round_trip(self) -> None:
        samples = [
            DecisionSample(
                seed="s-1",
                turn=2,
                phase="ActionWindow",
                active_player_id="PlayerA",
                action_key="a1",
                action_id="trade",
                action_index=0,
                observation=[0.1, 0.2, 0.3],
                action_features=[[0.0, 1.0], [1.0, 0.0]],
                winner="Draw",
                reward=0.0,
            ),
            DecisionSample(
                seed="s-2",
                turn=3,
                phase="CollectIncome",
                active_player_id="PlayerB",
                action_key="a2",
                action_id="choose-income-suit",
                action_index=1,
                observation=[0.4, 0.5, 0.6],
                action_features=[[0.2, 0.8], [0.6, 0.4]],
                winner="PlayerA",
                reward=-1.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "samples.jsonl"
            write_samples_jsonl(samples, path)
            loaded = read_samples_jsonl(path)

        self.assertEqual(len(loaded), len(samples))
        self.assertEqual(loaded, samples)


if __name__ == "__main__":
    unittest.main()
