from __future__ import annotations

import random
import unittest

from trainer.td.replay import OpponentReplayBuffer, ValueReplayBuffer
from trainer.td.types import OpponentSample, ValueTransition


class TDReplayBufferTests(unittest.TestCase):
    def test_value_replay_respects_capacity(self) -> None:
        replay = ValueReplayBuffer(capacity=2)
        replay.add(
            ValueTransition(
                observation=[1.0],
                reward=0.0,
                done=False,
                next_observation=[2.0],
                player_id="PlayerA",
            )
        )
        replay.add(
            ValueTransition(
                observation=[3.0],
                reward=0.0,
                done=False,
                next_observation=[4.0],
                player_id="PlayerA",
            )
        )
        replay.add(
            ValueTransition(
                observation=[5.0],
                reward=1.0,
                done=True,
                next_observation=None,
                player_id="PlayerA",
            )
        )

        items = replay.as_list()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].observation, [3.0])
        self.assertEqual(items[1].observation, [5.0])

    def test_opponent_replay_sample_returns_full_buffer_when_batch_too_large(self) -> None:
        replay = OpponentReplayBuffer(capacity=4)
        replay.extend(
            [
                OpponentSample(
                    observation=[0.0, 1.0],
                    action_features=[[1.0], [0.0]],
                    action_index=0,
                    player_id="PlayerA",
                ),
                OpponentSample(
                    observation=[1.0, 0.0],
                    action_features=[[0.0], [1.0]],
                    action_index=1,
                    player_id="PlayerB",
                ),
            ]
        )

        sampled = replay.sample(batch_size=8, rng=random.Random(7))
        self.assertEqual(len(sampled), 2)

    def test_sampling_empty_buffer_raises(self) -> None:
        replay = ValueReplayBuffer(capacity=2)
        with self.assertRaises(ValueError):
            replay.sample(batch_size=1, rng=random.Random(0))


if __name__ == "__main__":
    unittest.main()
