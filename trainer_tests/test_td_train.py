from __future__ import annotations

import unittest

import torch

from trainer.td.models import OpponentModel, ValueNet
from trainer.td.train import (
    OpponentTrainConfig,
    TDOpponentTrainer,
    TDTrainConfig,
    TDValueTrainer,
    TD_VALUE_TARGET_MODE_TD_LAMBDA,
    build_value_sequence_index,
    train_opponent_batch,
    train_value_batch,
)
from trainer.td.types import OpponentSample, ValueTransition


class _ConstantValueModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, observation_batch: torch.Tensor) -> torch.Tensor:
        batch_size = int(observation_batch.shape[0])
        return self._value.expand(batch_size)


def _sample_transitions() -> list[ValueTransition]:
    return [
        ValueTransition(
            observation=[0.1, 0.2, 0.3, 0.4],
            reward=0.0,
            done=False,
            next_observation=[0.2, 0.1, 0.4, 0.3],
            player_id="PlayerA",
        ),
        ValueTransition(
            observation=[0.4, 0.2, 0.1, 0.3],
            reward=1.0,
            done=True,
            next_observation=None,
            player_id="PlayerA",
        ),
        ValueTransition(
            observation=[0.3, 0.2, 0.4, 0.1],
            reward=0.0,
            done=False,
            next_observation=[0.1, 0.4, 0.2, 0.3],
            player_id="PlayerB",
        ),
    ]


def _sample_opponent_samples() -> list[OpponentSample]:
    return [
        OpponentSample(
            observation=[0.1, 0.2, 0.3, 0.4],
            action_features=[[1.0, 0.0], [0.0, 1.0]],
            action_index=0,
            player_id="PlayerA",
        ),
        OpponentSample(
            observation=[0.4, 0.3, 0.2, 0.1],
            action_features=[[0.0, 1.0], [1.0, 0.0]],
            action_index=1,
            player_id="PlayerB",
        ),
    ]


class TDTrainTests(unittest.TestCase):
    def test_train_value_batch_uses_one_step_td_zero_target(self) -> None:
        model = _ConstantValueModel(value=0.0)
        target_model = _ConstantValueModel(value=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2, 0.3],
                reward=1.0,
                done=False,
                next_observation=[0.3, 0.2, 0.1],
                player_id="PlayerA",
            ),
            ValueTransition(
                observation=[0.2, 0.3, 0.4],
                reward=-1.0,
                done=True,
                next_observation=None,
                player_id="PlayerB",
            ),
        ]

        _loss, prediction_mean, target_mean = train_value_batch(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            transitions=transitions,
            gamma=0.5,
            max_grad_norm=1.0,
            use_huber_loss=True,
        )

        # TD(0) expected targets:
        # non-terminal: 1.0 + 0.5 * 2.0 = 2.0
        # terminal: -1.0
        self.assertAlmostEqual(prediction_mean, 0.0, places=6)
        self.assertAlmostEqual(target_mean, 0.5, places=6)

    def test_train_value_batch_updates_model(self) -> None:
        model = ValueNet(observation_dim=4, hidden_dim=16)
        target_model = ValueNet(observation_dim=4, hidden_dim=16)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss, prediction_mean, target_mean = train_value_batch(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            transitions=_sample_transitions(),
            gamma=0.99,
            max_grad_norm=1.0,
            use_huber_loss=True,
        )
        after = [parameter.detach() for parameter in model.parameters()]

        self.assertGreaterEqual(loss, 0.0)
        self.assertNotEqual(prediction_mean, target_mean)
        changed = any(not torch.allclose(before[index], after[index]) for index in range(len(before)))
        self.assertTrue(changed)

    def test_td_value_trainer_syncs_on_interval(self) -> None:
        model = ValueNet(observation_dim=4, hidden_dim=16)
        target_model = ValueNet(observation_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = TDValueTrainer(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            config=TDTrainConfig(target_sync_interval=2),
        )

        summary1 = trainer.train_batch(transitions=_sample_transitions())
        summary2 = trainer.train_batch(transitions=_sample_transitions())

        self.assertEqual(summary1.step, 1)
        self.assertFalse(summary1.target_synced)
        self.assertEqual(summary2.step, 2)
        self.assertTrue(summary2.target_synced)

        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, target_model.state_dict()[key]))

    def test_train_opponent_batch_updates_model(self) -> None:
        model = OpponentModel(observation_dim=4, action_feature_dim=2, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss, accuracy = train_opponent_batch(
            model=model,
            optimizer=optimizer,
            samples=_sample_opponent_samples(),
            max_grad_norm=1.0,
        )
        after = [parameter.detach() for parameter in model.parameters()]

        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        changed = any(not torch.allclose(before[index], after[index]) for index in range(len(before)))
        self.assertTrue(changed)

    def test_td_opponent_trainer_tracks_steps(self) -> None:
        model = OpponentModel(observation_dim=4, action_feature_dim=2, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = TDOpponentTrainer(
            model=model,
            optimizer=optimizer,
            config=OpponentTrainConfig(max_grad_norm=1.0),
        )
        summary = trainer.train_batch(samples=_sample_opponent_samples())
        self.assertEqual(summary.step, 1)
        self.assertGreaterEqual(summary.loss, 0.0)

    def test_train_value_batch_rejects_non_terminal_without_next_observation(self) -> None:
        model = ValueNet(observation_dim=4, hidden_dim=16)
        target_model = ValueNet(observation_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2, 0.3, 0.4],
                reward=0.0,
                done=False,
                next_observation=None,
                player_id="PlayerA",
            )
        ]
        with self.assertRaises(ValueError):
            train_value_batch(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                transitions=transitions,
                gamma=0.99,
                max_grad_norm=1.0,
                use_huber_loss=True,
            )

    def test_train_value_batch_rejects_terminal_with_next_observation(self) -> None:
        model = ValueNet(observation_dim=4, hidden_dim=16)
        target_model = ValueNet(observation_dim=4, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2, 0.3, 0.4],
                reward=1.0,
                done=True,
                next_observation=[0.2, 0.3, 0.4, 0.5],
                player_id="PlayerA",
            )
        ]
        with self.assertRaises(ValueError):
            train_value_batch(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                transitions=transitions,
                gamma=0.99,
                max_grad_norm=1.0,
                use_huber_loss=True,
            )

    def test_build_value_sequence_index_rejects_missing_episode_metadata(self) -> None:
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2, 0.3, 0.4],
                reward=0.0,
                done=False,
                next_observation=[0.2, 0.1, 0.4, 0.3],
                player_id="PlayerA",
            )
        ]
        with self.assertRaises(ValueError):
            build_value_sequence_index(transitions=transitions)

    def test_train_value_batch_td_lambda_uses_sequence_targets(self) -> None:
        model = _ConstantValueModel(value=0.0)
        target_model = _ConstantValueModel(value=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
        transitions = [
            ValueTransition(
                observation=[0.1, 0.2, 0.3],
                reward=0.0,
                done=False,
                next_observation=[0.3, 0.2, 0.1],
                player_id="PlayerA",
                episode_id="ep-1",
                timestep=0,
            ),
            ValueTransition(
                observation=[0.3, 0.2, 0.1],
                reward=1.0,
                done=True,
                next_observation=None,
                player_id="PlayerA",
                episode_id="ep-1",
                timestep=1,
            ),
        ]
        sequence_index = build_value_sequence_index(transitions=transitions)
        _loss, prediction_mean, target_mean = train_value_batch(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            transitions=transitions,
            gamma=1.0,
            max_grad_norm=1.0,
            use_huber_loss=True,
            target_mode=TD_VALUE_TARGET_MODE_TD_LAMBDA,
            td_lambda=1.0,
            sequence_index=sequence_index,
        )
        self.assertAlmostEqual(prediction_mean, 0.0, places=6)
        self.assertAlmostEqual(target_mean, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
