from __future__ import annotations

import unittest

import torch

from trainer.td.models import ValueNet
from trainer.td.train import TDTrainConfig, TDValueTrainer, train_value_batch
from trainer.td.types import ValueTransition


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


class TDTrainTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
