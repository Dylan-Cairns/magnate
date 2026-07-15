from __future__ import annotations

import math
import unittest

import torch
from trainer.td.holdout import evaluate_opponent_holdout, evaluate_value_holdout
from trainer.td.models import OpponentModel, ValueNet
from trainer.td.types import OpponentSample, ValueTransition


class TDHoldoutTests(unittest.TestCase):
    def test_value_holdout_uses_discounted_terminal_outcomes(self) -> None:
        model = ValueNet(observation_dim=2, hidden_dim=4)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        transitions = [
            ValueTransition(
                observation=[1.0, 0.0],
                reward=0.0,
                done=False,
                next_observation=[0.0, 1.0],
                player_id="PlayerA",
                episode_id="game-1",
                timestep=0,
            ),
            ValueTransition(
                observation=[0.0, 1.0],
                reward=1.0,
                done=True,
                next_observation=None,
                player_id="PlayerA",
                episode_id="game-1",
                timestep=1,
            ),
            ValueTransition(
                observation=[1.0, 1.0],
                reward=0.0,
                done=False,
                next_observation=[0.0, 0.0],
                player_id="PlayerB",
                episode_id="game-1",
                timestep=0,
            ),
            ValueTransition(
                observation=[0.0, 0.0],
                reward=-1.0,
                done=True,
                next_observation=None,
                player_id="PlayerB",
                episode_id="game-1",
                timestep=1,
            ),
        ]

        metrics = evaluate_value_holdout(
            model=model,
            transitions=transitions,
            gamma=0.5,
            batch_size=2,
        )

        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["sequences"], 2)
        self.assertAlmostEqual(float(metrics["monteCarloMse"]), 0.625)
        self.assertAlmostEqual(float(metrics["monteCarloMae"]), 0.75)
        self.assertAlmostEqual(float(metrics["meanPredictionBias"]), 0.0)

    def test_opponent_holdout_scores_soft_and_selected_targets(self) -> None:
        model = OpponentModel(
            observation_dim=2,
            action_feature_dim=1,
            hidden_dim=4,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        samples = [
            OpponentSample(
                observation=[1.0, 0.0],
                action_features=[[1.0], [0.0]],
                action_index=0,
                action_probs=[1.0, 0.0],
                player_id="PlayerA",
            ),
            OpponentSample(
                observation=[0.0, 1.0],
                action_features=[[0.0], [1.0]],
                action_index=1,
                action_probs=[0.75, 0.25],
                player_id="PlayerB",
            ),
        ]

        metrics = evaluate_opponent_holdout(model=model, samples=samples)

        self.assertEqual(metrics["rows"], 2)
        self.assertAlmostEqual(float(metrics["softTargetCrossEntropy"]), math.log(2.0), places=6)
        self.assertEqual(metrics["teacherTopActionAgreement"], 1.0)
        self.assertEqual(metrics["selectedActionAccuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
