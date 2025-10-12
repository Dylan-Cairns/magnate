from __future__ import annotations

import random
import unittest

from trainer.behavior_cloning import BehaviorCloningModel
from trainer.bridge_client import BridgeClient
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.env import MagnateBridgeEnv
from trainer.reinforcement import (
    ReinforceConfig,
    apply_reinforce_update,
    fine_tune_with_reinforce,
    sample_opponent_kind,
    softmax_with_temperature,
)


class ReinforcementMathTests(unittest.TestCase):
    def test_positive_advantage_increases_selected_action_score(self) -> None:
        model = BehaviorCloningModel.zeros(observation_dim=1, action_feature_dim=1)
        observation = [1.0]
        action_features = [[1.0], [0.5]]

        probs = softmax_with_temperature(
            model.score_candidates(observation, action_features),
            temperature=1.0,
        )
        apply_reinforce_update(
            model=model,
            observation=observation,
            action_features=action_features,
            chosen_index=0,
            probs=probs,
            advantage=1.0,
            learning_rate=0.5,
            l2=0.0,
        )

        updated_scores = model.score_candidates(observation, action_features)
        self.assertGreater(updated_scores[0], updated_scores[1])

    def test_anchor_regularization_pulls_weights_toward_source(self) -> None:
        model = BehaviorCloningModel.zeros(observation_dim=1, action_feature_dim=1)
        anchor = BehaviorCloningModel.zeros(observation_dim=1, action_feature_dim=1)
        model.action_weights[0] = 2.0
        anchor.action_weights[0] = 1.0

        apply_reinforce_update(
            model=model,
            observation=[1.0],
            action_features=[[1.0], [0.0]],
            chosen_index=0,
            probs=[1.0, 0.0],
            advantage=0.0,
            learning_rate=0.5,
            l2=0.0,
            anchor_model=anchor,
            anchor_coeff=1.0,
        )

        self.assertLess(model.action_weights[0], 2.0)
        self.assertGreater(model.action_weights[0], 1.0)

    def test_single_weight_opponent_mix_is_deterministic(self) -> None:
        rng = random.Random(0)
        kind = sample_opponent_kind(
            ReinforceConfig(
                episodes=1,
                learning_rate=0.01,
                l2=1e-5,
                temperature=1.0,
                seed=0,
                max_decisions_per_game=2000,
                self_play_weight=0.0,
                heuristic_weight=1.0,
                random_weight=0.0,
            ),
            rng=rng,
        )
        self.assertEqual(kind, "heuristic")


class ReinforcementIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_fine_tune_runs_single_episode(self) -> None:
        model = BehaviorCloningModel.zeros(
            observation_dim=OBSERVATION_DIM,
            action_feature_dim=ACTION_FEATURE_DIM,
        )
        summary = fine_tune_with_reinforce(
            env=self.env,
            model=model,
            config=ReinforceConfig(
                episodes=1,
                learning_rate=0.01,
                l2=1e-5,
                temperature=1.0,
                seed=0,
                max_decisions_per_game=2000,
            ),
            seed_prefix="rl-unit",
        )

        self.assertEqual(summary.episodes, 1)
        self.assertEqual(
            summary.winners["PlayerA"] + summary.winners["PlayerB"] + summary.winners["Draw"],
            1,
        )
        self.assertGreater(summary.average_decisions, 0.0)


if __name__ == "__main__":
    unittest.main()
