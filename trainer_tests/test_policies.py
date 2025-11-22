from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from trainer.encoding import OBSERVATION_DIM
from trainer.policies import (
    DeterminizedSearchPolicy,
    SearchConfig,
    TDDeterminizedSearchPolicy,
    TDSearchPolicyConfig,
    TDValuePolicy,
    TDValuePolicyConfig,
    policy_from_name,
)
from trainer.td.checkpoint import save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet
from trainer.encoding import ACTION_FEATURE_DIM
from trainer.td.checkpoint import save_opponent_checkpoint
from trainer.types import KeyedAction, LegalActionsResult, StateResult


class _ConstantValueModel:
    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, _observation) -> torch.Tensor:
        return torch.tensor([self._value], dtype=torch.float32)


class _StubForwardModel:
    def __init__(self, *, post_root_active: str, post_rollout_active: str | None = None) -> None:
        self._post_root_active = post_root_active
        self._post_rollout_active = (
            post_rollout_active if post_rollout_active is not None else post_root_active
        )
        self._step_calls = 0

    def reset_state(self, state) -> StateResult:
        self._step_calls = 0
        return StateResult(
            state=dict(state),
            view={"activePlayerId": "PlayerA"},
            terminal=False,
        )

    def step(self, _action_key: str) -> StateResult:
        self._step_calls += 1
        if self._step_calls == 1:
            return StateResult(
                state={"turn": 1, "phase": "ActionWindow"},
                view={"activePlayerId": self._post_root_active},
                terminal=False,
            )
        return StateResult(
            state={"turn": 2, "phase": "ActionWindow"},
            view={"activePlayerId": self._post_rollout_active},
            terminal=False,
        )

    def legal_actions(self) -> LegalActionsResult:
        return LegalActionsResult(
            actions=[KeyedAction(action_id="end-turn", action_key="a0", action={})],
            active_player_id="PlayerA",
            phase="ActionWindow",
        )

    def close(self) -> None:
        return None


class PolicyFactoryTests(unittest.TestCase):
    def test_policy_factory_creates_random_policy(self) -> None:
        policy = policy_from_name("random")
        self.assertEqual(policy.name, "random")

    def test_policy_factory_creates_heuristic_policy(self) -> None:
        policy = policy_from_name("heuristic")
        self.assertEqual(policy.name, "heuristic")

    def test_policy_factory_creates_search_policy(self) -> None:
        config = SearchConfig(worlds=1, rollouts=1, depth=1, max_root_actions=1, rollout_epsilon=0.0)
        policy = policy_from_name("search", search_config=config)
        try:
            self.assertIsInstance(policy, DeterminizedSearchPolicy)
        finally:
            policy.close()

    def test_policy_factory_rejects_unknown_policy(self) -> None:
        with self.assertRaises(ValueError):
            policy_from_name("legacy")

    def test_policy_factory_creates_td_value_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=checkpoint_path,
            )
            policy = policy_from_name(
                "td-value",
                td_value_config=TDValuePolicyConfig(
                    checkpoint_path=checkpoint_path,
                    worlds=2,
                ),
            )
            try:
                self.assertIsInstance(policy, TDValuePolicy)
            finally:
                policy.close()

    def test_policy_factory_creates_td_search_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=32,
                ),
                output_path=opponent_path,
            )
            policy = policy_from_name(
                "td-search",
                td_search_config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                ),
            )
            try:
                self.assertIsInstance(policy, TDDeterminizedSearchPolicy)
            finally:
                policy.close()

    def test_td_search_config_requires_opponent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            with self.assertRaises(TypeError):
                TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                )

    def test_td_search_config_accepts_opponent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=16,
                ),
                output_path=opponent_path,
            )
            policy = policy_from_name(
                "td-search",
                td_search_config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                ),
            )
            try:
                self.assertIsInstance(policy, TDDeterminizedSearchPolicy)
            finally:
                policy.close()


class TDLeafPerspectiveTests(unittest.TestCase):
    @patch("trainer.policies.encode_observation", return_value=[0.0] * OBSERVATION_DIM)
    def test_td_value_leaf_converts_active_value_to_root_perspective(self, _mock_encode) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=checkpoint_path,
            )
            policy = TDValuePolicy(
                config=TDValuePolicyConfig(
                    checkpoint_path=checkpoint_path,
                    worlds=1,
                )
            )
            try:
                policy._forward_model = _StubForwardModel(post_root_active="PlayerB")
                policy._model = _ConstantValueModel(0.4)
                score_opponent_active = policy._score_action_world(
                    world_state={"turn": 0, "phase": "ActionWindow"},
                    action_key="a0",
                    root_player="PlayerA",
                )
                self.assertAlmostEqual(score_opponent_active, -0.4)

                policy._forward_model = _StubForwardModel(post_root_active="PlayerA")
                score_root_active = policy._score_action_world(
                    world_state={"turn": 0, "phase": "ActionWindow"},
                    action_key="a0",
                    root_player="PlayerA",
                )
                self.assertAlmostEqual(score_root_active, 0.4)
            finally:
                policy.close()

    @patch("trainer.policies.encode_observation", return_value=[0.0] * OBSERVATION_DIM)
    def test_td_search_leaf_converts_active_value_to_root_perspective(self, _mock_encode) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            opponent_path = Path(tmp_dir) / "opponent.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            save_opponent_checkpoint(
                model=OpponentModel(
                    observation_dim=OBSERVATION_DIM,
                    action_feature_dim=ACTION_FEATURE_DIM,
                    hidden_dim=16,
                ),
                output_path=opponent_path,
            )
            policy = TDDeterminizedSearchPolicy(
                config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                )
            )
            try:
                policy._forward_model = _StubForwardModel(
                    post_root_active="PlayerB",
                    post_rollout_active="PlayerB",
                )
                policy._value_model = _ConstantValueModel(0.6)
                policy._opponent_rollout_action_key = lambda **_kwargs: "a0"
                score = policy._run_rollout(
                    world_state={"turn": 0, "phase": "ActionWindow"},
                    root_player="PlayerA",
                    root_action_key="root",
                    rng=random.Random(0),
                )
                self.assertAlmostEqual(score, -0.6)
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
