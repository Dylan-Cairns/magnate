from __future__ import annotations

import random
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch
from trainer.bridge_payloads import (
    GameActionPayload,
    GamePhase,
    PlayerId,
    PlayerViewPayload,
    ResourcePoolPayload,
    SerializedStatePayload,
)
from trainer.encoding import ACTION_FEATURE_DIM, OBSERVATION_DIM
from trainer.policies import (
    DeterminizedSearchPolicy,
    SearchConfig,
    TDDeterminizedSearchPolicy,
    TDSearchPolicyConfig,
    TDValuePolicy,
    TDValuePolicyConfig,
    policy_from_name,
)
from trainer.search import BridgeForwardModel
from trainer.td.checkpoint import save_opponent_checkpoint, save_value_checkpoint
from trainer.td.models import OpponentModel, ValueNet
from trainer.types import KeyedAction, LegalActionsResult, ObservationResult, StateResult


def _resource_pool() -> ResourcePoolPayload:
    return {
        "Moons": 0,
        "Suns": 0,
        "Waves": 0,
        "Leaves": 0,
        "Wyrms": 0,
        "Knots": 0,
    }


def _empty_state(
    *,
    turn: int,
    phase: GamePhase,
    active_player: PlayerId,
) -> SerializedStatePayload:
    return {
        "schemaVersion": 1,
        "seed": "test-seed",
        "rngCursor": turn,
        "deck": {"draw": [], "discard": [], "reshuffles": 0},
        "players": [
            {"id": "PlayerA", "hand": [], "crowns": [], "resources": _resource_pool()},
            {"id": "PlayerB", "hand": [], "crowns": [], "resources": _resource_pool()},
        ],
        "activePlayerIndex": 0 if active_player == "PlayerA" else 1,
        "turn": turn,
        "phase": phase,
        "districts": [
            {
                "id": "D0",
                "markerSuitMask": [],
                "stacks": {
                    "PlayerA": {"developed": []},
                    "PlayerB": {"developed": []},
                },
            }
        ],
        "cardPlayedThisTurn": False,
        "log": [],
    }


def _empty_view(
    *,
    active_player: PlayerId,
    turn: int,
    viewer_id: PlayerId | None = None,
) -> PlayerViewPayload:
    resolved_viewer = active_player if viewer_id is None else viewer_id
    return {
        "viewerId": resolved_viewer,
        "activePlayerId": active_player,
        "turn": turn,
        "phase": "ActionWindow",
        "districts": [
            {
                "id": "D0",
                "markerSuitMask": [],
                "stacks": {
                    "PlayerA": {"developed": []},
                    "PlayerB": {"developed": []},
                },
            }
        ],
        "players": [
            {
                "id": "PlayerA",
                "crowns": [],
                "resources": _resource_pool(),
                "hand": [],
                "handCount": 0,
                "handHidden": resolved_viewer != "PlayerA",
            },
            {
                "id": "PlayerB",
                "crowns": [],
                "resources": _resource_pool(),
                "hand": [],
                "handCount": 0,
                "handHidden": resolved_viewer != "PlayerB",
            },
        ],
        "deck": {"drawCount": 0, "discard": [], "reshuffles": 0},
        "cardPlayedThisTurn": False,
        "log": [],
    }


def _end_turn_action() -> GameActionPayload:
    return {"type": "end-turn"}


class _ConstantValueModel:
    def __init__(self, value: float) -> None:
        self._value = value

    def __call__(self, _observation) -> torch.Tensor:
        return torch.tensor([self._value], dtype=torch.float32)


class _StubForwardModel:
    def __init__(
        self,
        *,
        post_root_active: PlayerId,
        post_rollout_active: PlayerId | None = None,
    ) -> None:
        self._post_root_active: PlayerId = post_root_active
        self._post_rollout_active: PlayerId = (
            post_rollout_active if post_rollout_active is not None else post_root_active
        )
        self._step_calls = 0

    def reset_state(self, state: SerializedStatePayload) -> StateResult:
        self._step_calls = 0
        return StateResult(
            state=state,
            view=_empty_view(active_player="PlayerA", turn=state["turn"]),
            terminal=False,
        )

    def step(self, _action_key: str) -> StateResult:
        self._step_calls += 1
        if self._step_calls == 1:
            return StateResult(
                state=_empty_state(
                    turn=1,
                    phase="ActionWindow",
                    active_player=self._post_root_active,
                ),
                view=_empty_view(active_player=self._post_root_active, turn=1),
                terminal=False,
            )
        return StateResult(
            state=_empty_state(
                turn=2,
                phase="ActionWindow",
                active_player=self._post_rollout_active,
            ),
            view=_empty_view(active_player=self._post_rollout_active, turn=2),
            terminal=False,
        )

    def legal_actions(self) -> LegalActionsResult:
        return LegalActionsResult(
            actions=[KeyedAction(action_id="end-turn", action_key="a0", action=_end_turn_action())],
            active_player_id="PlayerA",
            phase="ActionWindow",
        )

    def close(self) -> None:
        return None


class _CachedOnlyForwardModel:
    def __init__(self, *, post_root_active: PlayerId, post_rollout_active: PlayerId) -> None:
        self.transition_cached_calls = 0
        self.legal_actions_cached_calls = 0
        self.observation_cached_calls = 0
        self._post_root_active = post_root_active
        self._post_rollout_active = post_rollout_active

    def reset_state(self, _state: SerializedStatePayload) -> StateResult:
        raise AssertionError("reset_state should not be used by cached search rollout.")

    def step(self, _action_key: str) -> StateResult:
        raise AssertionError("step should not be used by cached search rollout.")

    def legal_actions(self) -> LegalActionsResult:
        raise AssertionError("legal_actions should not be used by cached search rollout.")

    def observation(self, viewer_id: PlayerId | None = None) -> ObservationResult:
        raise AssertionError("observation should not be used by cached search rollout.")

    def transition_cached(self, state: SerializedStatePayload, action_key: str) -> StateResult:
        self.transition_cached_calls += 1
        if action_key == "root":
            next_active = cast(PlayerId, self._post_root_active)
            return StateResult(
                state=_empty_state(
                    turn=state["turn"] + 1,
                    phase="ActionWindow",
                    active_player=next_active,
                ),
                view=_empty_view(active_player=next_active, turn=state["turn"] + 1),
                terminal=False,
            )
        next_active = cast(PlayerId, self._post_rollout_active)
        return StateResult(
            state=_empty_state(
                turn=state["turn"] + 2,
                phase="ActionWindow",
                active_player=next_active,
            ),
            view=_empty_view(active_player=next_active, turn=state["turn"] + 2),
            terminal=False,
        )

    def legal_actions_cached(self, _state: SerializedStatePayload) -> LegalActionsResult:
        self.legal_actions_cached_calls += 1
        return LegalActionsResult(
            actions=[KeyedAction(action_id="end-turn", action_key="a0", action=_end_turn_action())],
            active_player_id="PlayerA",
            phase="ActionWindow",
        )

    def observation_cached(
        self,
        _state: SerializedStatePayload,
        viewer_id: PlayerId | None = None,
    ) -> ObservationResult:
        self.observation_cached_calls += 1
        resolved_viewer = "PlayerA" if viewer_id is None else viewer_id
        return ObservationResult(
            view=_empty_view(active_player="PlayerA", turn=3, viewer_id=resolved_viewer),
            legal_action_mask=None,
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

    def test_search_policy_applies_cache_limits_to_forward_model(self) -> None:
        policy = DeterminizedSearchPolicy(
            config=SearchConfig(
                worlds=1,
                rollouts=1,
                depth=1,
                max_root_actions=1,
                rollout_epsilon=0.0,
                transition_cache_limit=11,
                legal_actions_cache_limit=7,
                observation_cache_limit=5,
            )
        )
        try:
            self.assertEqual(policy._forward_model.transition_cache_limit, 11)
            self.assertEqual(policy._forward_model.legal_actions_cache_limit, 7)
            self.assertEqual(policy._forward_model.observation_cache_limit, 5)
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

    def test_td_value_policy_applies_cache_limits_to_forward_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=32),
                output_path=checkpoint_path,
            )
            policy = TDValuePolicy(
                config=TDValuePolicyConfig(
                    checkpoint_path=checkpoint_path,
                    worlds=2,
                    transition_cache_limit=13,
                    legal_actions_cache_limit=9,
                    observation_cache_limit=4,
                )
            )
            try:
                self.assertEqual(policy._forward_model.transition_cache_limit, 13)
                self.assertEqual(policy._forward_model.legal_actions_cache_limit, 9)
                self.assertEqual(policy._forward_model.observation_cache_limit, 4)
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

    def test_td_search_policy_applies_cache_limits_to_forward_model(self) -> None:
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
            policy = TDDeterminizedSearchPolicy(
                config=TDSearchPolicyConfig(
                    value_checkpoint_path=value_path,
                    opponent_checkpoint_path=opponent_path,
                    worlds=1,
                    rollouts=1,
                    depth=1,
                    max_root_actions=1,
                    rollout_epsilon=0.0,
                    transition_cache_limit=17,
                    legal_actions_cache_limit=8,
                    observation_cache_limit=6,
                )
            )
            try:
                self.assertEqual(policy._forward_model.transition_cache_limit, 17)
                self.assertEqual(policy._forward_model.legal_actions_cache_limit, 8)
                self.assertEqual(policy._forward_model.observation_cache_limit, 6)
            finally:
                policy.close()

    def test_td_search_config_requires_opponent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            value_path = Path(tmp_dir) / "value.pt"
            save_value_checkpoint(
                model=ValueNet(observation_dim=OBSERVATION_DIM, hidden_dim=16),
                output_path=value_path,
            )
            constructor = cast(Callable[..., TDSearchPolicyConfig], TDSearchPolicyConfig)
            with self.assertRaises(TypeError):
                constructor(
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
    @patch("trainer.search_policy.value_from_player_view", return_value=0.25)
    def test_search_rollout_uses_cached_state_queries(self, _mock_value_from_player_view) -> None:
        policy = DeterminizedSearchPolicy(
            config=SearchConfig(
                worlds=1,
                rollouts=1,
                depth=1,
                max_root_actions=1,
                rollout_epsilon=0.0,
            )
        )
        try:
            cached_forward_model = _CachedOnlyForwardModel(
                post_root_active="PlayerB",
                post_rollout_active="PlayerA",
            )
            policy._forward_model = cast(BridgeForwardModel, cached_forward_model)
            score = policy._run_rollout(
                world_state=_empty_state(turn=0, phase="ActionWindow", active_player="PlayerA"),
                root_player="PlayerA",
                root_action_key="root",
                rng=random.Random(0),
            )
            self.assertAlmostEqual(score, 0.25)
            self.assertEqual(cached_forward_model.transition_cached_calls, 2)
            self.assertEqual(cached_forward_model.legal_actions_cached_calls, 1)
            self.assertEqual(cached_forward_model.observation_cached_calls, 1)
        finally:
            policy.close()

    @patch("trainer.value_policy.encode_observation", return_value=[0.0] * OBSERVATION_DIM)
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
                policy._forward_model = cast(
                    BridgeForwardModel,
                    _StubForwardModel(post_root_active="PlayerB"),
                )
                policy._model = cast(ValueNet, _ConstantValueModel(0.4))
                score_opponent_active = policy._score_action_world(
                    world_state=_empty_state(turn=0, phase="ActionWindow", active_player="PlayerA"),
                    action_key="a0",
                    root_player="PlayerA",
                )
                self.assertAlmostEqual(score_opponent_active, -0.4)

                policy._forward_model = cast(
                    BridgeForwardModel,
                    _StubForwardModel(post_root_active="PlayerA"),
                )
                score_root_active = policy._score_action_world(
                    world_state=_empty_state(turn=0, phase="ActionWindow", active_player="PlayerA"),
                    action_key="a0",
                    root_player="PlayerA",
                )
                self.assertAlmostEqual(score_root_active, 0.4)
            finally:
                policy.close()

    @patch("trainer.search_policy.encode_observation", return_value=[0.0] * OBSERVATION_DIM)
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
                policy._forward_model = cast(
                    BridgeForwardModel,
                    _StubForwardModel(
                        post_root_active="PlayerB",
                        post_rollout_active="PlayerB",
                    ),
                )
                policy._value_model = cast(ValueNet, _ConstantValueModel(0.6))
                with patch.object(policy, "_opponent_rollout_action_key", return_value="a0"):
                    score = policy._run_rollout(
                        world_state=_empty_state(turn=0, phase="ActionWindow", active_player="PlayerA"),
                        root_player="PlayerA",
                        root_action_key="root",
                        rng=random.Random(0),
                    )
                self.assertAlmostEqual(score, -0.6)
            finally:
                policy.close()


if __name__ == "__main__":
    unittest.main()
