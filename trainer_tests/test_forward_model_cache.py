from __future__ import annotations

import copy
import unittest
from typing import Any, cast

from trainer.bridge_payloads import (
    PlayerId,
    PlayerViewPayload,
    ResourcePoolPayload,
    SerializedStatePayload,
)
from trainer.search.forward_model import BridgeForwardModel, ForwardModelCacheStats
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


def _empty_state(*, turn: int, active_player: PlayerId) -> SerializedStatePayload:
    return {
        "schemaVersion": 1,
        "seed": "forward-model-cache",
        "rngCursor": turn,
        "deck": {"draw": [], "discard": [], "reshuffles": 0},
        "players": [
            {"id": "PlayerA", "hand": [], "crowns": [], "resources": _resource_pool()},
            {"id": "PlayerB", "hand": [], "crowns": [], "resources": _resource_pool()},
        ],
        "activePlayerIndex": 0 if active_player == "PlayerA" else 1,
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
        "cardPlayedThisTurn": False,
        "log": [],
    }


def _empty_view(
    *,
    active_player: PlayerId,
    viewer_id: PlayerId,
    turn: int,
) -> PlayerViewPayload:
    return {
        "viewerId": viewer_id,
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
                "handHidden": viewer_id != "PlayerA",
            },
            {
                "id": "PlayerB",
                "crowns": [],
                "resources": _resource_pool(),
                "hand": [],
                "handCount": 0,
                "handHidden": viewer_id != "PlayerB",
            },
        ],
        "deck": {"drawCount": 0, "discard": [], "reshuffles": 0},
        "cardPlayedThisTurn": False,
        "log": [],
    }


def _active_player_id(state: SerializedStatePayload) -> PlayerId:
    return "PlayerA" if state["activePlayerIndex"] == 0 else "PlayerB"


class _FakeEnv:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.legal_actions_calls = 0
        self.observation_calls = 0
        self.step_calls = 0
        self.observation_viewers: list[PlayerId | None] = []
        self._current_state: SerializedStatePayload | None = None

    def reset(
        self,
        *,
        serialized_state: SerializedStatePayload | None = None,
        skip_advance_to_decision: bool = False,
        **_kwargs: object,
    ) -> StateResult:
        if serialized_state is None:
            raise AssertionError("reset requires serialized_state in the cache tests.")
        if not skip_advance_to_decision:
            raise AssertionError("cache tests expect skip_advance_to_decision=True.")
        self.reset_calls += 1
        self._current_state = copy.deepcopy(serialized_state)
        active_player = _active_player_id(serialized_state)
        return StateResult(
            state=copy.deepcopy(self._current_state),
            view=_empty_view(
                active_player=active_player,
                viewer_id=active_player,
                turn=serialized_state["turn"],
            ),
            terminal=False,
        )

    def legal_actions(self) -> LegalActionsResult:
        if self._current_state is None:
            raise AssertionError("legal_actions called before reset.")
        self.legal_actions_calls += 1
        return LegalActionsResult(
            actions=[
                KeyedAction(
                    action_id="end-turn",
                    action_key="a0",
                    action={"type": "end-turn"},
                )
            ],
            active_player_id=_active_player_id(self._current_state),
            phase="ActionWindow",
        )

    def observation(
        self,
        *,
        viewer_id: PlayerId | None = None,
        include_legal_action_mask: bool = False,
    ) -> ObservationResult:
        if self._current_state is None:
            raise AssertionError("observation called before reset.")
        if include_legal_action_mask:
            raise AssertionError("cache tests expect include_legal_action_mask=False.")
        self.observation_calls += 1
        self.observation_viewers.append(viewer_id)
        active_player = _active_player_id(self._current_state)
        resolved_viewer = active_player if viewer_id is None else viewer_id
        return ObservationResult(
            view=_empty_view(
                active_player=active_player,
                viewer_id=resolved_viewer,
                turn=self._current_state["turn"],
            ),
            legal_action_mask=None,
        )

    def step(self, *, action_key: str | None = None, **_kwargs: object) -> StateResult:
        if self._current_state is None:
            raise AssertionError("step called before reset.")
        if action_key is None:
            raise AssertionError("step requires an action_key in the cache tests.")
        self.step_calls += 1
        next_active = "PlayerB" if _active_player_id(self._current_state) == "PlayerA" else "PlayerA"
        turn_delta = 1 if action_key == "a0" else 2
        next_state = _empty_state(turn=self._current_state["turn"] + turn_delta, active_player=next_active)
        self._current_state = copy.deepcopy(next_state)
        return StateResult(
            state=copy.deepcopy(next_state),
            view=_empty_view(
                active_player=next_active,
                viewer_id=next_active,
                turn=next_state["turn"],
            ),
            terminal=False,
        )


class ForwardModelCacheTests(unittest.TestCase):
    def _model(
        self,
        *,
        transition_cache_limit: int = 0,
        legal_actions_cache_limit: int = 0,
        observation_cache_limit: int = 0,
    ) -> tuple[BridgeForwardModel, _FakeEnv]:
        model = BridgeForwardModel(
            transition_cache_limit=transition_cache_limit,
            legal_actions_cache_limit=legal_actions_cache_limit,
            observation_cache_limit=observation_cache_limit,
        )
        fake_env = _FakeEnv()
        model._sim_env = cast(Any, fake_env)
        return model, fake_env

    def test_transition_cached_reuses_result_and_preserves_mutation_isolation(self) -> None:
        model, fake_env = self._model(transition_cache_limit=2)
        state = _empty_state(turn=0, active_player="PlayerA")

        first = model.transition_cached(state, "a0")
        first.state["turn"] = 999
        first.view["turn"] = 999
        second = model.transition_cached(state, "a0")

        self.assertEqual(fake_env.reset_calls, 1)
        self.assertEqual(fake_env.step_calls, 1)
        self.assertEqual(second.state["turn"], 1)
        self.assertEqual(second.view["turn"], 1)
        self.assertEqual(
            model.cache_stats(),
            ForwardModelCacheStats(
                transition_hits=1,
                transition_misses=1,
                legal_actions_hits=0,
                legal_actions_misses=0,
                observation_hits=0,
                observation_misses=0,
                transition_entries=1,
                legal_actions_entries=0,
                observation_entries=0,
            ),
        )

    def test_step_state_cached_wraps_transition_cache(self) -> None:
        model, fake_env = self._model(transition_cache_limit=1)
        state = _empty_state(turn=2, active_player="PlayerB")

        first = model.step_state_cached(state, "a0")
        second = model.step_state_cached(state, "a0")

        self.assertEqual(first["turn"], 3)
        self.assertEqual(second["turn"], 3)
        self.assertEqual(fake_env.reset_calls, 1)
        self.assertEqual(fake_env.step_calls, 1)
        self.assertEqual(model.cache_stats().transition_hits, 1)
        self.assertEqual(model.cache_stats().transition_misses, 1)

    def test_legal_actions_cached_reuses_result_and_preserves_mutation_isolation(self) -> None:
        model, fake_env = self._model(legal_actions_cache_limit=2)
        state = _empty_state(turn=0, active_player="PlayerA")

        first = model.legal_actions_cached(state)
        cast(dict[str, object], first.actions[0].action)["type"] = "sell-card"
        second = model.legal_actions_cached(state)

        self.assertEqual(fake_env.reset_calls, 1)
        self.assertEqual(fake_env.legal_actions_calls, 1)
        self.assertEqual(second.actions[0].action["type"], "end-turn")
        self.assertEqual(model.cache_stats().legal_actions_hits, 1)
        self.assertEqual(model.cache_stats().legal_actions_misses, 1)

    def test_observation_cached_keys_by_viewer_and_preserves_mutation_isolation(self) -> None:
        model, fake_env = self._model(observation_cache_limit=2)
        state = _empty_state(turn=4, active_player="PlayerA")

        first_a = model.observation_cached(state, viewer_id="PlayerA")
        first_a.view["turn"] = 999
        second_a = model.observation_cached(state, viewer_id="PlayerA")
        player_b = model.observation_cached(state, viewer_id="PlayerB")

        self.assertEqual(second_a.view["turn"], 4)
        self.assertEqual(player_b.view["viewerId"], "PlayerB")
        self.assertEqual(fake_env.reset_calls, 2)
        self.assertEqual(fake_env.observation_calls, 2)
        self.assertEqual(fake_env.observation_viewers, ["PlayerA", "PlayerB"])
        self.assertEqual(model.cache_stats().observation_hits, 1)
        self.assertEqual(model.cache_stats().observation_misses, 2)

    def test_transition_cache_evicts_oldest_entry(self) -> None:
        model, fake_env = self._model(transition_cache_limit=1)
        state = _empty_state(turn=1, active_player="PlayerA")

        model.transition_cached(state, "a0")
        model.transition_cached(state, "a0")
        model.transition_cached(state, "a1")
        model.transition_cached(state, "a0")

        self.assertEqual(fake_env.reset_calls, 3)
        self.assertEqual(fake_env.step_calls, 3)
        self.assertEqual(model.cache_stats().transition_hits, 1)
        self.assertEqual(model.cache_stats().transition_misses, 3)
        self.assertEqual(len(model.transition_cache), 1)

    def test_clear_caches_resets_entries_and_stats(self) -> None:
        model, _fake_env = self._model(
            transition_cache_limit=2,
            legal_actions_cache_limit=2,
            observation_cache_limit=2,
        )
        state = _empty_state(turn=0, active_player="PlayerA")

        model.transition_cached(state, "a0")
        model.legal_actions_cached(state)
        model.observation_cached(state, viewer_id="PlayerA")
        self.assertGreater(model.cache_stats().transition_entries, 0)
        self.assertGreater(model.cache_stats().legal_actions_entries, 0)
        self.assertGreater(model.cache_stats().observation_entries, 0)

        model.clear_caches()

        self.assertEqual(
            model.cache_stats(),
            ForwardModelCacheStats(
                transition_hits=0,
                transition_misses=0,
                legal_actions_hits=0,
                legal_actions_misses=0,
                observation_hits=0,
                observation_misses=0,
                transition_entries=0,
                legal_actions_entries=0,
                observation_entries=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
