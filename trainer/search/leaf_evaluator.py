from __future__ import annotations

import json
import math
from collections import OrderedDict
from typing import Any, Mapping, Protocol

from trainer.encoding import _card_rank
from trainer.search.forward_model import BridgeForwardModel
from trainer.types import PlayerId


class GuidanceValueModel(Protocol):
    def value_from_view(self, view: Mapping[str, Any]) -> float: ...


class LeafEvaluator:
    def __init__(
        self,
        *,
        forward_model: BridgeForwardModel,
        guidance_model: GuidanceValueModel | None,
        value_cache_limit: int = 0,
    ) -> None:
        self._forward_model = forward_model
        self._guidance_model = guidance_model
        self._value_cache_limit = max(0, value_cache_limit)
        self._value_cache: OrderedDict[tuple[str, PlayerId], float] = OrderedDict()

    @property
    def value_cache(self) -> OrderedDict[tuple[str, PlayerId], float]:
        return self._value_cache

    def clear(self) -> None:
        self._value_cache.clear()

    def value(self, state: Mapping[str, Any], root_player: PlayerId) -> float:
        if self._guidance_model is None:
            return value_from_serialized_state(state, root_player)

        if self._value_cache_limit > 0:
            state_key = _state_cache_key(state)
            cache_key = (state_key, root_player)
            cached = self._value_cache.get(cache_key)
            if cached is not None:
                self._value_cache.move_to_end(cache_key)
                return cached

        step_result = self._forward_model.reset_state(state)
        if step_result.terminal:
            value = terminal_value(step_result.state, root_player)
        else:
            active_player = active_player_id(step_result.view)
            value = self._guidance_model.value_from_view(step_result.view)
            if active_player != root_player:
                value = -value
            value = max(-1.0, min(1.0, value))

        if self._value_cache_limit > 0:
            state_key = _state_cache_key(state)
            cache_key = (state_key, root_player)
            self._value_cache[cache_key] = value
            self._value_cache.move_to_end(cache_key)
            while len(self._value_cache) > self._value_cache_limit:
                self._value_cache.popitem(last=False)
        return value


def terminal_value(state: Mapping[str, Any], root_player: PlayerId) -> float:
    final_score = state.get("finalScore")
    if not isinstance(final_score, dict):
        return 0.0
    winner = final_score.get("winner")
    if winner == "Draw":
        return 0.0
    if winner == root_player:
        return 1.0
    if winner in ("PlayerA", "PlayerB"):
        return -1.0
    return 0.0


def is_terminal_state(state: Mapping[str, Any]) -> bool:
    phase = state.get("phase")
    if phase == "GameOver":
        return True
    return isinstance(state.get("finalScore"), dict)


def state_active_player_id(state: Mapping[str, Any]) -> PlayerId | None:
    if is_terminal_state(state):
        return None
    players = state.get("players")
    if not isinstance(players, list):
        raise ValueError("Serialized state is missing players list.")
    active_index = as_int(state.get("activePlayerIndex"))
    if active_index < 0 or active_index >= len(players):
        raise ValueError(f"Serialized state activePlayerIndex out of range: {active_index}")
    active_player = players[active_index]
    if not isinstance(active_player, dict):
        raise ValueError("Serialized state active player is not an object.")
    player_id = active_player.get("id")
    if player_id not in ("PlayerA", "PlayerB"):
        raise ValueError(f"Serialized state has invalid active player id: {player_id!r}")
    return player_id


def active_player_id(view: Mapping[str, Any]) -> PlayerId:
    value = view.get("activePlayerId")
    if value not in ("PlayerA", "PlayerB"):
        raise ValueError(f"Invalid activePlayerId in view: {value!r}")
    return value


def value_from_player_view(view: Mapping[str, Any], root_player: PlayerId) -> float:
    opponent = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players_by_id = player_views_by_id(view)
    root_state = players_by_id[root_player]
    opponent_state = players_by_id[opponent]

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = as_int(root_state.get("handCount")) - as_int(opponent_state.get("handCount"))

    districts = view.get("districts")
    district_count = len(districts) if isinstance(districts, list) else 0
    district_lead = 0.0
    rank_diff = 0.0
    progress_diff = 0.0
    if isinstance(districts, list):
        for district in districts:
            if not isinstance(district, dict):
                continue
            stacks = district.get("stacks")
            if not isinstance(stacks, dict):
                continue
            root_stack = as_mapping(stacks.get(root_player))
            opponent_stack = as_mapping(stacks.get(opponent))
            root_rank, root_progress = stack_score(root_stack)
            opponent_rank, opponent_progress = stack_score(opponent_stack)
            rank_diff += root_rank - opponent_rank
            progress_diff += root_progress - opponent_progress
            if root_rank > opponent_rank:
                district_lead += 1.0
            elif root_rank < opponent_rank:
                district_lead -= 1.0

    district_term = district_lead / float(max(1, district_count))
    rank_term = math.tanh(rank_diff / 18.0)
    progress_term = math.tanh(progress_diff / 8.0)
    resource_term = math.tanh((resource_root - resource_opponent) / 10.0)
    hand_term = math.tanh(hand_diff / 4.0)

    score = (
        (0.55 * district_term)
        + (0.2 * rank_term)
        + (0.1 * progress_term)
        + (0.1 * resource_term)
        + (0.05 * hand_term)
    )
    return max(-1.0, min(1.0, score))


def value_from_serialized_state(state: Mapping[str, Any], root_player: PlayerId) -> float:
    if is_terminal_state(state):
        return terminal_value(state, root_player)

    opponent = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players = state.get("players")
    if not isinstance(players, list):
        return 0.0

    root_state = None
    opponent_state = None
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = player.get("id")
        if player_id == root_player:
            root_state = player
        elif player_id == opponent:
            opponent_state = player
    if root_state is None or opponent_state is None:
        return 0.0

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = len(as_card_list(root_state.get("hand"))) - len(as_card_list(opponent_state.get("hand")))
    resource_diff = resource_root - resource_opponent

    districts = state.get("districts")
    district_count = len(districts) if isinstance(districts, list) else 0
    district_point_diff = 0.0
    rank_total_diff = 0.0
    deed_completion_diff = 0.0
    if isinstance(districts, list):
        for district in districts:
            if not isinstance(district, dict):
                continue
            stacks = district.get("stacks")
            if not isinstance(stacks, dict):
                continue
            root_stack = stacks.get(root_player)
            opponent_stack = stacks.get(opponent)
            if not isinstance(root_stack, dict) or not isinstance(opponent_stack, dict):
                continue

            root_developed_rank = developed_rank_total(root_stack)
            opponent_developed_rank = developed_rank_total(opponent_stack)
            rank_total_diff += root_developed_rank - opponent_developed_rank

            root_district_strength = root_developed_rank + ace_pressure_proxy(root_stack)
            opponent_district_strength = opponent_developed_rank + ace_pressure_proxy(opponent_stack)
            if root_district_strength > opponent_district_strength:
                district_point_diff += 1.0
            elif root_district_strength < opponent_district_strength:
                district_point_diff -= 1.0

            deed_completion_diff += (
                deed_completion_ratio(root_stack) - deed_completion_ratio(opponent_stack)
            )

    deck = state.get("deck")
    reshuffles = as_int(as_mapping(deck).get("reshuffles")) if isinstance(deck, dict) else 0
    final_turns_remaining = as_int(state.get("finalTurnsRemaining"))
    endgame = reshuffles >= 2 or final_turns_remaining > 0

    district_term = district_point_diff / float(max(1, district_count))
    rank_term = math.tanh(rank_total_diff / 16.0)
    progress_term = math.tanh(deed_completion_diff / 2.5)
    resource_term = math.tanh(resource_diff / 8.0)
    hand_term = math.tanh(hand_diff / 3.0)

    district_weight = 0.72 if endgame else 0.6
    rank_weight = 0.16 if endgame else 0.2
    progress_weight = 0.05 if endgame else 0.12
    resource_weight = 0.05
    hand_weight = 0.02

    score = (
        (district_weight * district_term)
        + (rank_weight * rank_term)
        + (progress_weight * progress_term)
        + (resource_weight * resource_term)
        + (hand_weight * hand_term)
    )
    return max(-1.0, min(1.0, score))


def player_views_by_id(view: Mapping[str, Any]) -> dict[PlayerId, dict[str, Any]]:
    players = view.get("players")
    if not isinstance(players, list):
        raise ValueError("View payload is missing players list.")

    out: dict[PlayerId, dict[str, Any]] = {}
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = player.get("id")
        if player_id in ("PlayerA", "PlayerB"):
            out[player_id] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def stack_score(stack: Mapping[str, Any]) -> tuple[float, float]:
    rank_total = 0.0
    progress_total = 0.0
    for card_id in as_card_list(stack.get("developed")):
        rank_total += float(_card_rank(card_id))

    deed = stack.get("deed")
    if isinstance(deed, dict):
        rank_total += float(_card_rank(str(deed.get("cardId", ""))))
        progress = as_int(deed.get("progress"))
        progress_total += float(progress)
        rank_total += 0.35 * float(progress)
    return rank_total, progress_total


def developed_rank_total(stack: Mapping[str, Any]) -> float:
    total = 0.0
    for card_id in as_card_list(stack.get("developed")):
        total += float(_card_rank(card_id))
    return total


def ace_pressure_proxy(stack: Mapping[str, Any]) -> float:
    developed = as_card_list(stack.get("developed"))
    ace_count = sum(1 for card_id in developed if _card_rank(card_id) == 1)
    if ace_count <= 0:
        return 0.0
    return 0.35 * float(ace_count * max(0, len(developed) - 1))


def deed_completion_ratio(stack: Mapping[str, Any]) -> float:
    deed = stack.get("deed")
    if not isinstance(deed, dict):
        return 0.0
    card_id = str(deed.get("cardId", ""))
    rank = _card_rank(card_id)
    if rank <= 0:
        return 0.0
    target = 3 if rank == 1 else rank
    progress = as_int(deed.get("progress"))
    clipped = min(max(progress, 0), target)
    return clipped / float(target)


def resource_total(player_state: Mapping[str, Any]) -> int:
    resources = player_state.get("resources")
    if not isinstance(resources, dict):
        return 0
    total = 0
    for suit in ("Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots"):
        total += as_int(resources.get(suit))
    return total


def as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object mapping, got {type(value).__name__}.")


def as_card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            out.append(entry)
    return out


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _state_cache_key(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
