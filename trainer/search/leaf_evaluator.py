from __future__ import annotations

import json
import math
from collections import OrderedDict
from typing import Any, Dict, Mapping

from trainer.encoding import _card_rank
from trainer.types import PlayerId


class LeafEvaluator:
    def __init__(self, *, value_cache_limit: int = 0) -> None:
        self._value_cache_limit = max(0, value_cache_limit)
        self._value_cache: OrderedDict[tuple[str, PlayerId], float] = OrderedDict()

    @property
    def value_cache(self) -> OrderedDict[tuple[str, PlayerId], float]:
        return self._value_cache

    def clear(self) -> None:
        self._value_cache.clear()

    def value(self, state: Mapping[str, Any], root_player: PlayerId) -> float:
        if self._value_cache_limit > 0:
            state_key = _state_cache_key(state)
            cache_key = (state_key, root_player)
            cached = self._value_cache.get(cache_key)
            if cached is not None:
                self._value_cache.move_to_end(cache_key)
                return cached

        value = value_from_serialized_state(state, root_player)

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
        raise ValueError(
            "Terminal state is missing finalScore. "
            f"turn={state.get('turn')} phase={state.get('phase')!r}"
        )
    winner = final_score.get("winner")
    if winner == "Draw":
        return 0.0
    if winner == root_player:
        return 1.0
    if winner in ("PlayerA", "PlayerB"):
        return -1.0
    raise ValueError(f"Invalid finalScore.winner value: {winner!r}")


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


def active_value_to_root_value(
    *,
    active_value: float,
    active_player: PlayerId,
    root_player: PlayerId,
) -> float:
    if active_player == root_player:
        return active_value
    return -active_value


def value_from_player_view(view: Mapping[str, Any], root_player: PlayerId) -> float:
    opponent = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players_by_id = player_views_by_id(view)
    root_state = players_by_id[root_player]
    opponent_state = players_by_id[opponent]

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = as_int(root_state.get("handCount")) - as_int(opponent_state.get("handCount"))

    districts = view.get("districts")
    if not isinstance(districts, list):
        raise ValueError("View payload is missing districts list.")
    district_count = len(districts)
    district_lead = 0.0
    rank_diff = 0.0
    progress_diff = 0.0
    for district in districts:
        if not isinstance(district, dict):
            raise ValueError(f"District entry must be an object, got {type(district).__name__}.")
        stacks = district.get("stacks")
        if not isinstance(stacks, dict):
            raise ValueError("District payload is missing stacks object.")
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
        raise ValueError("Serialized state is missing players list.")

    root_state = None
    opponent_state = None
    for player in players:
        if not isinstance(player, dict):
            raise ValueError(f"Player entry must be an object, got {type(player).__name__}.")
        player_id = player.get("id")
        if player_id == root_player:
            root_state = player
        elif player_id == opponent:
            opponent_state = player
    if root_state is None or opponent_state is None:
        raise ValueError(
            "Serialized state is missing one or more players required for evaluation. "
            f"root={root_player} opponent={opponent}"
        )

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = len(as_card_list(root_state.get("hand"))) - len(as_card_list(opponent_state.get("hand")))
    resource_diff = resource_root - resource_opponent

    districts = state.get("districts")
    if not isinstance(districts, list):
        raise ValueError("Serialized state is missing districts list.")
    district_count = len(districts)
    district_point_diff = 0.0
    rank_total_diff = 0.0
    deed_completion_diff = 0.0
    for district in districts:
        if not isinstance(district, dict):
            raise ValueError(f"District entry must be an object, got {type(district).__name__}.")
        stacks = district.get("stacks")
        if not isinstance(stacks, dict):
            raise ValueError("District payload is missing stacks object.")
        root_stack = stacks.get(root_player)
        opponent_stack = stacks.get(opponent)
        if not isinstance(root_stack, dict) or not isinstance(opponent_stack, dict):
            raise ValueError("District stacks are missing root/opponent entries.")

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

    district_term = district_point_diff / float(max(1, district_count))
    rank_term = math.tanh(rank_total_diff / 18.0)
    deed_term = math.tanh(deed_completion_diff / 5.0)
    resource_term = math.tanh(resource_diff / 10.0)
    hand_term = math.tanh(hand_diff / 4.0)

    score = (
        (0.55 * district_term)
        + (0.2 * rank_term)
        + (0.1 * deed_term)
        + (0.1 * resource_term)
        + (0.05 * hand_term)
    )
    return max(-1.0, min(1.0, score))


def stack_score(stack: Mapping[str, Any]) -> tuple[float, float]:
    developed = as_card_list(stack.get("developed"))
    developed_rank = sum(_card_rank(card_id) for card_id in developed)

    deed = as_optional_mapping(stack.get("deed"))
    deed_card = str(deed.get("cardId", ""))
    deed_rank = _card_rank(deed_card)
    deed_progress = as_optional_int(deed.get("progress"), default=0)

    progress_ratio = 0.0
    if deed_card and deed_rank > 0:
        progress_ratio = min(1.0, deed_progress / float(deed_rank))

    return developed_rank, progress_ratio


def developed_rank_total(stack: Mapping[str, Any]) -> int:
    return sum(_card_rank(card_id) for card_id in as_card_list(stack.get("developed")))


def ace_pressure_proxy(stack: Mapping[str, Any]) -> int:
    developed = as_card_list(stack.get("developed"))
    non_ace = sum(1 for card_id in developed if _card_rank(card_id) != 1)
    ace_bonus = sum(1 for card_id in developed if _card_rank(card_id) == 1)
    return ace_bonus * non_ace


def deed_completion_ratio(stack: Mapping[str, Any]) -> float:
    deed = as_optional_mapping(stack.get("deed"))
    card_id = str(deed.get("cardId", ""))
    rank = _card_rank(card_id)
    if rank <= 0:
        return 0.0
    progress = as_optional_int(deed.get("progress"), default=0)
    return min(1.0, max(0.0, progress / float(rank)))


def player_views_by_id(view: Mapping[str, Any]) -> Dict[PlayerId, Dict[str, Any]]:
    players = view.get("players")
    if not isinstance(players, list):
        raise ValueError("View payload is missing players list.")

    out: Dict[PlayerId, Dict[str, Any]] = {}
    for player in players:
        if not isinstance(player, dict):
            raise ValueError(f"Player entry must be an object, got {type(player).__name__}.")
        player_id = player.get("id")
        if player_id in ("PlayerA", "PlayerB"):
            out[player_id] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def resource_total(player_state: Mapping[str, Any]) -> int:
    resources = player_state.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("Player payload is missing resources object.")
    total = 0
    for value in resources.values():
        total += as_int(value)
    return total


def as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object mapping, got {type(value).__name__}.")


def as_optional_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return as_mapping(value)


def as_card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list of card ids, got {type(value).__name__}.")
    out: list[str] = []
    for card_id in value:
        if not isinstance(card_id, str):
            raise ValueError(f"Expected card id string, got {type(card_id).__name__}.")
        out.append(card_id)
    return out


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Expected integer value, got bool.")
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Expected numeric value, got {type(value).__name__}.")


def as_optional_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return as_int(value)


def _state_cache_key(state: Mapping[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
