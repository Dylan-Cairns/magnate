from __future__ import annotations

import json
import math
from collections import OrderedDict

from trainer.bridge_payloads import (
    DistrictStackPayload,
    ObservedPlayerPayload,
    PlayerId,
    PlayerStatePayload,
    PlayerViewPayload,
    SerializedStatePayload,
)
from trainer.encoding import SUITS, _card_rank


class LeafEvaluator:
    def __init__(self, *, value_cache_limit: int = 0) -> None:
        self._value_cache_limit = max(0, value_cache_limit)
        self._value_cache: OrderedDict[tuple[str, PlayerId], float] = OrderedDict()

    @property
    def value_cache(self) -> OrderedDict[tuple[str, PlayerId], float]:
        return self._value_cache

    def clear(self) -> None:
        self._value_cache.clear()

    def value(self, state: SerializedStatePayload, root_player: PlayerId) -> float:
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


def terminal_value(state: SerializedStatePayload, root_player: PlayerId) -> float:
    final_score = state.get("finalScore")
    if final_score is None:
        raise ValueError(
            "Terminal state is missing finalScore. "
            f"turn={state['turn']} phase={state['phase']!r}"
        )
    winner = final_score["winner"]
    if winner == "Draw":
        return 0.0
    if winner == root_player:
        return 1.0
    if winner in ("PlayerA", "PlayerB"):
        return -1.0
    raise ValueError(f"Invalid finalScore.winner value: {winner!r}")


def is_terminal_state(state: SerializedStatePayload) -> bool:
    return state["phase"] == "GameOver" or state.get("finalScore") is not None


def state_active_player_id(state: SerializedStatePayload) -> PlayerId | None:
    if is_terminal_state(state):
        return None
    active_index = state["activePlayerIndex"]
    if active_index < 0 or active_index >= len(state["players"]):
        raise ValueError(f"Serialized state activePlayerIndex out of range: {active_index}")
    return state["players"][active_index]["id"]


def active_player_id(view: PlayerViewPayload) -> PlayerId:
    return view["activePlayerId"]


def active_value_to_root_value(
    *,
    active_value: float,
    active_player: PlayerId,
    root_player: PlayerId,
) -> float:
    if active_player == root_player:
        return active_value
    return -active_value


def value_from_player_view(view: PlayerViewPayload, root_player: PlayerId) -> float:
    opponent = _opponent_player_id(root_player)
    players_by_id = player_views_by_id(view)
    root_state = players_by_id[root_player]
    opponent_state = players_by_id[opponent]

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = root_state["handCount"] - opponent_state["handCount"]

    district_count = len(view["districts"])
    district_lead = 0.0
    rank_diff = 0.0
    progress_diff = 0.0
    for district in view["districts"]:
        root_stack = district["stacks"][root_player]
        opponent_stack = district["stacks"][opponent]
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


def value_from_serialized_state(state: SerializedStatePayload, root_player: PlayerId) -> float:
    if is_terminal_state(state):
        return terminal_value(state, root_player)

    opponent = _opponent_player_id(root_player)
    players_by_id = _player_states_by_id(state)
    root_state = players_by_id[root_player]
    opponent_state = players_by_id[opponent]

    resource_root = resource_total(root_state)
    resource_opponent = resource_total(opponent_state)
    hand_diff = len(root_state["hand"]) - len(opponent_state["hand"])
    resource_diff = resource_root - resource_opponent

    district_count = len(state["districts"])
    district_point_diff = 0.0
    rank_total_diff = 0.0
    deed_completion_diff = 0.0
    for district in state["districts"]:
        root_stack = district["stacks"][root_player]
        opponent_stack = district["stacks"][opponent]

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


def stack_score(stack: DistrictStackPayload) -> tuple[float, float]:
    developed_rank = sum(_card_rank(card_id) for card_id in stack["developed"])

    deed = stack.get("deed")
    deed_card = deed["cardId"] if deed is not None else ""
    deed_rank = _card_rank(deed_card)
    deed_progress = deed["progress"] if deed is not None else 0

    progress_ratio = 0.0
    if deed_card and deed_rank > 0:
        progress_ratio = min(1.0, deed_progress / float(deed_rank))

    return float(developed_rank), progress_ratio


def developed_rank_total(stack: DistrictStackPayload) -> int:
    return sum(_card_rank(card_id) for card_id in stack["developed"])


def ace_pressure_proxy(stack: DistrictStackPayload) -> int:
    developed = stack["developed"]
    non_ace = sum(1 for card_id in developed if _card_rank(card_id) != 1)
    ace_bonus = sum(1 for card_id in developed if _card_rank(card_id) == 1)
    return ace_bonus * non_ace


def deed_completion_ratio(stack: DistrictStackPayload) -> float:
    deed = stack.get("deed")
    if deed is None:
        return 0.0
    card_id = deed["cardId"]
    rank = _card_rank(card_id)
    if rank <= 0:
        return 0.0
    progress = deed["progress"]
    return min(1.0, max(0.0, progress / float(rank)))


def player_views_by_id(view: PlayerViewPayload) -> dict[PlayerId, ObservedPlayerPayload]:
    players_by_id: dict[PlayerId, ObservedPlayerPayload] = {}
    for player in view["players"]:
        players_by_id[player["id"]] = player
    if "PlayerA" not in players_by_id or "PlayerB" not in players_by_id:
        raise ValueError("View payload is missing one or more players.")
    return players_by_id


def resource_total(player_state: ObservedPlayerPayload | PlayerStatePayload) -> int:
    return sum(player_state["resources"][suit] for suit in SUITS)


def _player_states_by_id(state: SerializedStatePayload) -> dict[PlayerId, PlayerStatePayload]:
    players_by_id: dict[PlayerId, PlayerStatePayload] = {}
    for player in state["players"]:
        players_by_id[player["id"]] = player
    if "PlayerA" not in players_by_id or "PlayerB" not in players_by_id:
        raise ValueError("Serialized state is missing one or more players.")
    return players_by_id


def _opponent_player_id(player_id: PlayerId) -> PlayerId:
    return "PlayerB" if player_id == "PlayerA" else "PlayerA"


def _state_cache_key(state: SerializedStatePayload) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
