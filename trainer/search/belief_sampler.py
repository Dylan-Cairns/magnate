from __future__ import annotations

import copy
import random
from typing import Any, Dict, Mapping, Sequence

from trainer.types import PlayerId

PROPERTY_CARD_IDS: tuple[str, ...] = tuple(str(card_id) for card_id in range(30))


def sample_determinized_worlds(
    *,
    state: Mapping[str, Any],
    view: Mapping[str, Any],
    root_player: PlayerId,
    worlds: int,
    rng: random.Random,
) -> list[Dict[str, Any]]:
    if worlds <= 0:
        raise ValueError("worlds must be > 0.")

    opponent_player = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players_by_id = _player_views_by_id(view)
    root_view = players_by_id[root_player]
    opponent_view = players_by_id[opponent_player]
    root_hand = _as_card_list(root_view.get("hand"))
    opponent_hand_count = _as_int(opponent_view.get("handCount"))
    draw_count = _as_int(_as_mapping(view.get("deck")).get("drawCount"))

    known_cards = set(root_hand)
    known_cards.update(_as_card_list(_as_mapping(view.get("deck")).get("discard")))
    known_cards.update(_district_property_cards(view))
    hidden_pool = [card_id for card_id in PROPERTY_CARD_IDS if card_id not in known_cards]

    expected_hidden = opponent_hand_count + draw_count
    if len(hidden_pool) != expected_hidden:
        raise ValueError(
            "Determinization card accounting mismatch. "
            f"expected={expected_hidden}, actual={len(hidden_pool)}"
        )

    sampled_worlds: list[Dict[str, Any]] = []
    for _ in range(worlds):
        shuffled_hidden = list(hidden_pool)
        rng.shuffle(shuffled_hidden)
        opponent_hand = shuffled_hidden[:opponent_hand_count]
        draw_cards = shuffled_hidden[opponent_hand_count : opponent_hand_count + draw_count]

        world_state = copy.deepcopy(dict(state))
        _replace_player_hand(world_state, root_player, root_hand)
        _replace_player_hand(world_state, opponent_player, opponent_hand)
        deck = _as_mapping(world_state.get("deck"))
        deck["draw"] = draw_cards
        sampled_worlds.append(world_state)
    return sampled_worlds


def _player_views_by_id(view: Mapping[str, Any]) -> Dict[PlayerId, Dict[str, Any]]:
    players = view.get("players")
    if not isinstance(players, list):
        raise ValueError("View payload is missing players list.")

    out: Dict[PlayerId, Dict[str, Any]] = {}
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = player.get("id")
        if player_id in ("PlayerA", "PlayerB"):
            out[player_id] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def _district_property_cards(view: Mapping[str, Any]) -> set[str]:
    cards: set[str] = set()
    districts = view.get("districts")
    if not isinstance(districts, list):
        return cards

    for district in districts:
        if not isinstance(district, dict):
            continue
        stacks = district.get("stacks")
        if not isinstance(stacks, dict):
            continue
        for player_id in ("PlayerA", "PlayerB"):
            stack = stacks.get(player_id)
            if not isinstance(stack, dict):
                continue
            for card_id in _as_card_list(stack.get("developed")):
                cards.add(card_id)
            deed = stack.get("deed")
            if isinstance(deed, dict):
                deed_card = deed.get("cardId")
                if isinstance(deed_card, str):
                    cards.add(deed_card)
    return cards


def _replace_player_hand(state: Dict[str, Any], player_id: PlayerId, hand: Sequence[str]) -> None:
    players = state.get("players")
    if not isinstance(players, list):
        raise ValueError("Serialized state is missing players list.")
    for player in players:
        if isinstance(player, dict) and player.get("id") == player_id:
            player["hand"] = list(hand)
            return
    raise ValueError(f"Serialized state is missing player {player_id}.")


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object mapping, got {type(value).__name__}.")


def _as_card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            out.append(entry)
    return out


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0
