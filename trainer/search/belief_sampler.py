from __future__ import annotations

import copy
import random
from typing import Sequence, cast

from trainer.bridge_payloads import (
    ObservedPlayerPayload,
    PlayerId,
    PlayerViewPayload,
    SerializedStatePayload,
)

PROPERTY_CARD_IDS: tuple[str, ...] = tuple(str(card_id) for card_id in range(30))


def sample_determinized_worlds(
    *,
    state: SerializedStatePayload,
    view: PlayerViewPayload,
    root_player: PlayerId,
    worlds: int,
    rng: random.Random,
) -> list[SerializedStatePayload]:
    if worlds <= 0:
        raise ValueError("worlds must be > 0.")

    opponent_player = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players_by_id = _player_views_by_id(view)
    root_view = players_by_id[root_player]
    opponent_view = players_by_id[opponent_player]
    root_hand = list(root_view["hand"])
    opponent_hand_count = opponent_view["handCount"]
    draw_count = view["deck"]["drawCount"]

    known_cards = set(root_hand)
    known_cards.update(view["deck"]["discard"])
    known_cards.update(_district_property_cards(view))
    hidden_pool = [card_id for card_id in PROPERTY_CARD_IDS if card_id not in known_cards]

    expected_hidden = opponent_hand_count + draw_count
    if len(hidden_pool) != expected_hidden:
        raise ValueError(
            "Determinization card accounting mismatch. "
            f"expected={expected_hidden}, actual={len(hidden_pool)}"
        )

    sampled_worlds: list[SerializedStatePayload] = []
    for _ in range(worlds):
        shuffled_hidden = list(hidden_pool)
        rng.shuffle(shuffled_hidden)
        opponent_hand = shuffled_hidden[:opponent_hand_count]
        draw_cards = shuffled_hidden[opponent_hand_count : opponent_hand_count + draw_count]

        world_state = cast(SerializedStatePayload, copy.deepcopy(state))
        _replace_player_hand(world_state, root_player, root_hand)
        _replace_player_hand(world_state, opponent_player, opponent_hand)
        deck = world_state["deck"]
        deck["draw"] = draw_cards
        sampled_worlds.append(world_state)
    return sampled_worlds


def _player_views_by_id(view: PlayerViewPayload) -> dict[PlayerId, ObservedPlayerPayload]:
    out: dict[PlayerId, ObservedPlayerPayload] = {}
    for player in view["players"]:
        out[player["id"]] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def _district_property_cards(view: PlayerViewPayload) -> set[str]:
    cards: set[str] = set()
    for district in view["districts"]:
        stacks = district["stacks"]
        for player_id in ("PlayerA", "PlayerB"):
            stack = stacks[player_id]
            for card_id in stack["developed"]:
                cards.add(card_id)
            deed = stack.get("deed")
            if deed is not None:
                cards.add(deed["cardId"])
    return cards


def _replace_player_hand(
    state: SerializedStatePayload,
    player_id: PlayerId,
    hand: Sequence[str],
) -> None:
    for player in state["players"]:
        if player["id"] == player_id:
            player["hand"] = list(hand)
            return
    raise ValueError(f"Serialized state is missing player {player_id}.")
