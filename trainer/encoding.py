from __future__ import annotations

import math
from collections.abc import Sequence

from .bridge_payloads import (
    ActionId,
    DistrictStackPayload,
    GameActionPayload,
    GamePhase,
    ObservedPlayerPayload,
    PlayerId,
    PlayerViewPayload,
    ResourcePoolPayload,
    Suit,
    SuitCountsPayload,
)
from .types import KeyedAction

SUITS: tuple[Suit, ...] = ("Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots")
SUIT_INDEX: dict[str, int] = {suit: i for i, suit in enumerate(SUITS)}

PHASES: tuple[GamePhase, ...] = (
    "StartTurn",
    "TaxCheck",
    "CollectIncome",
    "ActionWindow",
    "DrawCard",
    "GameOver",
)
PHASE_INDEX: dict[GamePhase, int] = {phase: i for i, phase in enumerate(PHASES)}

ACTION_IDS: tuple[ActionId, ...] = (
    "buy-deed",
    "choose-income-suit",
    "develop-deed",
    "develop-outright",
    "end-turn",
    "sell-card",
    "trade",
)
ACTION_ID_INDEX: dict[ActionId, int] = {action_id: i for i, action_id in enumerate(ACTION_IDS)}

PLAYER_IDS: tuple[PlayerId, ...] = ("PlayerA", "PlayerB")
PLAYER_INDEX: dict[str, int] = {player_id: i for i, player_id in enumerate(PLAYER_IDS)}

CROWN_SUIT_BY_CARD_ID: dict[str, Suit] = {
    "30": "Knots",
    "31": "Leaves",
    "32": "Moons",
    "33": "Suns",
    "34": "Waves",
    "35": "Wyrms",
}

MAX_CARD_ID = 40.0
MAX_TURN = 40.0
MAX_DECK_SIZE = 41.0
MAX_RESOURCES = 20.0
MAX_HAND_COUNT = 10.0
MAX_DISTRICT_STACK = 9.0
MAX_DISTRICT_RANK_SUM = 60.0
MAX_DEED_PROGRESS = 9.0
MAX_TOKEN_COUNT = 9.0

ENCODING_VERSION = 2
OBSERVATION_DIM = 206
ACTION_FEATURE_DIM = 40

PROPERTY_SUITS_BY_CARD_ID: dict[str, tuple[Suit, ...]] = {
    "0": ("Knots",),
    "1": ("Leaves",),
    "2": ("Moons",),
    "3": ("Suns",),
    "4": ("Waves",),
    "5": ("Wyrms",),
    "6": ("Moons", "Knots"),
    "7": ("Suns", "Wyrms"),
    "8": ("Waves", "Leaves"),
    "9": ("Moons", "Waves"),
    "10": ("Suns", "Knots"),
    "11": ("Leaves", "Wyrms"),
    "12": ("Wyrms", "Knots"),
    "13": ("Moons", "Suns"),
    "14": ("Waves", "Leaves"),
    "15": ("Suns", "Waves"),
    "16": ("Moons", "Leaves"),
    "17": ("Wyrms", "Knots"),
    "18": ("Moons", "Waves"),
    "19": ("Leaves", "Knots"),
    "20": ("Suns", "Wyrms"),
    "21": ("Suns", "Knots"),
    "22": ("Waves", "Wyrms"),
    "23": ("Moons", "Leaves"),
    "24": ("Wyrms", "Knots"),
    "25": ("Moons", "Suns"),
    "26": ("Waves", "Leaves"),
    "27": ("Waves", "Wyrms"),
    "28": ("Leaves", "Knots"),
    "29": ("Moons", "Suns"),
}


def encode_observation(view: PlayerViewPayload) -> list[float]:
    active_player_id = view["activePlayerId"]
    opponent_id = _opponent_player_id(active_player_id)
    players = _player_views_by_id(view)
    active_player = players[active_player_id]
    opponent_player = players[opponent_id]

    vector: list[float] = []

    vector.extend(_one_hot(PHASE_INDEX.get(view["phase"], -1), len(PHASES)))
    vector.append(_norm(view["turn"], MAX_TURN))
    vector.append(_bool(view["cardPlayedThisTurn"]))
    vector.append(_norm(view.get("finalTurnsRemaining", 0), 2.0))

    deck = view["deck"]
    vector.append(_norm(deck["drawCount"], MAX_DECK_SIZE))
    vector.append(_norm(len(deck["discard"]), MAX_DECK_SIZE))
    vector.append(_norm(deck["reshuffles"], 2.0))

    roll = view.get("lastIncomeRoll")
    if roll is None:
        raise ValueError("View payload is missing lastIncomeRoll.")
    vector.append(_norm(roll["die1"], 10.0))
    vector.append(_norm(roll["die2"], 10.0))

    vector.extend(_suit_one_hot(view.get("lastTaxSuit", "")))
    vector.extend(_resource_vector(active_player["resources"]))
    vector.extend(_resource_vector(opponent_player["resources"]))

    vector.append(_norm(active_player["handCount"], MAX_HAND_COUNT))
    vector.append(_norm(opponent_player["handCount"], MAX_HAND_COUNT))

    vector.extend(_crown_suit_counts(active_player["crowns"]))
    vector.extend(_crown_suit_counts(opponent_player["crowns"]))
    vector.extend(_hand_suit_histogram(active_player["hand"]))
    vector.extend(_hand_rank_histogram(active_player["hand"]))
    vector.extend(
        _endgame_tiebreak_features(
            view=view,
            active_player_id=active_player_id,
            opponent_id=opponent_id,
            active_player=active_player,
            opponent_player=opponent_player,
        )
    )

    districts = sorted(view["districts"], key=lambda district: district["id"])
    for district in districts:
        vector.extend(_suit_count_vector(district["markerSuitMask"], normalize_by=3.0))

        stacks = district["stacks"]
        for player_id in (active_player_id, opponent_id):
            vector.extend(_district_stack_features(stacks[player_id]))

    if len(vector) != OBSERVATION_DIM:
        raise ValueError(
            f"Observation vector length mismatch. expected={OBSERVATION_DIM}, actual={len(vector)}"
        )
    return vector


def encode_action_candidates(actions: Sequence[KeyedAction]) -> list[list[float]]:
    return [encode_action(action) for action in actions]


def encode_action(action: KeyedAction) -> list[float]:
    payload = action.action
    card_id = _action_card_id(payload)
    district_id = _action_district_id(payload)
    player_id = _action_player_id(payload)
    suit = _action_suit(payload)
    give = payload["give"] if payload["type"] == "trade" else ""
    receive = payload["receive"] if payload["type"] == "trade" else ""
    token_map = _action_suit_counts(payload)

    vector: list[float] = []
    vector.extend(_one_hot(ACTION_ID_INDEX.get(action.action_id, -1), len(ACTION_IDS)))

    card_rank = _card_rank(card_id)
    vector.append(_norm(_card_numeric_id(card_id), MAX_CARD_ID))
    vector.append(_norm(card_rank, 10.0))

    vector.append(_norm(_district_index(district_id), 5.0))
    vector.extend(_one_hot(PLAYER_INDEX.get(player_id, -1), len(PLAYER_IDS)))
    vector.extend(_suit_one_hot(suit))
    vector.extend(_suit_one_hot(give))
    vector.extend(_suit_one_hot(receive))

    token_vector = _suit_count_payload_vector(token_map, normalize_by=MAX_TOKEN_COUNT)
    vector.extend(token_vector)
    vector.append(_norm(_suit_count_payload_total(token_map), MAX_TOKEN_COUNT))

    vector.append(1.0 if card_id else 0.0)
    vector.append(1.0 if district_id else 0.0)
    vector.append(1.0 if _is_property_card(card_id) else 0.0)

    if len(vector) != ACTION_FEATURE_DIM:
        raise ValueError(
            f"Action feature length mismatch. expected={ACTION_FEATURE_DIM}, actual={len(vector)}"
        )
    return vector


def _district_stack_features(stack: DistrictStackPayload) -> list[float]:
    developed = stack["developed"]
    developed_ranks = [_card_rank(card_id) for card_id in developed if _is_property_card(card_id)]
    developed_count = len(developed)
    developed_rank_sum = sum(developed_ranks)

    deed = stack.get("deed")
    deed_present = deed is not None
    deed_card_id = deed["cardId"] if deed is not None else ""
    deed_progress = deed["progress"] if deed is not None else 0
    deed_target = _development_target(deed_card_id)
    deed_tokens: SuitCountsPayload = deed["tokens"] if deed is not None else {}

    features: list[float] = [
        _norm(developed_count, MAX_DISTRICT_STACK),
        _norm(developed_rank_sum, MAX_DISTRICT_RANK_SUM),
        1.0 if deed_present else 0.0,
        _norm(deed_progress, MAX_DEED_PROGRESS),
        _norm(deed_target, MAX_DEED_PROGRESS),
    ]
    features.extend(_suit_count_payload_vector(deed_tokens, normalize_by=MAX_TOKEN_COUNT))
    return features


def _crown_suit_counts(crowns: Sequence[str]) -> list[float]:
    counts: dict[Suit, int] = {suit: 0 for suit in SUITS}
    for card_id in crowns:
        suit = CROWN_SUIT_BY_CARD_ID.get(card_id)
        if suit is not None:
            counts[suit] += 1
    return [_norm(counts[suit], 3.0) for suit in SUITS]


def _hand_suit_histogram(hand: Sequence[str]) -> list[float]:
    counts: dict[Suit, int] = {suit: 0 for suit in SUITS}
    for card_id in hand:
        for suit in PROPERTY_SUITS_BY_CARD_ID.get(card_id, ()):
            counts[suit] += 1
    return [_norm(counts[suit], MAX_HAND_COUNT * 2.0) for suit in SUITS]


def _hand_rank_histogram(hand: Sequence[str]) -> list[float]:
    counts = [0] * 10
    for card_id in hand:
        rank = _card_rank(card_id)
        if 1 <= rank <= 10:
            counts[rank - 1] += 1
    return [_norm(value, MAX_HAND_COUNT) for value in counts]


def _endgame_tiebreak_features(
    *,
    view: PlayerViewPayload,
    active_player_id: PlayerId,
    opponent_id: PlayerId,
    active_player: ObservedPlayerPayload,
    opponent_player: ObservedPlayerPayload,
) -> list[float]:
    deck = view["deck"]
    reshuffles = deck["reshuffles"]
    final_turns_remaining = view.get("finalTurnsRemaining", 0)
    endgame_flag = 1.0 if reshuffles >= 2 or final_turns_remaining > 0 else 0.0

    district_point_diff = 0.0
    developed_rank_diff = 0.0
    for district in view["districts"]:
        stacks = district["stacks"]
        active_stack = stacks[active_player_id]
        opponent_stack = stacks[opponent_id]
        active_rank = _developed_rank_sum(active_stack)
        opponent_rank = _developed_rank_sum(opponent_stack)
        developed_rank_diff += active_rank - opponent_rank
        if active_rank > opponent_rank:
            district_point_diff += 1.0
        elif active_rank < opponent_rank:
            district_point_diff -= 1.0

    district_term = district_point_diff / float(max(1, len(view["districts"])))
    rank_term = math.tanh(developed_rank_diff / 18.0)
    resource_term = math.tanh((_resource_total(active_player) - _resource_total(opponent_player)) / 10.0)
    return [
        endgame_flag,
        _signed_to_unit_interval(district_term),
        _signed_to_unit_interval(rank_term),
        _signed_to_unit_interval(resource_term),
    ]


def _developed_rank_sum(stack: DistrictStackPayload) -> int:
    return sum(_card_rank(card_id) for card_id in stack["developed"])


def _resource_vector(
    resource_map: ResourcePoolPayload,
    normalize_by: float = MAX_RESOURCES,
) -> list[float]:
    return [_norm(resource_map[suit], normalize_by) for suit in SUITS]


def _suit_count_payload_vector(
    suit_counts: SuitCountsPayload,
    normalize_by: float = 1.0,
) -> list[float]:
    return [_norm(suit_counts.get(suit, 0), normalize_by) for suit in SUITS]


def _suit_count_vector(suits: Sequence[Suit], normalize_by: float = 1.0) -> list[float]:
    counts: dict[Suit, int] = {suit: 0 for suit in SUITS}
    for suit in suits:
        counts[suit] += 1
    return [_norm(counts[suit], normalize_by) for suit in SUITS]


def _suit_count_payload_total(suit_counts: SuitCountsPayload) -> int:
    return sum(suit_counts.get(suit, 0) for suit in SUITS)


def _action_card_id(payload: GameActionPayload) -> str:
    return payload["cardId"] if "cardId" in payload else ""


def _action_district_id(payload: GameActionPayload) -> str:
    return payload["districtId"] if "districtId" in payload else ""


def _action_player_id(payload: GameActionPayload) -> PlayerId | str:
    return payload["playerId"] if payload["type"] == "choose-income-suit" else ""


def _action_suit(payload: GameActionPayload) -> Suit | str:
    return payload["suit"] if payload["type"] == "choose-income-suit" else ""


def _action_suit_counts(payload: GameActionPayload) -> SuitCountsPayload:
    if payload["type"] == "develop-deed":
        return payload["tokens"]
    if payload["type"] == "develop-outright":
        return payload["payment"]
    return {}


def _player_views_by_id(view: PlayerViewPayload) -> dict[PlayerId, ObservedPlayerPayload]:
    players_by_id: dict[PlayerId, ObservedPlayerPayload] = {}
    for player in view["players"]:
        players_by_id[player["id"]] = player
    if "PlayerA" not in players_by_id or "PlayerB" not in players_by_id:
        raise ValueError("View payload is missing one or more players.")
    return players_by_id


def _opponent_player_id(player_id: PlayerId) -> PlayerId:
    return "PlayerB" if player_id == "PlayerA" else "PlayerA"


def _suit_one_hot(suit: Suit | str) -> list[float]:
    return _one_hot(SUIT_INDEX.get(suit, -1), len(SUITS))


def _district_index(district_id: str) -> int:
    if district_id.startswith("D") and district_id[1:].isdigit():
        return int(district_id[1:])
    return 0


def _development_target(card_id: str) -> int:
    if not _is_property_card(card_id):
        return 0
    rank = _card_rank(card_id)
    if rank == 1:
        return 3
    return rank


def _is_property_card(card_id: str) -> bool:
    numeric = _card_numeric_id(card_id)
    return 0 <= numeric <= 29


def _card_numeric_id(card_id: str) -> int:
    if card_id.isdigit():
        return int(card_id)
    return 0


def _card_rank(card_id: str) -> int:
    numeric = _card_numeric_id(card_id)
    if 0 <= numeric <= 5:
        return 1
    if 6 <= numeric <= 29:
        return 2 + ((numeric - 6) // 3)
    if 30 <= numeric <= 35:
        return 10
    return 0


def _norm(value: int, ceiling: float) -> float:
    if ceiling <= 0:
        return 0.0
    clipped = min(max(float(value), 0.0), ceiling)
    return clipped / ceiling


def _signed_to_unit_interval(value: float) -> float:
    clipped = min(max(value, -1.0), 1.0)
    return (clipped + 1.0) * 0.5


def _one_hot(index: int, length: int) -> list[float]:
    out = [0.0] * length
    if 0 <= index < length:
        out[index] = 1.0
    return out


def _bool(value: bool) -> float:
    return 1.0 if value else 0.0


def _resource_total(player_state: ObservedPlayerPayload) -> int:
    return sum(player_state["resources"][suit] for suit in SUITS)
