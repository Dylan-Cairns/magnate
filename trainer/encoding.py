from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from .types import KeyedAction

SUITS: Sequence[str] = ("Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots")
SUIT_INDEX: Dict[str, int] = {suit: i for i, suit in enumerate(SUITS)}

PHASES: Sequence[str] = (
    "StartTurn",
    "TaxCheck",
    "CollectIncome",
    "ActionWindow",
    "DrawCard",
    "GameOver",
)
PHASE_INDEX: Dict[str, int] = {phase: i for i, phase in enumerate(PHASES)}

ACTION_IDS: Sequence[str] = (
    "buy-deed",
    "choose-income-suit",
    "develop-deed",
    "develop-outright",
    "end-turn",
    "sell-card",
    "trade",
)
ACTION_ID_INDEX: Dict[str, int] = {action_id: i for i, action_id in enumerate(ACTION_IDS)}

PLAYER_IDS: Sequence[str] = ("PlayerA", "PlayerB")
PLAYER_INDEX: Dict[str, int] = {player_id: i for i, player_id in enumerate(PLAYER_IDS)}

CROWN_SUIT_BY_CARD_ID: Dict[str, str] = {
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

OBSERVATION_DIM = 186
ACTION_FEATURE_DIM = 40


def encode_observation(view: Mapping[str, Any]) -> List[float]:
    active_player_id = str(view.get("activePlayerId", "PlayerA"))
    opponent_id = "PlayerB" if active_player_id == "PlayerA" else "PlayerA"
    players = {
        str(player.get("id")): player for player in _as_list(view.get("players", []))
    }
    active_player = _as_mapping(players.get(active_player_id, {}))
    opponent_player = _as_mapping(players.get(opponent_id, {}))

    vector: List[float] = []

    vector.extend(_one_hot(PHASE_INDEX.get(str(view.get("phase")), -1), len(PHASES)))
    vector.append(_norm(_as_int(view.get("turn", 0)), MAX_TURN))
    vector.append(_bool(view.get("cardPlayedThisTurn")))
    vector.append(_norm(_as_int(view.get("finalTurnsRemaining", 0)), 2.0))

    deck = _as_mapping(view.get("deck", {}))
    vector.append(_norm(_as_int(deck.get("drawCount", 0)), MAX_DECK_SIZE))
    vector.append(_norm(len(_as_list(deck.get("discard", []))), MAX_DECK_SIZE))
    vector.append(_norm(_as_int(deck.get("reshuffles", 0)), 2.0))

    roll = _as_mapping(view.get("lastIncomeRoll", {}))
    vector.append(_norm(_as_int(roll.get("die1", 0)), 10.0))
    vector.append(_norm(_as_int(roll.get("die2", 0)), 10.0))

    vector.extend(_suit_one_hot(str(view.get("lastTaxSuit", ""))))
    vector.extend(_resource_vector(_as_mapping(active_player.get("resources", {}))))
    vector.extend(_resource_vector(_as_mapping(opponent_player.get("resources", {}))))

    vector.append(_norm(_as_int(active_player.get("handCount", 0)), MAX_HAND_COUNT))
    vector.append(_norm(_as_int(opponent_player.get("handCount", 0)), MAX_HAND_COUNT))

    vector.extend(_crown_suit_counts(_as_list(active_player.get("crowns", []))))
    vector.extend(_crown_suit_counts(_as_list(opponent_player.get("crowns", []))))

    districts = sorted(
        _as_list(view.get("districts", [])),
        key=lambda district: str(_as_mapping(district).get("id", "")),
    )
    for district in districts:
        district_map = _as_mapping(district)
        marker_suits = _as_list(district_map.get("markerSuitMask", []))
        vector.extend(_suit_count_vector(marker_suits, normalize_by=3.0))

        stacks = _as_mapping(district_map.get("stacks", {}))
        for player_id in (active_player_id, opponent_id):
            stack = _as_mapping(stacks.get(player_id, {}))
            vector.extend(_district_stack_features(stack))

    if len(vector) != OBSERVATION_DIM:
        raise ValueError(
            f"Observation vector length mismatch. expected={OBSERVATION_DIM}, actual={len(vector)}"
        )
    return vector


def encode_action_candidates(actions: Sequence[KeyedAction]) -> List[List[float]]:
    vectors = [encode_action(action) for action in actions]
    return vectors


def encode_action(action: KeyedAction) -> List[float]:
    payload = _as_mapping(action.action)
    vector: List[float] = []

    vector.extend(_one_hot(ACTION_ID_INDEX.get(action.action_id, -1), len(ACTION_IDS)))

    card_id = _as_str(payload.get("cardId", ""))
    card_rank = _card_rank(card_id)
    vector.append(_norm(_card_numeric_id(card_id), MAX_CARD_ID))
    vector.append(_norm(card_rank, 10.0))

    district_id = _as_str(payload.get("districtId", ""))
    vector.append(_norm(_district_index(district_id), 5.0))

    vector.extend(_one_hot(PLAYER_INDEX.get(_as_str(payload.get("playerId", "")), -1), 2))
    vector.extend(_suit_one_hot(_as_str(payload.get("suit", ""))))
    vector.extend(_suit_one_hot(_as_str(payload.get("give", ""))))
    vector.extend(_suit_one_hot(_as_str(payload.get("receive", ""))))

    token_map = _as_mapping(payload.get("payment", payload.get("tokens", {})))
    token_vector = _resource_vector(token_map, normalize_by=MAX_TOKEN_COUNT)
    vector.extend(token_vector)
    vector.append(_norm(sum(_as_int(value) for value in token_map.values()), MAX_TOKEN_COUNT))

    vector.append(1.0 if card_id else 0.0)
    vector.append(1.0 if district_id else 0.0)
    vector.append(1.0 if _is_property_card(card_id) else 0.0)

    if len(vector) != ACTION_FEATURE_DIM:
        raise ValueError(
            f"Action feature length mismatch. expected={ACTION_FEATURE_DIM}, actual={len(vector)}"
        )
    return vector


def _district_stack_features(stack: Mapping[str, Any]) -> List[float]:
    developed = _as_list(stack.get("developed", []))
    developed_ranks = [_card_rank(_as_str(card_id)) for card_id in developed if _is_property_card(_as_str(card_id))]
    developed_count = len(developed)
    developed_rank_sum = sum(developed_ranks)

    deed = _as_mapping(stack.get("deed", {}))
    deed_present = bool(deed)
    deed_card_id = _as_str(deed.get("cardId", ""))
    deed_progress = _as_int(deed.get("progress", 0))
    deed_target = _development_target(deed_card_id)
    deed_tokens = _as_mapping(deed.get("tokens", {}))

    features: List[float] = [
        _norm(developed_count, MAX_DISTRICT_STACK),
        _norm(developed_rank_sum, MAX_DISTRICT_RANK_SUM),
        1.0 if deed_present else 0.0,
        _norm(deed_progress, MAX_DEED_PROGRESS),
        _norm(deed_target, MAX_DEED_PROGRESS),
    ]
    features.extend(_resource_vector(deed_tokens, normalize_by=MAX_TOKEN_COUNT))
    return features


def _crown_suit_counts(crowns: Sequence[Any]) -> List[float]:
    counts: MutableMapping[str, int] = {suit: 0 for suit in SUITS}
    for card_id in crowns:
        suit = CROWN_SUIT_BY_CARD_ID.get(_as_str(card_id))
        if suit:
            counts[suit] += 1
    return [_norm(counts[suit], 3.0) for suit in SUITS]


def _resource_vector(
    resource_map: Mapping[str, Any],
    normalize_by: float = MAX_RESOURCES,
) -> List[float]:
    return [_norm(_as_int(resource_map.get(suit, 0)), normalize_by) for suit in SUITS]


def _suit_count_vector(suits: Sequence[Any], normalize_by: float = 1.0) -> List[float]:
    counts: MutableMapping[str, int] = {suit: 0 for suit in SUITS}
    for suit in suits:
        suit_name = _as_str(suit)
        if suit_name in counts:
            counts[suit_name] += 1
    return [_norm(counts[suit], normalize_by) for suit in SUITS]


def _suit_one_hot(suit: str) -> List[float]:
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


def _one_hot(index: int, length: int) -> List[float]:
    out = [0.0] * length
    if 0 <= index < length:
        out[index] = 1.0
    return out


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""
