from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

from .bridge_payloads import (
    ActionId,
    BridgeActionSurfacePayload,
    BridgeCommandName,
    BridgeMetadataPayload,
    BridgeModelIOInputsPayload,
    BridgeModelIOOutputsPayload,
    BridgeModelIOPayload,
    BridgeObservationSpecPayload,
    DeckStatePayload,
    DeckViewPayload,
    DeedPayload,
    DistrictPayload,
    DistrictStackPayload,
    DistrictStacksPayload,
    FinalScorePayload,
    GameActionPayload,
    GameLogEntryPayload,
    GamePhase,
    IncomeChoicePayload,
    IncomeRollPayload,
    ObservedPlayerPayload,
    PlayerId,
    PlayerStatePayload,
    PlayerTotalsPayload,
    PlayerViewPayload,
    ResourcePoolPayload,
    SerializedStatePayload,
    Suit,
    SuitCountsPayload,
    Winner,
    WinnerDecider,
)
from .types import KeyedAction, LegalActionsResult, ObservationResult, StateResult

JsonMapping = Mapping[str, object]

_SUITS: tuple[Suit, ...] = ("Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots")
_PLAYER_IDS: tuple[PlayerId, ...] = ("PlayerA", "PlayerB")
_GAME_PHASES: tuple[GamePhase, ...] = (
    "StartTurn",
    "TaxCheck",
    "CollectIncome",
    "ActionWindow",
    "DrawCard",
    "GameOver",
)
_ACTION_IDS: tuple[ActionId, ...] = (
    "buy-deed",
    "choose-income-suit",
    "develop-deed",
    "develop-outright",
    "end-turn",
    "sell-card",
    "trade",
)
_BRIDGE_COMMANDS: tuple[BridgeCommandName, ...] = (
    "metadata",
    "reset",
    "step",
    "legalActions",
    "observation",
    "serialize",
)
_WINNER_DECIDERS: tuple[WinnerDecider, ...] = ("districts", "rank-total", "resources", "draw")


def parse_bridge_metadata_result(result: JsonMapping) -> BridgeMetadataPayload:
    commands: list[BridgeCommandName] = [
        _require_bridge_command(entry, f"metadata.commands[{index}]")
        for index, entry in enumerate(_require_list(result.get("commands"), "metadata.commands"))
    ]
    action_ids: list[ActionId] = [
        _require_action_id(entry, f"metadata.actionIds[{index}]")
        for index, entry in enumerate(_require_list(result.get("actionIds"), "metadata.actionIds"))
    ]
    action_surface_raw = _require_mapping(result.get("actionSurface"), "metadata.actionSurface")
    observation_spec_raw = _require_mapping(result.get("observationSpec"), "metadata.observationSpec")
    model_io_raw = _require_mapping(result.get("modelIO"), "metadata.modelIO")
    inputs_raw = _require_mapping(model_io_raw.get("inputs"), "metadata.modelIO.inputs")
    outputs_raw = _require_mapping(model_io_raw.get("outputs"), "metadata.modelIO.outputs")

    action_surface: BridgeActionSurfacePayload = {
        "stableKey": _require_literal(
            action_surface_raw.get("stableKey"),
            "metadata.actionSurface.stableKey",
            "actionKey",
        ),
        "canonicalOrder": _require_literal(
            action_surface_raw.get("canonicalOrder"),
            "metadata.actionSurface.canonicalOrder",
            "ascending_lexicographic_action_key",
        ),
    }
    observation_spec: BridgeObservationSpecPayload = {
        "name": _require_literal(
            observation_spec_raw.get("name"),
            "metadata.observationSpec.name",
            "player_view_v1",
        ),
        "defaultViewer": _require_literal(
            observation_spec_raw.get("defaultViewer"),
            "metadata.observationSpec.defaultViewer",
            "active-player",
        ),
        "optionalMask": _require_literal(
            observation_spec_raw.get("optionalMask"),
            "metadata.observationSpec.optionalMask",
            "legal action keys",
        ),
    }
    model_inputs: BridgeModelIOInputsPayload = {
        "observation": _require_literal(
            inputs_raw.get("observation"),
            "metadata.modelIO.inputs.observation",
            "observation",
        ),
        "actionMask": _require_literal(
            inputs_raw.get("actionMask"),
            "metadata.modelIO.inputs.actionMask",
            "action_mask",
        ),
    }
    model_outputs: BridgeModelIOOutputsPayload = {
        "maskedLogits": _require_literal(
            outputs_raw.get("maskedLogits"),
            "metadata.modelIO.outputs.maskedLogits",
            "masked_logits",
        ),
        "value": _require_literal(
            outputs_raw.get("value"),
            "metadata.modelIO.outputs.value",
            "value",
        ),
    }
    model_io: BridgeModelIOPayload = {
        "inputs": model_inputs,
        "outputs": model_outputs,
    }
    return {
        "contractName": _require_literal(
            result.get("contractName"),
            "metadata.contractName",
            "magnate_bridge",
        ),
        "contractVersion": _require_literal(
            result.get("contractVersion"),
            "metadata.contractVersion",
            "v1",
        ),
        "schemaVersion": _require_int(result.get("schemaVersion"), "metadata.schemaVersion"),
        "commands": commands,
        "actionIds": action_ids,
        "actionSurface": action_surface,
        "observationSpec": observation_spec,
        "modelIO": model_io,
    }


def parse_state_result_payload(result: JsonMapping) -> StateResult:
    return StateResult(
        state=parse_serialized_state_payload(result.get("state"), label="stateResult.state"),
        view=parse_player_view_payload(result.get("view"), label="stateResult.view"),
        terminal=_require_bool(result.get("terminal"), "stateResult.terminal"),
    )


def parse_legal_actions_result_payload(result: JsonMapping) -> LegalActionsResult:
    raw_actions = _require_list(result.get("actions"), "legalActions.actions")
    actions = [
        parse_keyed_action_payload(entry, label=f"legalActions.actions[{index}]")
        for index, entry in enumerate(raw_actions)
    ]
    return LegalActionsResult(
        actions=actions,
        active_player_id=_require_player_id(
            result.get("activePlayerId"),
            "legalActions.activePlayerId",
        ),
        phase=_require_phase(result.get("phase"), "legalActions.phase"),
    )


def parse_observation_result_payload(result: JsonMapping) -> ObservationResult:
    legal_action_mask: list[str] | None = None
    if "legalActionMask" in result and result["legalActionMask"] is not None:
        legal_action_mask = [
            _require_str(entry, f"observation.legalActionMask[{index}]")
            for index, entry in enumerate(
                _require_list(result.get("legalActionMask"), "observation.legalActionMask")
            )
        ]
    return ObservationResult(
        view=parse_player_view_payload(result.get("view"), label="observation.view"),
        legal_action_mask=legal_action_mask,
    )


def parse_keyed_action_payload(value: object, *, label: str) -> KeyedAction:
    mapping = _require_mapping(value, label)
    action_id = _require_action_id(mapping.get("actionId"), f"{label}.actionId")
    action_key = _require_str(mapping.get("actionKey"), f"{label}.actionKey")
    action = parse_game_action_payload(mapping.get("action"), label=f"{label}.action")
    if action_id != action["type"]:
        raise RuntimeError(
            f"{label}.actionId must match action.type, got {action_id!r} vs {action['type']!r}."
        )
    return KeyedAction(
        action_id=action_id,
        action_key=action_key,
        action=action,
    )


def parse_serialized_state_payload(value: object, *, label: str) -> SerializedStatePayload:
    mapping = _require_mapping(value, label)
    payload: SerializedStatePayload = {
        "schemaVersion": _require_int(mapping.get("schemaVersion"), f"{label}.schemaVersion"),
        "seed": _require_str(mapping.get("seed"), f"{label}.seed"),
        "rngCursor": _require_int(mapping.get("rngCursor"), f"{label}.rngCursor"),
        "deck": _parse_deck_state_payload(mapping.get("deck"), label=f"{label}.deck"),
        "players": _parse_player_state_list(mapping.get("players"), label=f"{label}.players"),
        "activePlayerIndex": _require_int(
            mapping.get("activePlayerIndex"),
            f"{label}.activePlayerIndex",
        ),
        "turn": _require_int(mapping.get("turn"), f"{label}.turn"),
        "phase": _require_phase(mapping.get("phase"), f"{label}.phase"),
        "districts": _parse_district_list(mapping.get("districts"), label=f"{label}.districts"),
        "cardPlayedThisTurn": _require_bool(
            mapping.get("cardPlayedThisTurn"),
            f"{label}.cardPlayedThisTurn",
        ),
        "log": _parse_log_list(mapping.get("log"), label=f"{label}.log"),
    }
    if (raw := mapping.get("finalTurnsRemaining")) is not None:
        payload["finalTurnsRemaining"] = _require_int(raw, f"{label}.finalTurnsRemaining")
    if (raw := mapping.get("lastIncomeRoll")) is not None:
        payload["lastIncomeRoll"] = _parse_income_roll(raw, label=f"{label}.lastIncomeRoll")
    if (raw := mapping.get("lastTaxSuit")) is not None:
        payload["lastTaxSuit"] = _require_suit(raw, f"{label}.lastTaxSuit")
    if (raw := mapping.get("pendingIncomeChoices")) is not None:
        payload["pendingIncomeChoices"] = _parse_income_choice_list(
            raw,
            label=f"{label}.pendingIncomeChoices",
        )
    if (raw := mapping.get("incomeChoiceReturnPlayerId")) is not None:
        payload["incomeChoiceReturnPlayerId"] = _require_player_id(
            raw,
            f"{label}.incomeChoiceReturnPlayerId",
        )
    if (raw := mapping.get("finalScore")) is not None:
        payload["finalScore"] = _parse_final_score(raw, label=f"{label}.finalScore")
    return payload


def parse_player_view_payload(value: object, *, label: str) -> PlayerViewPayload:
    mapping = _require_mapping(value, label)
    payload: PlayerViewPayload = {
        "viewerId": _require_player_id(mapping.get("viewerId"), f"{label}.viewerId"),
        "activePlayerId": _require_player_id(
            mapping.get("activePlayerId"),
            f"{label}.activePlayerId",
        ),
        "turn": _require_int(mapping.get("turn"), f"{label}.turn"),
        "phase": _require_phase(mapping.get("phase"), f"{label}.phase"),
        "districts": _parse_district_list(mapping.get("districts"), label=f"{label}.districts"),
        "players": _parse_observed_player_list(mapping.get("players"), label=f"{label}.players"),
        "deck": _parse_deck_view_payload(mapping.get("deck"), label=f"{label}.deck"),
        "cardPlayedThisTurn": _require_bool(
            mapping.get("cardPlayedThisTurn"),
            f"{label}.cardPlayedThisTurn",
        ),
        "log": _parse_log_list(mapping.get("log"), label=f"{label}.log"),
    }
    if (raw := mapping.get("finalTurnsRemaining")) is not None:
        payload["finalTurnsRemaining"] = _require_int(raw, f"{label}.finalTurnsRemaining")
    if (raw := mapping.get("lastIncomeRoll")) is not None:
        payload["lastIncomeRoll"] = _parse_income_roll(raw, label=f"{label}.lastIncomeRoll")
    if (raw := mapping.get("lastTaxSuit")) is not None:
        payload["lastTaxSuit"] = _require_suit(raw, f"{label}.lastTaxSuit")
    if (raw := mapping.get("pendingIncomeChoices")) is not None:
        payload["pendingIncomeChoices"] = _parse_income_choice_list(
            raw,
            label=f"{label}.pendingIncomeChoices",
        )
    if (raw := mapping.get("incomeChoiceReturnPlayerId")) is not None:
        payload["incomeChoiceReturnPlayerId"] = _require_player_id(
            raw,
            f"{label}.incomeChoiceReturnPlayerId",
        )
    if (raw := mapping.get("finalScore")) is not None:
        payload["finalScore"] = _parse_final_score(raw, label=f"{label}.finalScore")
    return payload


def parse_game_action_payload(value: object, *, label: str) -> GameActionPayload:
    mapping = _require_mapping(value, label)
    action_type = _require_action_id(mapping.get("type"), f"{label}.type")
    if action_type == "buy-deed":
        return {
            "type": "buy-deed",
            "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
            "districtId": _require_str(mapping.get("districtId"), f"{label}.districtId"),
        }
    if action_type == "choose-income-suit":
        return {
            "type": "choose-income-suit",
            "playerId": _require_player_id(mapping.get("playerId"), f"{label}.playerId"),
            "districtId": _require_str(mapping.get("districtId"), f"{label}.districtId"),
            "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
            "suit": _require_suit(mapping.get("suit"), f"{label}.suit"),
        }
    if action_type == "develop-deed":
        return {
            "type": "develop-deed",
            "districtId": _require_str(mapping.get("districtId"), f"{label}.districtId"),
            "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
            "tokens": _parse_suit_counts(mapping.get("tokens"), label=f"{label}.tokens"),
        }
    if action_type == "develop-outright":
        return {
            "type": "develop-outright",
            "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
            "districtId": _require_str(mapping.get("districtId"), f"{label}.districtId"),
            "payment": _parse_suit_counts(mapping.get("payment"), label=f"{label}.payment"),
        }
    if action_type == "end-turn":
        return {"type": "end-turn"}
    if action_type == "sell-card":
        return {
            "type": "sell-card",
            "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
        }
    return {
        "type": "trade",
        "give": _require_suit(mapping.get("give"), f"{label}.give"),
        "receive": _require_suit(mapping.get("receive"), f"{label}.receive"),
    }


def _parse_resource_pool(value: object, *, label: str) -> ResourcePoolPayload:
    mapping = _require_mapping(value, label)
    return {
        "Moons": _require_int(mapping.get("Moons"), f"{label}.Moons"),
        "Suns": _require_int(mapping.get("Suns"), f"{label}.Suns"),
        "Waves": _require_int(mapping.get("Waves"), f"{label}.Waves"),
        "Leaves": _require_int(mapping.get("Leaves"), f"{label}.Leaves"),
        "Wyrms": _require_int(mapping.get("Wyrms"), f"{label}.Wyrms"),
        "Knots": _require_int(mapping.get("Knots"), f"{label}.Knots"),
    }


def _parse_suit_counts(value: object, *, label: str) -> SuitCountsPayload:
    mapping = _require_mapping(value, label)
    payload: SuitCountsPayload = {}
    for suit in _SUITS:
        if suit in mapping and mapping[suit] is not None:
            payload[suit] = _require_int(mapping[suit], f"{label}.{suit}")
    return payload


def _parse_income_roll(value: object, *, label: str) -> IncomeRollPayload:
    mapping = _require_mapping(value, label)
    return {
        "die1": _require_int(mapping.get("die1"), f"{label}.die1"),
        "die2": _require_int(mapping.get("die2"), f"{label}.die2"),
    }


def _parse_income_choice(value: object, *, label: str) -> IncomeChoicePayload:
    mapping = _require_mapping(value, label)
    return {
        "playerId": _require_player_id(mapping.get("playerId"), f"{label}.playerId"),
        "districtId": _require_str(mapping.get("districtId"), f"{label}.districtId"),
        "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
        "suits": _parse_suit_list(mapping.get("suits"), label=f"{label}.suits"),
    }


def _parse_income_choice_list(value: object, *, label: str) -> list[IncomeChoicePayload]:
    return [
        _parse_income_choice(entry, label=f"{label}[{index}]")
        for index, entry in enumerate(_require_list(value, label))
    ]


def _parse_log_entry(value: object, *, label: str) -> GameLogEntryPayload:
    mapping = _require_mapping(value, label)
    payload: GameLogEntryPayload = {
        "turn": _require_int(mapping.get("turn"), f"{label}.turn"),
        "player": _require_player_id(mapping.get("player"), f"{label}.player"),
        "phase": _require_phase(mapping.get("phase"), f"{label}.phase"),
        "summary": _require_str(mapping.get("summary"), f"{label}.summary"),
    }
    if (raw := mapping.get("details")) is not None:
        payload["details"] = dict(_require_mapping(raw, f"{label}.details"))
    return payload


def _parse_log_list(value: object, *, label: str) -> list[GameLogEntryPayload]:
    return [
        _parse_log_entry(entry, label=f"{label}[{index}]")
        for index, entry in enumerate(_require_list(value, label))
    ]


def _parse_player_totals(value: object, *, label: str) -> PlayerTotalsPayload:
    mapping = _require_mapping(value, label)
    return {
        "PlayerA": _require_int(mapping.get("PlayerA"), f"{label}.PlayerA"),
        "PlayerB": _require_int(mapping.get("PlayerB"), f"{label}.PlayerB"),
    }


def _parse_final_score(value: object, *, label: str) -> FinalScorePayload:
    mapping = _require_mapping(value, label)
    return {
        "districtPoints": _parse_player_totals(
            mapping.get("districtPoints"),
            label=f"{label}.districtPoints",
        ),
        "rankTotals": _parse_player_totals(mapping.get("rankTotals"), label=f"{label}.rankTotals"),
        "resourceTotals": _parse_player_totals(
            mapping.get("resourceTotals"),
            label=f"{label}.resourceTotals",
        ),
        "winner": _require_winner(mapping.get("winner"), f"{label}.winner"),
        "decidedBy": _require_winner_decider(mapping.get("decidedBy"), f"{label}.decidedBy"),
    }


def _parse_deed_payload(value: object, *, label: str) -> DeedPayload:
    mapping = _require_mapping(value, label)
    return {
        "cardId": _require_str(mapping.get("cardId"), f"{label}.cardId"),
        "progress": _require_int(mapping.get("progress"), f"{label}.progress"),
        "tokens": _parse_suit_counts(mapping.get("tokens"), label=f"{label}.tokens"),
    }


def _parse_district_stack(value: object, *, label: str) -> DistrictStackPayload:
    mapping = _require_mapping(value, label)
    payload: DistrictStackPayload = {
        "developed": _parse_string_list(mapping.get("developed"), label=f"{label}.developed"),
    }
    if (raw := mapping.get("deed")) is not None:
        payload["deed"] = _parse_deed_payload(raw, label=f"{label}.deed")
    return payload


def _parse_district_payload(value: object, *, label: str) -> DistrictPayload:
    mapping = _require_mapping(value, label)
    stacks_raw = _require_mapping(mapping.get("stacks"), f"{label}.stacks")
    stacks: DistrictStacksPayload = {
        "PlayerA": _parse_district_stack(stacks_raw.get("PlayerA"), label=f"{label}.stacks.PlayerA"),
        "PlayerB": _parse_district_stack(stacks_raw.get("PlayerB"), label=f"{label}.stacks.PlayerB"),
    }
    return {
        "id": _require_str(mapping.get("id"), f"{label}.id"),
        "markerSuitMask": _parse_suit_list(mapping.get("markerSuitMask"), label=f"{label}.markerSuitMask"),
        "stacks": stacks,
    }


def _parse_district_list(value: object, *, label: str) -> list[DistrictPayload]:
    return [
        _parse_district_payload(entry, label=f"{label}[{index}]")
        for index, entry in enumerate(_require_list(value, label))
    ]


def _parse_deck_state_payload(value: object, *, label: str) -> DeckStatePayload:
    mapping = _require_mapping(value, label)
    return {
        "draw": _parse_string_list(mapping.get("draw"), label=f"{label}.draw"),
        "discard": _parse_string_list(mapping.get("discard"), label=f"{label}.discard"),
        "reshuffles": _require_int(mapping.get("reshuffles"), f"{label}.reshuffles"),
    }


def _parse_deck_view_payload(value: object, *, label: str) -> DeckViewPayload:
    mapping = _require_mapping(value, label)
    return {
        "drawCount": _require_int(mapping.get("drawCount"), f"{label}.drawCount"),
        "discard": _parse_string_list(mapping.get("discard"), label=f"{label}.discard"),
        "reshuffles": _require_int(mapping.get("reshuffles"), f"{label}.reshuffles"),
    }


def _parse_player_state(value: object, *, label: str) -> PlayerStatePayload:
    mapping = _require_mapping(value, label)
    return {
        "id": _require_player_id(mapping.get("id"), f"{label}.id"),
        "hand": _parse_string_list(mapping.get("hand"), label=f"{label}.hand"),
        "crowns": _parse_string_list(mapping.get("crowns"), label=f"{label}.crowns"),
        "resources": _parse_resource_pool(mapping.get("resources"), label=f"{label}.resources"),
    }


def _parse_player_state_list(value: object, *, label: str) -> list[PlayerStatePayload]:
    players = [
        _parse_player_state(entry, label=f"{label}[{index}]")
        for index, entry in enumerate(_require_list(value, label))
    ]
    _require_complete_player_set(players, label=label)
    return players


def _parse_observed_player(value: object, *, label: str) -> ObservedPlayerPayload:
    mapping = _require_mapping(value, label)
    return {
        "id": _require_player_id(mapping.get("id"), f"{label}.id"),
        "crowns": _parse_string_list(mapping.get("crowns"), label=f"{label}.crowns"),
        "resources": _parse_resource_pool(mapping.get("resources"), label=f"{label}.resources"),
        "hand": _parse_string_list(mapping.get("hand"), label=f"{label}.hand"),
        "handCount": _require_int(mapping.get("handCount"), f"{label}.handCount"),
        "handHidden": _require_bool(mapping.get("handHidden"), f"{label}.handHidden"),
    }


def _parse_observed_player_list(value: object, *, label: str) -> list[ObservedPlayerPayload]:
    players = [
        _parse_observed_player(entry, label=f"{label}[{index}]")
        for index, entry in enumerate(_require_list(value, label))
    ]
    _require_complete_player_set(players, label=label)
    return players


def _require_complete_player_set(
    players: Sequence[PlayerStatePayload | ObservedPlayerPayload],
    *,
    label: str,
) -> None:
    if len(players) != 2:
        raise RuntimeError(f"{label} must contain exactly two players, got {len(players)}.")
    player_ids: set[PlayerId] = {player["id"] for player in players}
    if player_ids != {"PlayerA", "PlayerB"}:
        raise RuntimeError(f"{label} must contain PlayerA and PlayerB, got {sorted(player_ids)!r}.")


def _parse_string_list(value: object, *, label: str) -> list[str]:
    return [_require_str(entry, f"{label}[{index}]") for index, entry in enumerate(_require_list(value, label))]


def _parse_suit_list(value: object, *, label: str) -> list[Suit]:
    return [_require_suit(entry, f"{label}[{index}]") for index, entry in enumerate(_require_list(value, label))]


def _require_mapping(value: object, label: str) -> JsonMapping:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object, got {type(value).__name__}.")
    return value


def _require_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list, got {type(value).__name__}.")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string, got {type(value).__name__}.")
    return value


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer, got {type(value).__name__}.")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{label} must be a boolean, got {type(value).__name__}.")
    return value


def _require_suit(value: object, label: str) -> Suit:
    if value not in _SUITS:
        raise RuntimeError(f"{label} must be a suit, got {value!r}.")
    return value


def _require_player_id(value: object, label: str) -> PlayerId:
    if value not in _PLAYER_IDS:
        raise RuntimeError(f"{label} must be PlayerA|PlayerB, got {value!r}.")
    return value


def _require_phase(value: object, label: str) -> GamePhase:
    if value not in _GAME_PHASES:
        raise RuntimeError(f"{label} must be a valid game phase, got {value!r}.")
    return value


def _require_action_id(value: object, label: str) -> ActionId:
    if value not in _ACTION_IDS:
        raise RuntimeError(f"{label} must be a valid action id, got {value!r}.")
    return value


def _require_bridge_command(value: object, label: str) -> BridgeCommandName:
    if value not in _BRIDGE_COMMANDS:
        raise RuntimeError(f"{label} must be a valid bridge command, got {value!r}.")
    return value


def _require_winner(value: object, label: str) -> Winner:
    if value not in ("PlayerA", "PlayerB", "Draw"):
        raise RuntimeError(f"{label} must be PlayerA|PlayerB|Draw, got {value!r}.")
    return value


def _require_winner_decider(value: object, label: str) -> WinnerDecider:
    if value not in _WINNER_DECIDERS:
        raise RuntimeError(f"{label} must be a valid winner decider, got {value!r}.")
    return value


def _require_literal[T: str](value: object, label: str, expected: T) -> T:
    if value != expected:
        raise RuntimeError(f"{label} must be {expected!r}, got {value!r}.")
    return expected
