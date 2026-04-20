from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, TypedDict

from .bridge_payloads import (
    ActionId,
    GameActionPayload,
    GamePhase,
    PlayerId,
    PlayerViewPayload,
    SerializedStatePayload,
    Winner,
)


@dataclass(frozen=True)
class KeyedAction:
    action_id: ActionId
    action_key: str
    action: GameActionPayload


@dataclass(frozen=True)
class StateResult:
    state: SerializedStatePayload
    view: PlayerViewPayload
    terminal: bool


@dataclass(frozen=True)
class LegalActionsResult:
    actions: List[KeyedAction]
    active_player_id: PlayerId
    phase: GamePhase


@dataclass(frozen=True)
class ObservationResult:
    view: PlayerViewPayload
    legal_action_mask: Optional[List[str]]


@dataclass
class DecisionSample:
    seed: str
    turn: int
    phase: GamePhase
    active_player_id: PlayerId
    action_key: str
    action_id: ActionId
    action_index: int
    observation: List[float]
    action_features: List[List[float]]
    winner: Winner
    reward: float
    action_probs: Optional[List[float]] = None

    def as_json(self) -> "DecisionSamplePayload":
        return {
            "seed": self.seed,
            "turn": self.turn,
            "phase": self.phase,
            "activePlayerId": self.active_player_id,
            "actionKey": self.action_key,
            "actionId": self.action_id,
            "actionIndex": self.action_index,
            "observation": self.observation,
            "actionFeatures": self.action_features,
            "actionProbs": self.action_probs,
            "winner": self.winner,
            "reward": self.reward,
        }


class DecisionSamplePayload(TypedDict):
    seed: str
    turn: int
    phase: GamePhase
    activePlayerId: PlayerId
    actionKey: str
    actionId: ActionId
    actionIndex: int
    observation: list[float]
    actionFeatures: list[list[float]]
    actionProbs: list[float] | None
    winner: Winner
    reward: float


def require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object, got {type(value).__name__}.")
    return value
