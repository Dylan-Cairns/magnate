from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional

PlayerId = Literal["PlayerA", "PlayerB"]
Winner = Literal["PlayerA", "PlayerB", "Draw"]


@dataclass(frozen=True)
class KeyedAction:
    action_id: str
    action_key: str
    action: Dict[str, Any]


@dataclass(frozen=True)
class StateResult:
    state: Dict[str, Any]
    view: Dict[str, Any]
    terminal: bool


@dataclass(frozen=True)
class LegalActionsResult:
    actions: List[KeyedAction]
    active_player_id: PlayerId
    phase: str


@dataclass(frozen=True)
class ObservationResult:
    view: Dict[str, Any]
    legal_action_mask: Optional[List[str]]


@dataclass
class DecisionSample:
    seed: str
    turn: int
    phase: str
    active_player_id: PlayerId
    action_key: str
    action_id: str
    action_index: int
    observation: List[float]
    action_features: List[List[float]]
    winner: Winner
    reward: float

    def as_json(self) -> Dict[str, Any]:
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
            "winner": self.winner,
            "reward": self.reward,
        }


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object, got {type(value).__name__}.")
    return value

