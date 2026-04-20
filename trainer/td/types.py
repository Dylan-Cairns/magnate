from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, Sequence, TypedDict

from trainer.types import PlayerId


@dataclass(frozen=True)
class ValueTransition:
    observation: Sequence[float]
    reward: float
    done: bool
    next_observation: Sequence[float] | None
    player_id: PlayerId
    episode_id: str | None = None
    timestep: int | None = None


@dataclass(frozen=True)
class OpponentSample:
    observation: Sequence[float]
    action_features: Sequence[Sequence[float]]
    action_index: int
    player_id: PlayerId


class ValueTransitionPayload(TypedDict):
    observation: list[float]
    reward: float
    done: bool
    nextObservation: list[float] | None
    playerId: PlayerId
    episodeId: NotRequired[str]
    timestep: NotRequired[int]


class OpponentSamplePayload(TypedDict):
    observation: list[float]
    actionFeatures: list[list[float]]
    actionIndex: int
    playerId: PlayerId
