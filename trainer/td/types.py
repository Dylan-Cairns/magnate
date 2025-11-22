from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
