from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List, Sequence, TypeVar

from .types import OpponentSample, ValueTransition

T = TypeVar("T")


def _sample_items(
    *,
    items: Sequence[T],
    batch_size: int,
    rng: random.Random,
) -> list[T]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if len(items) == 0:
        raise ValueError("Cannot sample from an empty replay buffer.")
    if batch_size >= len(items):
        return list(items)
    indices = rng.sample(range(len(items)), k=batch_size)
    return [items[index] for index in indices]


class ValueReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._capacity = capacity
        self._items: Deque[ValueTransition] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ValueTransition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Iterable[ValueTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, *, batch_size: int, rng: random.Random) -> list[ValueTransition]:
        return _sample_items(items=list(self._items), batch_size=batch_size, rng=rng)

    def as_list(self) -> List[ValueTransition]:
        return list(self._items)


class OpponentReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._capacity = capacity
        self._items: Deque[OpponentSample] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def add(self, sample: OpponentSample) -> None:
        self._items.append(sample)

    def extend(self, samples: Iterable[OpponentSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, *, batch_size: int, rng: random.Random) -> list[OpponentSample]:
        return _sample_items(items=list(self._items), batch_size=batch_size, rng=rng)

    def as_list(self) -> List[OpponentSample]:
        return list(self._items)
