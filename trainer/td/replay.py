from __future__ import annotations

import random
from typing import Iterable, List, Sequence, TypeVar

from .types import OpponentSample, ValueTransition

T = TypeVar("T")


def _sample_items(
    *,
    items: Sequence[T],
    batch_size: int,
    rng: random.Random,
) -> list[T]:
    _indices, sampled = _sample_items_with_indices(
        items=items,
        batch_size=batch_size,
        rng=rng,
    )
    return sampled


def _sample_items_with_indices(
    *,
    items: Sequence[T],
    batch_size: int,
    rng: random.Random,
) -> tuple[list[int], list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if len(items) == 0:
        raise ValueError("Cannot sample from an empty replay buffer.")
    if batch_size >= len(items):
        indices = list(range(len(items)))
    else:
        indices = rng.sample(range(len(items)), k=batch_size)
    return indices, [items[index] for index in indices]


class ValueReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._capacity = capacity
        self._items: list[ValueTransition] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ValueTransition) -> None:
        if len(self._items) >= self._capacity:
            del self._items[0 : len(self._items) - self._capacity + 1]
        self._items.append(transition)

    def extend(self, transitions: Iterable[ValueTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, *, batch_size: int, rng: random.Random) -> list[ValueTransition]:
        return _sample_items(items=self._items, batch_size=batch_size, rng=rng)

    def sample_with_indices(
        self, *, batch_size: int, rng: random.Random
    ) -> tuple[list[int], list[ValueTransition]]:
        return _sample_items_with_indices(
            items=self._items,
            batch_size=batch_size,
            rng=rng,
        )

    def as_list(self) -> List[ValueTransition]:
        return list(self._items)


class OpponentReplayBuffer:
    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._capacity = capacity
        self._items: list[OpponentSample] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._items)

    def add(self, sample: OpponentSample) -> None:
        if len(self._items) >= self._capacity:
            del self._items[0 : len(self._items) - self._capacity + 1]
        self._items.append(sample)

    def extend(self, samples: Iterable[OpponentSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, *, batch_size: int, rng: random.Random) -> list[OpponentSample]:
        return _sample_items(items=self._items, batch_size=batch_size, rng=rng)

    def sample_with_indices(
        self, *, batch_size: int, rng: random.Random
    ) -> tuple[list[int], list[OpponentSample]]:
        return _sample_items_with_indices(
            items=self._items,
            batch_size=batch_size,
            rng=rng,
        )

    def as_list(self) -> List[OpponentSample]:
        return list(self._items)
