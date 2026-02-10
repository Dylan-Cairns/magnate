from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeVar

from trainer.bridge_client import BridgeClient
from trainer.bridge_payloads import PlayerId, SerializedStatePayload
from trainer.env import MagnateBridgeEnv
from trainer.types import LegalActionsResult, ObservationResult, StateResult

_ACTIVE_VIEWER_CACHE_KEY = "__active__"
_CacheKeyT = TypeVar("_CacheKeyT")
_CacheValueT = TypeVar("_CacheValueT")


@dataclass(frozen=True)
class ForwardModelCacheStats:
    transition_hits: int
    transition_misses: int
    legal_actions_hits: int
    legal_actions_misses: int
    observation_hits: int
    observation_misses: int
    transition_entries: int
    legal_actions_entries: int
    observation_entries: int


class BridgeForwardModel:
    def __init__(
        self,
        *,
        transition_cache_limit: int | None = None,
        legal_actions_cache_limit: int = 0,
        observation_cache_limit: int = 0,
        step_cache_limit: int | None = None,
    ) -> None:
        if transition_cache_limit is None:
            transition_cache_limit = 0 if step_cache_limit is None else step_cache_limit
        elif step_cache_limit is not None and step_cache_limit != transition_cache_limit:
            raise ValueError(
                "step_cache_limit and transition_cache_limit must match when both are provided."
            )
        self._sim_client: BridgeClient | None = None
        self._sim_env: MagnateBridgeEnv | None = None
        self._transition_cache: OrderedDict[tuple[str, str], StateResult] = OrderedDict()
        self._legal_actions_cache: OrderedDict[str, LegalActionsResult] = OrderedDict()
        self._observation_cache: OrderedDict[tuple[str, str], ObservationResult] = OrderedDict()
        self._transition_cache_limit = max(0, transition_cache_limit)
        self._legal_actions_cache_limit = max(0, legal_actions_cache_limit)
        self._observation_cache_limit = max(0, observation_cache_limit)
        self._transition_hits = 0
        self._transition_misses = 0
        self._legal_actions_hits = 0
        self._legal_actions_misses = 0
        self._observation_hits = 0
        self._observation_misses = 0

    @property
    def step_cache(self) -> OrderedDict[tuple[str, str], StateResult]:
        return self._transition_cache

    @property
    def transition_cache(self) -> OrderedDict[tuple[str, str], StateResult]:
        return self._transition_cache

    @property
    def legal_actions_cache(self) -> OrderedDict[str, LegalActionsResult]:
        return self._legal_actions_cache

    @property
    def observation_cache(self) -> OrderedDict[tuple[str, str], ObservationResult]:
        return self._observation_cache

    @property
    def transition_cache_limit(self) -> int:
        return self._transition_cache_limit

    @property
    def legal_actions_cache_limit(self) -> int:
        return self._legal_actions_cache_limit

    @property
    def observation_cache_limit(self) -> int:
        return self._observation_cache_limit

    def close(self) -> None:
        if self._sim_client is not None:
            self._sim_client.close()
            self._sim_client = None
            self._sim_env = None
        self.clear_caches()

    def clear_caches(self, *, reset_stats: bool = True) -> None:
        self._transition_cache.clear()
        self._legal_actions_cache.clear()
        self._observation_cache.clear()
        if reset_stats:
            self._transition_hits = 0
            self._transition_misses = 0
            self._legal_actions_hits = 0
            self._legal_actions_misses = 0
            self._observation_hits = 0
            self._observation_misses = 0

    def cache_stats(self) -> ForwardModelCacheStats:
        return ForwardModelCacheStats(
            transition_hits=self._transition_hits,
            transition_misses=self._transition_misses,
            legal_actions_hits=self._legal_actions_hits,
            legal_actions_misses=self._legal_actions_misses,
            observation_hits=self._observation_hits,
            observation_misses=self._observation_misses,
            transition_entries=len(self._transition_cache),
            legal_actions_entries=len(self._legal_actions_cache),
            observation_entries=len(self._observation_cache),
        )

    def reset_state(self, state: SerializedStatePayload) -> StateResult:
        return self._simulator_env().reset(
            serialized_state=copy.deepcopy(state),
            skip_advance_to_decision=True,
        )

    def legal_actions(self) -> LegalActionsResult:
        return self._simulator_env().legal_actions()

    def observation(self, viewer_id: PlayerId | None = None) -> ObservationResult:
        return self._simulator_env().observation(
            viewer_id=viewer_id,
            include_legal_action_mask=False,
        )

    def step(self, action_key: str) -> StateResult:
        return self._simulator_env().step(action_key=action_key)

    def transition_cached(self, state: SerializedStatePayload, action_key: str) -> StateResult:
        state_key = _state_cache_key(state)
        cache_key = (state_key, action_key)
        cached = self._transition_cache.get(cache_key)
        if cached is not None:
            self._transition_hits += 1
            self._transition_cache.move_to_end(cache_key)
            return copy.deepcopy(cached)

        self._transition_misses += 1
        self.reset_state(state)
        result = copy.deepcopy(self.step(action_key))
        if self._transition_cache_limit > 0:
            self._transition_cache[cache_key] = result
            self._transition_cache.move_to_end(cache_key)
            self._trim_cache(self._transition_cache, self._transition_cache_limit)
            return copy.deepcopy(result)
        return result

    def legal_actions_cached(self, state: SerializedStatePayload) -> LegalActionsResult:
        state_key = _state_cache_key(state)
        cached = self._legal_actions_cache.get(state_key)
        if cached is not None:
            self._legal_actions_hits += 1
            self._legal_actions_cache.move_to_end(state_key)
            return copy.deepcopy(cached)

        self._legal_actions_misses += 1
        self.reset_state(state)
        result = copy.deepcopy(self.legal_actions())
        if self._legal_actions_cache_limit > 0:
            self._legal_actions_cache[state_key] = result
            self._legal_actions_cache.move_to_end(state_key)
            self._trim_cache(self._legal_actions_cache, self._legal_actions_cache_limit)
            return copy.deepcopy(result)
        return result

    def observation_cached(
        self,
        state: SerializedStatePayload,
        viewer_id: PlayerId | None = None,
    ) -> ObservationResult:
        state_key = _state_cache_key(state)
        viewer_key = _ACTIVE_VIEWER_CACHE_KEY if viewer_id is None else viewer_id
        cache_key = (state_key, viewer_key)
        cached = self._observation_cache.get(cache_key)
        if cached is not None:
            self._observation_hits += 1
            self._observation_cache.move_to_end(cache_key)
            return copy.deepcopy(cached)

        self._observation_misses += 1
        self.reset_state(state)
        result = copy.deepcopy(self.observation(viewer_id=viewer_id))
        if self._observation_cache_limit > 0:
            self._observation_cache[cache_key] = result
            self._observation_cache.move_to_end(cache_key)
            self._trim_cache(self._observation_cache, self._observation_cache_limit)
            return copy.deepcopy(result)
        return result

    def step_state_cached(self, state: SerializedStatePayload, action_key: str) -> SerializedStatePayload:
        return self.transition_cached(state, action_key).state

    def _simulator_env(self) -> MagnateBridgeEnv:
        if self._sim_env is None:
            self._sim_client = BridgeClient()
            self._sim_env = MagnateBridgeEnv(client=self._sim_client)
        return self._sim_env

    def _trim_cache(
        self,
        cache: OrderedDict[_CacheKeyT, _CacheValueT],
        limit: int,
    ) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)


def _state_cache_key(state: SerializedStatePayload) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
