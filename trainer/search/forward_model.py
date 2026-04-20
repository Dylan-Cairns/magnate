from __future__ import annotations

import copy
import json
from collections import OrderedDict

from trainer.bridge_client import BridgeClient
from trainer.bridge_payloads import PlayerId, SerializedStatePayload
from trainer.env import MagnateBridgeEnv
from trainer.types import LegalActionsResult, ObservationResult, StateResult


class BridgeForwardModel:
    def __init__(self, *, step_cache_limit: int = 0) -> None:
        self._sim_client: BridgeClient | None = None
        self._sim_env: MagnateBridgeEnv | None = None
        self._step_cache: OrderedDict[tuple[str, str], SerializedStatePayload] = OrderedDict()
        self._step_cache_limit = max(0, step_cache_limit)

    @property
    def step_cache(self) -> OrderedDict[tuple[str, str], SerializedStatePayload]:
        return self._step_cache

    def close(self) -> None:
        if self._sim_client is not None:
            self._sim_client.close()
            self._sim_client = None
            self._sim_env = None
        self._step_cache.clear()

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

    def step_state_cached(self, state: SerializedStatePayload, action_key: str) -> SerializedStatePayload:
        if self._step_cache_limit <= 0:
            self.reset_state(state)
            return copy.deepcopy(self.step(action_key).state)

        state_key = _state_cache_key(state)
        cache_key = (state_key, action_key)
        cached = self._step_cache.get(cache_key)
        if cached is not None:
            self._step_cache.move_to_end(cache_key)
            return copy.deepcopy(cached)

        self.reset_state(state)
        next_state = copy.deepcopy(self.step(action_key).state)
        self._step_cache[cache_key] = next_state
        self._step_cache.move_to_end(cache_key)
        while len(self._step_cache) > self._step_cache_limit:
            self._step_cache.popitem(last=False)
        return copy.deepcopy(next_state)

    def _simulator_env(self) -> MagnateBridgeEnv:
        if self._sim_env is None:
            self._sim_client = BridgeClient()
            self._sim_env = MagnateBridgeEnv(client=self._sim_client)
        return self._sim_env


def _state_cache_key(state: SerializedStatePayload) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
