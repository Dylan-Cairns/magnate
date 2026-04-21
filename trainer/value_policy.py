from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .basic_policies import Policy
from .bridge_payloads import PlayerViewPayload, SerializedStatePayload
from .encoding import encode_observation
from .search import (
    BridgeForwardModel,
    active_player_id,
    active_value_to_root_value,
    sample_determinized_worlds,
    terminal_value,
)
from .td.checkpoint import load_value_checkpoint
from .td.models import ValueNet
from .types import KeyedAction, PlayerId


@dataclass(frozen=True)
class TDValuePolicyConfig:
    checkpoint_path: Path
    worlds: int = 8

    def __post_init__(self) -> None:
        if self.worlds <= 0:
            raise ValueError("TDValuePolicyConfig.worlds must be > 0.")


@dataclass
class TDValuePolicy(Policy):
    config: TDValuePolicyConfig
    name: str = "td-value"

    def __post_init__(self) -> None:
        model, _payload = load_value_checkpoint(path=self.config.checkpoint_path)
        self._model: ValueNet = model
        self._model.eval()
        self._forward_model = BridgeForwardModel(step_cache_limit=0)
        self._last_root_policy: dict[str, float] | None = None

    def choose_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: SerializedStatePayload | None = None,
    ) -> str:
        if not legal_actions:
            raise ValueError("TD value policy requires at least one legal action.")
        if len(legal_actions) == 1:
            self._last_root_policy = {legal_actions[0].action_key: 1.0}
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("TD value policy requires a serialized state payload.")

        root_player = active_player_id(view)
        worlds = sample_determinized_worlds(
            state=state,
            view=view,
            root_player=root_player,
            worlds=self.config.worlds,
            rng=rng,
        )
        if not worlds:
            raise RuntimeError(
                "TD value policy sampled zero worlds. "
                f"rootPlayer={root_player} "
                f"turn={state.get('turn')} phase={state.get('phase')}"
            )

        scores_by_key: dict[str, float] = {}
        for action in legal_actions:
            total = 0.0
            for world in worlds:
                total += self._score_action_world(
                    world_state=world,
                    action_key=action.action_key,
                    root_player=root_player,
                )
            scores_by_key[action.action_key] = total / float(len(worlds))

        best_key = min(legal_actions, key=lambda action: action.action_key).action_key
        best_score = scores_by_key[best_key]
        for action in legal_actions:
            score = scores_by_key[action.action_key]
            if score > best_score or (
                math.isclose(score, best_score, abs_tol=1e-9)
                and action.action_key < best_key
            ):
                best_key = action.action_key
                best_score = score

        self._last_root_policy = self._distribution_from_scores(
            legal_actions=legal_actions,
            scores_by_key=scores_by_key,
        )
        return best_key

    def close(self) -> None:
        self._forward_model.close()
        self._last_root_policy = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return None

    def root_action_probs(self) -> Mapping[str, float] | None:
        if self._last_root_policy is None:
            return None
        return dict(self._last_root_policy)

    def _score_action_world(
        self,
        *,
        world_state: SerializedStatePayload,
        action_key: str,
        root_player: PlayerId,
    ) -> float:
        step_result = self._forward_model.reset_state(world_state)
        del step_result
        step_result = self._forward_model.step(action_key)
        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        active_player = active_player_id(step_result.view)
        observation = torch.tensor(
            encode_observation(step_result.view),
            dtype=torch.float32,
        )
        with torch.no_grad():
            active_value = float(self._require_model()(observation).item())
        root_value = active_value_to_root_value(
            active_value=active_value,
            active_player=active_player,
            root_player=root_player,
        )
        return max(-1.0, min(1.0, root_value))

    def _distribution_from_scores(
        self,
        *,
        legal_actions: Sequence[KeyedAction],
        scores_by_key: Mapping[str, float],
    ) -> dict[str, float]:
        if not legal_actions:
            return {}
        logits = [float(scores_by_key.get(action.action_key, 0.0)) for action in legal_actions]
        max_logit = max(logits)
        exp_values = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_values)
        if total <= 0.0:
            uniform = 1.0 / float(len(legal_actions))
            return {action.action_key: uniform for action in legal_actions}
        return {
            action.action_key: exp_values[index] / total
            for index, action in enumerate(legal_actions)
        }

    def _require_model(self) -> ValueNet:
        return self._model


__all__ = [
    "TDValuePolicyConfig",
    "TDValuePolicy",
]
