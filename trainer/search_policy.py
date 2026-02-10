from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .basic_policies import HeuristicPolicy, Policy, RandomLegalPolicy
from .bridge_payloads import PlayerViewPayload, SerializedStatePayload
from .encoding import encode_action_candidates, encode_observation
from .search import (
    BridgeForwardModel,
    active_player_id,
    active_value_to_root_value,
    progressive_target_action_count,
    rank_root_actions,
    root_priors_by_key,
    sample_determinized_worlds,
    select_root_ucb_action,
    terminal_value,
    value_from_player_view,
)
from .td.checkpoint import load_opponent_checkpoint, load_value_checkpoint
from .td.models import OpponentModel, ValueNet
from .types import KeyedAction, PlayerId


@dataclass(frozen=True)
class SearchConfig:
    worlds: int = 6
    rollouts: int = 1
    depth: int = 14
    max_root_actions: int = 6
    rollout_epsilon: float = 0.04
    transition_cache_limit: int = 0
    legal_actions_cache_limit: int = 0
    observation_cache_limit: int = 0

    def __post_init__(self) -> None:
        if self.worlds <= 0:
            raise ValueError("SearchConfig.worlds must be > 0.")
        if self.rollouts <= 0:
            raise ValueError("SearchConfig.rollouts must be > 0.")
        if self.depth <= 0:
            raise ValueError("SearchConfig.depth must be > 0.")
        if self.max_root_actions <= 0:
            raise ValueError("SearchConfig.max_root_actions must be > 0.")
        if self.rollout_epsilon < 0.0 or self.rollout_epsilon > 1.0:
            raise ValueError("SearchConfig.rollout_epsilon must be in [0, 1].")
        if self.transition_cache_limit < 0:
            raise ValueError("SearchConfig.transition_cache_limit must be >= 0.")
        if self.legal_actions_cache_limit < 0:
            raise ValueError("SearchConfig.legal_actions_cache_limit must be >= 0.")
        if self.observation_cache_limit < 0:
            raise ValueError("SearchConfig.observation_cache_limit must be >= 0.")


@dataclass(frozen=True)
class TDSearchPolicyConfig:
    value_checkpoint_path: Path
    opponent_checkpoint_path: Path
    worlds: int = 6
    rollouts: int = 1
    depth: int = 14
    max_root_actions: int = 6
    rollout_epsilon: float = 0.04
    opponent_temperature: float = 1.0
    sample_opponent_actions: bool = False
    transition_cache_limit: int = 0
    legal_actions_cache_limit: int = 0
    observation_cache_limit: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_checkpoint_path, Path):
            raise ValueError("TDSearchPolicyConfig.opponent_checkpoint_path must be provided.")
        if self.worlds <= 0:
            raise ValueError("TDSearchPolicyConfig.worlds must be > 0.")
        if self.rollouts <= 0:
            raise ValueError("TDSearchPolicyConfig.rollouts must be > 0.")
        if self.depth <= 0:
            raise ValueError("TDSearchPolicyConfig.depth must be > 0.")
        if self.max_root_actions <= 0:
            raise ValueError("TDSearchPolicyConfig.max_root_actions must be > 0.")
        if self.rollout_epsilon < 0.0 or self.rollout_epsilon > 1.0:
            raise ValueError("TDSearchPolicyConfig.rollout_epsilon must be in [0, 1].")
        if self.opponent_temperature <= 0.0:
            raise ValueError("TDSearchPolicyConfig.opponent_temperature must be > 0.")
        if self.transition_cache_limit < 0:
            raise ValueError("TDSearchPolicyConfig.transition_cache_limit must be >= 0.")
        if self.legal_actions_cache_limit < 0:
            raise ValueError("TDSearchPolicyConfig.legal_actions_cache_limit must be >= 0.")
        if self.observation_cache_limit < 0:
            raise ValueError("TDSearchPolicyConfig.observation_cache_limit must be >= 0.")


@dataclass
class DeterminizedSearchPolicy(Policy):
    config: SearchConfig
    name: str = "search"

    def __post_init__(self) -> None:
        self._heuristic_policy = HeuristicPolicy()
        self._random_policy = RandomLegalPolicy()
        self._forward_model = BridgeForwardModel(
            transition_cache_limit=self.config.transition_cache_limit,
            legal_actions_cache_limit=self.config.legal_actions_cache_limit,
            observation_cache_limit=self.config.observation_cache_limit,
        )
        self._last_root_policy: dict[str, float] | None = None

    def choose_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: SerializedStatePayload | None = None,
    ) -> str:
        if not legal_actions:
            raise ValueError("Search policy requires at least one legal action.")
        if len(legal_actions) == 1:
            self._last_root_policy = {legal_actions[0].action_key: 1.0}
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("Search policy requires a serialized state payload.")

        root_player = active_player_id(view)
        ranked_actions = self._ranked_root_actions(legal_actions=legal_actions)
        worlds = self._sample_worlds(state=state, view=view, root_player=root_player, rng=rng)
        root_prior_by_key = self._root_priors_by_key(legal_actions=legal_actions)

        if not worlds:
            raise RuntimeError(
                "Determinized search sampled zero worlds. "
                f"seedlessRootPlayer={root_player} "
                f"turn={state.get('turn')} phase={state.get('phase')}"
            )

        expanded_count = min(len(ranked_actions), self.config.max_root_actions)
        expanded_actions = ranked_actions[:expanded_count]
        expanded_keys = [action.action_key for action in expanded_actions]
        pending_unvisited = list(expanded_keys)

        visit_count = len(worlds) * self.config.rollouts
        root_budget = max(1, visit_count * max(1, self.config.max_root_actions))

        root_visits: dict[str, int] = {action.action_key: 0 for action in ranked_actions}
        root_value_sum: dict[str, float] = {action.action_key: 0.0 for action in ranked_actions}

        for visit_index in range(root_budget):
            target_count = progressive_target_action_count(
                total_actions=len(ranked_actions),
                initial_actions=self.config.max_root_actions,
                visits=visit_index,
            )
            while len(expanded_keys) < target_count:
                next_action = ranked_actions[len(expanded_keys)]
                expanded_keys.append(next_action.action_key)
                pending_unvisited.append(next_action.action_key)

            if pending_unvisited:
                action_key = pending_unvisited.pop(0)
            else:
                action_key = select_root_ucb_action(
                    action_keys=expanded_keys,
                    visits_by_key=root_visits,
                    value_sum_by_key=root_value_sum,
                    priors_by_key=root_prior_by_key,
                    total_visits=visit_index,
                    c_puct=1.0,
                )

            world_index = visit_index % len(worlds)
            score = self._run_rollout(
                world_state=worlds[world_index],
                root_player=root_player,
                root_action_key=action_key,
                rng=rng,
            )
            root_visits[action_key] += 1
            root_value_sum[action_key] += score

        best_action = expanded_keys[0]
        best_visits = root_visits[best_action]
        best_value = _safe_div(root_value_sum[best_action], best_visits)
        best_prior = root_prior_by_key.get(best_action, 0.0)
        for action_key in expanded_keys[1:]:
            visits = root_visits[action_key]
            value = _safe_div(root_value_sum[action_key], visits)
            prior = root_prior_by_key.get(action_key, 0.0)
            if (
                visits > best_visits
                or (
                    visits == best_visits
                    and (
                        value > best_value
                        or (
                            math.isclose(value, best_value, abs_tol=1e-9)
                            and (
                                prior > best_prior
                                or (
                                    math.isclose(prior, best_prior, abs_tol=1e-9)
                                    and action_key < best_action
                                )
                            )
                        )
                    )
                )
            ):
                best_action = action_key
                best_visits = visits
                best_value = value
                best_prior = prior

        self._last_root_policy = self._distribution_from_visits(
            legal_actions=legal_actions,
            root_visits=root_visits,
            chosen_action_key=best_action,
        )
        return best_action

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

    def _ranked_root_actions(
        self,
        legal_actions: Sequence[KeyedAction],
    ) -> list[KeyedAction]:
        return rank_root_actions(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
        )

    def _run_rollout(
        self,
        world_state: SerializedStatePayload,
        root_player: PlayerId,
        root_action_key: str,
        rng: random.Random,
    ) -> float:
        step_result = self._forward_model.transition_cached(world_state, root_action_key)
        current_state = step_result.state

        depth = 0
        while not step_result.terminal and depth < self.config.depth:
            legal = self._forward_model.legal_actions_cached(current_state)
            action_key = self._rollout_action_key(
                view=step_result.view,
                legal_actions=legal.actions,
                rng=rng,
            )
            step_result = self._forward_model.transition_cached(current_state, action_key)
            current_state = step_result.state
            depth += 1

        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        root_view = self._forward_model.observation_cached(
            current_state,
            viewer_id=root_player,
        ).view
        return value_from_player_view(root_view, root_player)

    def _rollout_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if rng.random() < self.config.rollout_epsilon:
            return self._random_policy.choose_action_key(view, legal_actions, rng)
        return self._heuristic_policy.choose_action_key(view, legal_actions, rng)

    def _root_priors_by_key(
        self,
        legal_actions: Sequence[KeyedAction],
    ) -> dict[str, float]:
        return root_priors_by_key(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
        )

    def _sample_worlds(
        self,
        state: SerializedStatePayload,
        view: PlayerViewPayload,
        root_player: PlayerId,
        rng: random.Random,
    ) -> list[SerializedStatePayload]:
        return sample_determinized_worlds(
            state=state,
            view=view,
            root_player=root_player,
            worlds=self.config.worlds,
            rng=rng,
        )

    def _distribution_from_visits(
        self,
        *,
        legal_actions: Sequence[KeyedAction],
        root_visits: Mapping[str, int],
        chosen_action_key: str,
    ) -> dict[str, float]:
        total = sum(root_visits.get(action.action_key, 0) for action in legal_actions)
        if total <= 0:
            return {
                action.action_key: (1.0 if action.action_key == chosen_action_key else 0.0)
                for action in legal_actions
            }
        return {
            action.action_key: root_visits.get(action.action_key, 0) / float(total)
            for action in legal_actions
        }


class TDDeterminizedSearchPolicy(DeterminizedSearchPolicy):
    name: str = "td-search"

    def __init__(self, config: TDSearchPolicyConfig) -> None:
        self._td_config = config
        super().__init__(
            config=SearchConfig(
                worlds=config.worlds,
                rollouts=config.rollouts,
                depth=config.depth,
                max_root_actions=config.max_root_actions,
                rollout_epsilon=config.rollout_epsilon,
                transition_cache_limit=config.transition_cache_limit,
                legal_actions_cache_limit=config.legal_actions_cache_limit,
                observation_cache_limit=config.observation_cache_limit,
            ),
            name="td-search",
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        value_model, _payload = load_value_checkpoint(path=self._td_config.value_checkpoint_path)
        self._value_model: ValueNet | None = value_model
        self._value_model.eval()

        opponent_model, _opponent_payload = load_opponent_checkpoint(
            path=self._td_config.opponent_checkpoint_path
        )
        self._opponent_model: OpponentModel | None = opponent_model
        self._opponent_model.eval()

    def close(self) -> None:
        super().close()
        self._opponent_model = None
        self._value_model = None

    def _run_rollout(
        self,
        world_state: SerializedStatePayload,
        root_player: PlayerId,
        root_action_key: str,
        rng: random.Random,
    ) -> float:
        step_result = self._forward_model.reset_state(world_state)
        del step_result
        step_result = self._forward_model.step(root_action_key)

        depth = 0
        while not step_result.terminal and depth < self.config.depth:
            legal = self._forward_model.legal_actions()
            active_player = active_player_id(step_result.view)
            if active_player == root_player:
                action_key = self._rollout_action_key(
                    view=step_result.view,
                    legal_actions=legal.actions,
                    rng=rng,
                )
            else:
                action_key = self._opponent_rollout_action_key(
                    view=step_result.view,
                    legal_actions=legal.actions,
                    rng=rng,
                )
            step_result = self._forward_model.step(action_key)
            depth += 1

        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        active_player = active_player_id(step_result.view)
        observation = torch.tensor(encode_observation(step_result.view), dtype=torch.float32)
        with torch.no_grad():
            active_value = float(self._require_value_model()(observation).item())
        root_value = active_value_to_root_value(
            active_value=active_value,
            active_player=active_player,
            root_player=root_player,
        )
        return max(-1.0, min(1.0, root_value))

    def _opponent_rollout_action_key(
        self,
        *,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if rng.random() < self.config.rollout_epsilon:
            return self._random_policy.choose_action_key(view, legal_actions, rng)

        observation = encode_observation(view)
        action_features = encode_action_candidates(legal_actions)
        with torch.no_grad():
            distribution = self._require_opponent_model().action_distribution(
                observation=observation,
                action_features=action_features,
                temperature=self._td_config.opponent_temperature,
            )
            if self._td_config.sample_opponent_actions:
                index = int(distribution.sample().item())
            else:
                index = int(torch.argmax(distribution.probs).item())
        if index < 0 or index >= len(legal_actions):
            raise RuntimeError(
                "Opponent model produced invalid action index. "
                f"index={index} legalActions={len(legal_actions)}"
            )
        return legal_actions[index].action_key

    def _require_value_model(self) -> ValueNet:
        if self._value_model is None:
            raise RuntimeError("TD value model is unavailable; the policy may already be closed.")
        return self._value_model

    def _require_opponent_model(self) -> OpponentModel:
        if self._opponent_model is None:
            raise RuntimeError("TD opponent model is unavailable; the policy may already be closed.")
        return self._opponent_model


def _safe_div(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / float(count)


__all__ = [
    "SearchConfig",
    "TDSearchPolicyConfig",
    "DeterminizedSearchPolicy",
    "TDDeterminizedSearchPolicy",
]
