from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .encoding import _card_rank, encode_action_candidates, encode_observation
from .search import (
    BridgeForwardModel,
    progressive_target_action_count,
    rank_root_actions,
    root_priors_by_key,
    sample_determinized_worlds,
    select_root_ucb_action,
    terminal_value,
)
from .td.checkpoint import load_opponent_checkpoint, load_value_checkpoint
from .types import KeyedAction, PlayerId


class Policy:
    name: str

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def root_action_probs(self) -> Mapping[str, float] | None:
        return None


@dataclass
class RandomLegalPolicy(Policy):
    name: str = "random"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        del view
        del state
        if not legal_actions:
            raise ValueError("Random policy requires at least one legal action.")
        return legal_actions[rng.randrange(len(legal_actions))].action_key


@dataclass
class HeuristicPolicy(Policy):
    name: str = "heuristic"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        del view
        del rng
        del state
        if not legal_actions:
            raise ValueError("Heuristic policy requires at least one legal action.")

        ranked = sorted(
            legal_actions,
            key=lambda action: (
                -self.score_action(action),
                action.action_key,
            ),
        )
        return ranked[0].action_key

    def score_action(self, action: KeyedAction) -> float:
        payload = action.action
        action_id = action.action_id
        score = {
            "develop-outright": 8.0,
            "develop-deed": 6.0,
            "buy-deed": 5.0,
            "choose-income-suit": 4.0,
            "trade": 2.0,
            "sell-card": 1.0,
            "end-turn": 0.0,
        }.get(action_id, 0.0)

        card_id = str(payload.get("cardId", ""))
        card_rank = _card_rank(card_id)

        if action_id in ("develop-outright", "develop-deed"):
            score += card_rank * 0.4
        if action_id == "buy-deed":
            score += card_rank * 0.25
            if card_rank <= 2:
                score -= 1.5
        if action_id == "sell-card":
            score -= card_rank * 0.3
        if action_id == "trade":
            give = str(payload.get("give", ""))
            receive = str(payload.get("receive", ""))
            if give == receive:
                score -= 10.0
            else:
                score += 0.2
        return score


@dataclass(frozen=True)
class SearchConfig:
    worlds: int = 6
    rollouts: int = 1
    depth: int = 14
    max_root_actions: int = 6
    rollout_epsilon: float = 0.04

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


@dataclass(frozen=True)
class TDValuePolicyConfig:
    checkpoint_path: Path
    worlds: int = 8

    def __post_init__(self) -> None:
        if self.worlds <= 0:
            raise ValueError("TDValuePolicyConfig.worlds must be > 0.")


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


@dataclass
class DeterminizedSearchPolicy(Policy):
    config: SearchConfig
    name: str = "search"

    def __post_init__(self) -> None:
        self._heuristic_policy = HeuristicPolicy()
        self._random_policy = RandomLegalPolicy()
        self._forward_model = BridgeForwardModel(step_cache_limit=0)
        self._last_root_policy: Dict[str, float] | None = None

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        if not legal_actions:
            raise ValueError("Search policy requires at least one legal action.")
        if len(legal_actions) == 1:
            self._last_root_policy = {legal_actions[0].action_key: 1.0}
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("Search policy requires a serialized state payload.")

        root_player = _active_player_id(view)
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

        root_visits: Dict[str, int] = {action.action_key: 0 for action in ranked_actions}
        root_value_sum: Dict[str, float] = {action.action_key: 0.0 for action in ranked_actions}

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
        world_state: Dict[str, Any],
        root_player: PlayerId,
        root_action_key: str,
        rng: random.Random,
    ) -> float:
        step_result = self._forward_model.reset_state(world_state)
        step_result = self._forward_model.step(root_action_key)

        depth = 0
        while not step_result.terminal and depth < self.config.depth:
            legal = self._forward_model.legal_actions()
            action_key = self._rollout_action_key(
                view=step_result.view,
                legal_actions=legal.actions,
                rng=rng,
            )
            step_result = self._forward_model.step(action_key)
            depth += 1

        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        root_view = self._forward_model.observation(viewer_id=root_player).view
        return _value_from_player_view(root_view, root_player)

    def _rollout_action_key(
        self,
        view: Dict[str, Any],
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if rng.random() < self.config.rollout_epsilon:
            return self._random_policy.choose_action_key(view, legal_actions, rng)
        return self._heuristic_policy.choose_action_key(view, legal_actions, rng)

    def _root_priors_by_key(
        self,
        legal_actions: Sequence[KeyedAction],
    ) -> Dict[str, float]:
        return root_priors_by_key(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
        )

    def _sample_worlds(
        self,
        state: Mapping[str, Any],
        view: Mapping[str, Any],
        root_player: PlayerId,
        rng: random.Random,
    ) -> list[Dict[str, Any]]:
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
    ) -> Dict[str, float]:
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


@dataclass
class TDValuePolicy(Policy):
    config: TDValuePolicyConfig
    name: str = "td-value"

    def __post_init__(self) -> None:
        model, _payload = load_value_checkpoint(path=self.config.checkpoint_path)
        self._model = model
        self._model.eval()
        self._forward_model = BridgeForwardModel(step_cache_limit=0)
        self._last_root_policy: Dict[str, float] | None = None

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        if not legal_actions:
            raise ValueError("TD value policy requires at least one legal action.")
        if len(legal_actions) == 1:
            self._last_root_policy = {legal_actions[0].action_key: 1.0}
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("TD value policy requires a serialized state payload.")

        root_player = _active_player_id(view)
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

        scores_by_key: Dict[str, float] = {}
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
        world_state: Mapping[str, Any],
        action_key: str,
        root_player: PlayerId,
    ) -> float:
        step_result = self._forward_model.reset_state(world_state)
        del step_result
        step_result = self._forward_model.step(action_key)
        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        root_view = self._forward_model.observation(viewer_id=root_player).view
        observation = torch.tensor(
            encode_observation(root_view),
            dtype=torch.float32,
        )
        with torch.no_grad():
            value = float(self._model(observation).item())
        return max(-1.0, min(1.0, value))

    def _distribution_from_scores(
        self,
        *,
        legal_actions: Sequence[KeyedAction],
        scores_by_key: Mapping[str, float],
    ) -> Dict[str, float]:
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
            ),
            name="td-search",
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        value_model, _payload = load_value_checkpoint(path=self._td_config.value_checkpoint_path)
        self._value_model = value_model
        self._value_model.eval()

        opponent_model, _opponent_payload = load_opponent_checkpoint(
            path=self._td_config.opponent_checkpoint_path
        )
        self._opponent_model = opponent_model
        self._opponent_model.eval()

    def close(self) -> None:
        super().close()
        self._opponent_model = None
        self._value_model = None

    def _run_rollout(
        self,
        world_state: Dict[str, Any],
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
            active_player = _active_player_id(step_result.view)
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

        root_view = self._forward_model.observation(viewer_id=root_player).view
        observation = torch.tensor(encode_observation(root_view), dtype=torch.float32)
        with torch.no_grad():
            value = float(self._value_model(observation).item())
        return max(-1.0, min(1.0, value))

    def _opponent_rollout_action_key(
        self,
        *,
        view: Dict[str, Any],
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if rng.random() < self.config.rollout_epsilon:
            return self._random_policy.choose_action_key(view, legal_actions, rng)

        observation = encode_observation(view)
        action_features = encode_action_candidates(legal_actions)
        with torch.no_grad():
            distribution = self._opponent_model.action_distribution(
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


def policy_from_name(
    name: str,
    *,
    search_config: SearchConfig | None = None,
    td_value_config: TDValuePolicyConfig | None = None,
    td_search_config: TDSearchPolicyConfig | None = None,
) -> Policy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomLegalPolicy()
    if normalized == "heuristic":
        return HeuristicPolicy()
    if normalized == "search":
        return DeterminizedSearchPolicy(config=search_config or SearchConfig())
    if normalized == "td-value":
        if td_value_config is None:
            raise ValueError("td-value policy requires td_value_config with checkpoint path.")
        return TDValuePolicy(config=td_value_config)
    if normalized == "td-search":
        if td_search_config is None:
            raise ValueError("td-search policy requires td_search_config with value checkpoint.")
        return TDDeterminizedSearchPolicy(config=td_search_config)
    raise ValueError(
        f"Unknown policy name: {name!r}. Expected one of: random, heuristic, search, td-value, td-search."
    )


def _active_player_id(view: Mapping[str, Any]) -> PlayerId:
    value = view.get("activePlayerId")
    if value not in ("PlayerA", "PlayerB"):
        raise ValueError(f"Invalid activePlayerId in view: {value!r}")
    return value


def _player_views_by_id(view: Mapping[str, Any]) -> Dict[PlayerId, Dict[str, Any]]:
    players = view.get("players")
    if not isinstance(players, list):
        raise ValueError("View payload is missing players list.")

    out: Dict[PlayerId, Dict[str, Any]] = {}
    for player in players:
        if not isinstance(player, dict):
            raise ValueError(f"Player entry must be an object, got {type(player).__name__}.")
        player_id = player.get("id")
        if player_id in ("PlayerA", "PlayerB"):
            out[player_id] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def _value_from_player_view(view: Mapping[str, Any], root_player: PlayerId) -> float:
    opponent = "PlayerB" if root_player == "PlayerA" else "PlayerA"
    players_by_id = _player_views_by_id(view)
    root_state = players_by_id[root_player]
    opponent_state = players_by_id[opponent]

    resource_root = _resource_total(root_state)
    resource_opponent = _resource_total(opponent_state)
    hand_diff = _as_int(root_state.get("handCount")) - _as_int(opponent_state.get("handCount"))

    districts = view.get("districts")
    if not isinstance(districts, list):
        raise ValueError("View payload is missing districts list.")
    district_count = len(districts)
    district_lead = 0.0
    rank_diff = 0.0
    progress_diff = 0.0
    for district in districts:
        if not isinstance(district, dict):
            raise ValueError(f"District entry must be an object, got {type(district).__name__}.")
        stacks = district.get("stacks")
        if not isinstance(stacks, dict):
            raise ValueError("District payload is missing stacks object.")
        root_stack = _as_mapping(stacks.get(root_player))
        opponent_stack = _as_mapping(stacks.get(opponent))
        root_rank, root_progress = _stack_score(root_stack)
        opponent_rank, opponent_progress = _stack_score(opponent_stack)
        rank_diff += root_rank - opponent_rank
        progress_diff += root_progress - opponent_progress
        if root_rank > opponent_rank:
            district_lead += 1.0
        elif root_rank < opponent_rank:
            district_lead -= 1.0

    district_term = district_lead / float(max(1, district_count))
    rank_term = math.tanh(rank_diff / 18.0)
    progress_term = math.tanh(progress_diff / 8.0)
    resource_term = math.tanh((resource_root - resource_opponent) / 10.0)
    hand_term = math.tanh(hand_diff / 4.0)

    score = (
        (0.55 * district_term)
        + (0.2 * rank_term)
        + (0.1 * progress_term)
        + (0.1 * resource_term)
        + (0.05 * hand_term)
    )
    return max(-1.0, min(1.0, score))


def _stack_score(stack: Mapping[str, Any]) -> tuple[float, float]:
    developed = _as_card_list(stack.get("developed"))
    developed_rank = sum(_card_rank(card_id) for card_id in developed)

    deed = _as_optional_mapping(stack.get("deed"))
    deed_card = str(deed.get("cardId", ""))
    deed_rank = _card_rank(deed_card)
    deed_progress = _as_optional_int(deed.get("progress"), default=0)

    progress_ratio = 0.0
    if deed_card and deed_rank > 0:
        progress_ratio = min(1.0, deed_progress / float(deed_rank))

    return developed_rank, progress_ratio


def _resource_total(player_state: Mapping[str, Any]) -> int:
    resources = player_state.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("Player payload is missing resources object.")
    total = 0
    for value in resources.values():
        total += _as_int(value)
    return total


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object mapping, got {type(value).__name__}.")


def _as_optional_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return _as_mapping(value)


def _as_card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list of card ids, got {type(value).__name__}.")
    out: list[str] = []
    for card_id in value:
        if not isinstance(card_id, str):
            raise ValueError(f"Expected card id string, got {type(card_id).__name__}.")
        out.append(card_id)
    return out


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Expected integer value, got bool.")
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Expected numeric value, got {type(value).__name__}.")


def _as_optional_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return _as_int(value)


def _safe_div(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / float(count)
