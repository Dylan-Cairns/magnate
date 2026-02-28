from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .encoding import _card_rank, encode_action_candidates, encode_observation
from .ppo_model import CandidateActorCritic, load_ppo_checkpoint
from .search import (
    BridgeForwardModel,
    LeafEvaluator,
    is_terminal_state,
    progressive_target_action_count,
    rank_root_actions,
    root_priors_by_key,
    sample_determinized_worlds,
    select_root_ucb_action,
    state_active_player_id,
    terminal_value,
)
from .types import KeyedAction, PlayerId

PROPERTY_CARD_IDS: tuple[str, ...] = tuple(str(card_id) for card_id in range(30))


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
    rollout_epsilon: float = 0.08

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
class MctsConfig:
    worlds: int = 6
    simulations: int = 96
    depth: int = 24
    max_root_actions: int = 8
    c_puct: float = 1.25

    def __post_init__(self) -> None:
        if self.worlds <= 0:
            raise ValueError("MctsConfig.worlds must be > 0.")
        if self.simulations <= 0:
            raise ValueError("MctsConfig.simulations must be > 0.")
        if self.depth <= 0:
            raise ValueError("MctsConfig.depth must be > 0.")
        if self.max_root_actions <= 0:
            raise ValueError("MctsConfig.max_root_actions must be > 0.")
        if self.c_puct <= 0.0:
            raise ValueError("MctsConfig.c_puct must be > 0.")


@dataclass
class _GuidanceModel:
    model: CandidateActorCritic
    temperature: float = 1.0

    def policy_probs(
        self,
        view: Mapping[str, Any],
        legal_actions: Sequence[KeyedAction],
    ) -> list[float]:
        if not legal_actions:
            return []
        observation_vector = encode_observation(view)
        action_vectors = encode_action_candidates(legal_actions)
        observation = torch.tensor(observation_vector, dtype=torch.float32)
        action_features = torch.tensor(action_vectors, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model.policy_logits_tensor(observation, action_features)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            probs = torch.softmax(logits, dim=-1).tolist()
        return [float(value) for value in probs]

    def choose_action_key(
        self,
        view: Mapping[str, Any],
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        deterministic: bool,
    ) -> str:
        if not legal_actions:
            raise ValueError("Guidance policy requires at least one legal action.")
        probs = self.policy_probs(view=view, legal_actions=legal_actions)
        if deterministic:
            ranked = sorted(
                range(len(legal_actions)),
                key=lambda index: (
                    -probs[index],
                    legal_actions[index].action_key,
                ),
            )
            return legal_actions[ranked[0]].action_key

        draw = rng.random()
        cumulative = 0.0
        for index, prob in enumerate(probs):
            cumulative += prob
            if draw <= cumulative:
                return legal_actions[index].action_key
        return legal_actions[-1].action_key

    def value_from_view(self, view: Mapping[str, Any]) -> float:
        observation_vector = encode_observation(view)
        observation = torch.tensor(observation_vector, dtype=torch.float32)
        with torch.no_grad():
            raw = float(self.model.value_tensor(observation).item())
        return max(-1.0, min(1.0, raw))


@dataclass
class DeterminizedSearchPolicy(Policy):
    config: SearchConfig
    guidance_model: _GuidanceModel | None = None
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
        ranked_actions = self._ranked_root_actions(view=view, legal_actions=legal_actions)
        worlds = self._sample_worlds(state=state, view=view, root_player=root_player, rng=rng)
        root_prior_by_key = self._root_priors_by_key(
            view=view,
            legal_actions=legal_actions,
        )

        if not worlds:
            fallback = ranked_actions[0].action_key
            self._last_root_policy = {fallback: 1.0}
            return fallback

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
        view: Mapping[str, Any],
        legal_actions: Sequence[KeyedAction],
    ) -> list[KeyedAction]:
        return rank_root_actions(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
            guidance_model=self.guidance_model,
            view=view,
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
                root_player=root_player,
            )
            step_result = self._forward_model.step(action_key)
            depth += 1

        if step_result.terminal:
            return terminal_value(step_result.state, root_player)

        if self.guidance_model is not None:
            active_player = _active_player_id(step_result.view)
            value = self.guidance_model.value_from_view(step_result.view)
            if active_player != root_player:
                value = -value
            return max(-1.0, min(1.0, value))

        root_view = self._forward_model.observation(viewer_id=root_player).view
        return _value_from_player_view(root_view, root_player)

    def _rollout_action_key(
        self,
        view: Dict[str, Any],
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        root_player: PlayerId,
    ) -> str:
        if rng.random() < self.config.rollout_epsilon:
            return self._random_policy.choose_action_key(view, legal_actions, rng)

        if self.guidance_model is not None:
            active_player = _active_player_id(view)
            deterministic = active_player == root_player
            return self.guidance_model.choose_action_key(
                view=view,
                legal_actions=legal_actions,
                rng=rng,
                deterministic=deterministic,
            )

        return self._heuristic_policy.choose_action_key(view, legal_actions, rng)

    def _root_priors_by_key(
        self,
        view: Mapping[str, Any],
        legal_actions: Sequence[KeyedAction],
    ) -> Dict[str, float]:
        return root_priors_by_key(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
            guidance_model=self.guidance_model,
            view=view,
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
class _MctsEdge:
    action_key: str
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: "_MctsNode | None" = None


@dataclass
class _MctsNode:
    state: Dict[str, Any]
    terminal: bool
    active_player: PlayerId | None
    edges: Dict[str, _MctsEdge]
    root_ranked_actions: list[KeyedAction] | None = None
    root_priors: Dict[str, float] | None = None
    expanded: bool = False


@dataclass
class DeterminizedMctsPolicy(Policy):
    config: MctsConfig
    guidance_model: _GuidanceModel | None = None
    name: str = "mcts"

    def __post_init__(self) -> None:
        self._heuristic_policy = HeuristicPolicy()
        self._forward_model = BridgeForwardModel(step_cache_limit=20_000)
        self._leaf_evaluator = LeafEvaluator(
            forward_model=self._forward_model,
            guidance_model=self.guidance_model,
            value_cache_limit=20_000,
        )
        self._step_cache = self._forward_model.step_cache
        self._value_cache = self._leaf_evaluator.value_cache
        self._last_root_policy: Dict[str, float] | None = None

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        if not legal_actions:
            raise ValueError("MCTS policy requires at least one legal action.")
        if len(legal_actions) == 1:
            self._last_root_policy = {legal_actions[0].action_key: 1.0}
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("MCTS policy requires a serialized state payload.")

        root_player = _active_player_id(view)
        root_candidates = self._ranked_root_actions(view=view, legal_actions=legal_actions)
        worlds = self._sample_worlds(state=state, view=view, root_player=root_player, rng=rng)
        root_prior_by_key = self._root_priors_by_key(view=view, legal_actions=legal_actions)

        aggregate_visits: Dict[str, int] = {action.action_key: 0 for action in root_candidates}
        aggregate_value_sum: Dict[str, float] = {action.action_key: 0.0 for action in root_candidates}
        for world_state in worlds:
            world_stats = self._run_world_search(
                world_state=world_state,
                root_player=root_player,
                root_candidates=root_candidates,
            )
            for action in root_candidates:
                visits, value_sum = world_stats.get(action.action_key, (0, 0.0))
                aggregate_visits[action.action_key] += visits
                aggregate_value_sum[action.action_key] += value_sum

        best_action = root_candidates[0]
        best_visits = aggregate_visits[best_action.action_key]
        best_avg_value = _safe_div(
            aggregate_value_sum[best_action.action_key],
            aggregate_visits[best_action.action_key],
        )
        best_prior = root_prior_by_key.get(best_action.action_key, 0.0)
        for action in root_candidates[1:]:
            visits = aggregate_visits[action.action_key]
            avg_value = _safe_div(aggregate_value_sum[action.action_key], visits)
            prior = root_prior_by_key.get(action.action_key, 0.0)
            if (
                visits > best_visits
                or (
                    visits == best_visits
                    and (
                        avg_value > best_avg_value
                        or (
                            math.isclose(avg_value, best_avg_value, abs_tol=1e-9)
                            and (
                                prior > best_prior
                                or (
                                    math.isclose(prior, best_prior, abs_tol=1e-9)
                                    and action.action_key < best_action.action_key
                                )
                            )
                        )
                    )
                )
            ):
                best_action = action
                best_visits = visits
                best_avg_value = avg_value
                best_prior = prior

        self._last_root_policy = self._distribution_from_visits(
            legal_actions=legal_actions,
            root_visits=aggregate_visits,
            chosen_action_key=best_action.action_key,
        )
        return best_action.action_key

    def close(self) -> None:
        self._forward_model.close()
        self._leaf_evaluator.clear()
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
        view: Mapping[str, Any] | None = None,
    ) -> list[KeyedAction]:
        return rank_root_actions(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
            guidance_model=self.guidance_model,
            view=view,
        )

    def _run_world_search(
        self,
        world_state: Dict[str, Any],
        root_player: PlayerId,
        root_candidates: Sequence[KeyedAction],
    ) -> Dict[str, tuple[int, float]]:
        root_node = _MctsNode(
            state=copy.deepcopy(world_state),
            terminal=is_terminal_state(world_state),
            active_player=state_active_player_id(world_state),
            edges={},
            root_ranked_actions=list(root_candidates),
            expanded=False,
        )
        self._expand_node(node=root_node)
        if root_node.terminal or not root_node.edges:
            return {}

        for _ in range(self.config.simulations):
            self._simulate(node=root_node, root_player=root_player, depth_remaining=self.config.depth)

        stats: Dict[str, tuple[int, float]] = {}
        for action_key, edge in root_node.edges.items():
            stats[action_key] = (edge.visits, edge.value_sum)
        return stats

    def _simulate(self, node: _MctsNode, root_player: PlayerId, depth_remaining: int) -> float:
        if node.terminal:
            return terminal_value(node.state, root_player)
        if depth_remaining <= 0:
            return self._leaf_value(node.state, root_player)
        if not node.expanded:
            self._expand_node(node=node)
            if node.terminal:
                return terminal_value(node.state, root_player)
            return self._leaf_value(node.state, root_player)
        if not node.edges:
            return self._leaf_value(node.state, root_player)

        self._maybe_widen_root(node=node)
        edge = self._select_edge(node=node, root_player=root_player)
        if edge.child is None:
            child_state = self._step_state(node.state, edge.action_key)
            edge.child = _MctsNode(
                state=child_state,
                terminal=is_terminal_state(child_state),
                active_player=state_active_player_id(child_state),
                edges={},
                expanded=False,
            )
        value = self._simulate(node=edge.child, root_player=root_player, depth_remaining=depth_remaining - 1)
        edge.visits += 1
        edge.value_sum += value
        return value

    def _select_edge(self, node: _MctsNode, root_player: PlayerId) -> _MctsEdge:
        parent_visits = sum(edge.visits for edge in node.edges.values())
        sqrt_parent = math.sqrt(float(parent_visits) + 1.0)
        root_turn = node.active_player == root_player

        best_edge: _MctsEdge | None = None
        best_score = float("-inf")
        for edge in node.edges.values():
            q = _safe_div(edge.value_sum, edge.visits)
            exploit = q if root_turn else -q
            explore = self.config.c_puct * edge.prior * sqrt_parent / (1.0 + float(edge.visits))
            score = exploit + explore
            if (
                score > best_score
                or (
                    math.isclose(score, best_score, abs_tol=1e-9)
                    and best_edge is not None
                    and edge.action_key < best_edge.action_key
                )
                or best_edge is None
            ):
                best_edge = edge
                best_score = score

        if best_edge is None:
            raise RuntimeError("MCTS selection failed: node has no edges.")
        return best_edge

    def _expand_node(
        self,
        node: _MctsNode,
    ) -> None:
        if node.terminal:
            node.expanded = True
            return

        step_result = self._forward_model.reset_state(node.state)
        node.state = step_result.state
        if step_result.terminal:
            node.terminal = True
            node.active_player = None
            node.edges = {}
            node.expanded = True
            return

        legal = self._forward_model.legal_actions()
        legal_actions = legal.actions
        if not legal_actions:
            node.edges = {}
            node.active_player = legal.active_player_id
            node.expanded = True
            return

        priors = self._action_priors(
            legal_actions=legal_actions,
            view=step_result.view,
        )
        actions_to_expand = legal_actions
        if node.root_ranked_actions is not None:
            legal_by_key = {action.action_key: action for action in legal_actions}
            ranked_legal_actions = [
                legal_by_key[action.action_key]
                for action in node.root_ranked_actions
                if action.action_key in legal_by_key
            ]
            if not ranked_legal_actions:
                ranked_legal_actions = self._ranked_root_actions(
                    legal_actions=legal_actions,
                    view=step_result.view,
                )

            node.root_ranked_actions = ranked_legal_actions
            node.root_priors = priors
            initial_count = min(len(ranked_legal_actions), self.config.max_root_actions)
            actions_to_expand = ranked_legal_actions[:initial_count]

        node.edges = {
            action.action_key: _MctsEdge(
                action_key=action.action_key,
                prior=priors[action.action_key],
            )
            for action in actions_to_expand
        }
        node.active_player = legal.active_player_id
        node.expanded = True

    def _maybe_widen_root(self, node: _MctsNode) -> None:
        if node.root_ranked_actions is None or node.root_priors is None:
            return
        if not node.root_ranked_actions:
            return

        total_actions = len(node.root_ranked_actions)
        current_actions = len(node.edges)
        if current_actions >= total_actions:
            return

        parent_visits = sum(edge.visits for edge in node.edges.values())
        target_actions = min(
            total_actions,
            self.config.max_root_actions + int(math.sqrt(float(parent_visits) + 1.0)),
        )
        if target_actions <= current_actions:
            return

        for action in node.root_ranked_actions:
            if action.action_key in node.edges:
                continue
            node.edges[action.action_key] = _MctsEdge(
                action_key=action.action_key,
                prior=node.root_priors.get(action.action_key, 0.0),
            )
            if len(node.edges) >= target_actions:
                break

    def _action_priors(
        self,
        legal_actions: Sequence[KeyedAction],
        view: Mapping[str, Any] | None = None,
    ) -> Dict[str, float]:
        return root_priors_by_key(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
            guidance_model=self.guidance_model,
            view=view,
        )

    def _step_state(self, state: Mapping[str, Any], action_key: str) -> Dict[str, Any]:
        return self._forward_model.step_state_cached(state, action_key)

    def _leaf_value(
        self,
        state: Mapping[str, Any],
        root_player: PlayerId,
    ) -> float:
        return self._leaf_evaluator.value(state, root_player)

    def _root_priors_by_key(
        self,
        view: Mapping[str, Any],
        legal_actions: Sequence[KeyedAction],
    ) -> Dict[str, float]:
        return root_priors_by_key(
            legal_actions=legal_actions,
            heuristic_policy=self._heuristic_policy,
            guidance_model=self.guidance_model,
            view=view,
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
class TorchPpoPolicy(Policy):
    model: CandidateActorCritic
    checkpoint_path: str = ""
    deterministic: bool = True
    temperature: float = 1.0
    name: str = "ppo"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        del state
        if not legal_actions:
            raise ValueError("PPO policy requires at least one legal action.")

        observation_vector = encode_observation(view)
        action_vectors = encode_action_candidates(legal_actions)
        observation = torch.tensor(observation_vector, dtype=torch.float32)
        action_features = torch.tensor(action_vectors, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model.policy_logits_tensor(observation, action_features) / self.temperature
            if self.deterministic:
                action_index = int(torch.argmax(logits).item())
            else:
                probs = torch.softmax(logits, dim=-1).tolist()
                draw = rng.random()
                cumulative = 0.0
                action_index = 0
                for index, value in enumerate(probs):
                    cumulative += value
                    if draw <= cumulative:
                        action_index = index
                        break

        return legal_actions[action_index].action_key


def policy_from_name(
    name: str,
    checkpoint_path: str | Path | None = None,
    search_config: SearchConfig | None = None,
    mcts_config: MctsConfig | None = None,
    search_guidance_checkpoint: str | Path | None = None,
    mcts_guidance_checkpoint: str | Path | None = None,
    guidance_temperature: float = 1.0,
) -> Policy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomLegalPolicy()
    if normalized == "heuristic":
        return HeuristicPolicy()
    if normalized == "search":
        guidance_model = _load_guidance_model(
            checkpoint_path=search_guidance_checkpoint,
            temperature=guidance_temperature,
        )
        return DeterminizedSearchPolicy(
            config=search_config or SearchConfig(),
            guidance_model=guidance_model,
        )
    if normalized == "mcts":
        guidance_model = _load_guidance_model(
            checkpoint_path=mcts_guidance_checkpoint,
            temperature=guidance_temperature,
        )
        return DeterminizedMctsPolicy(
            config=mcts_config or MctsConfig(),
            guidance_model=guidance_model,
        )
    if normalized == "ppo":
        if checkpoint_path is None:
            raise ValueError("Policy 'ppo' requires a checkpoint path.")
        path = Path(checkpoint_path)
        model, _ = load_ppo_checkpoint(path)
        return TorchPpoPolicy(
            model=model,
            checkpoint_path=str(path),
            name=f"ppo:{path.name}",
        )
    raise ValueError(
        f"Unknown policy name: {name!r}. Expected one of: random, heuristic, search, mcts, ppo."
    )


def _load_guidance_model(
    checkpoint_path: str | Path | None,
    temperature: float,
) -> _GuidanceModel | None:
    if checkpoint_path is None:
        return None
    if temperature <= 0:
        raise ValueError("guidance_temperature must be > 0.")
    path = Path(checkpoint_path)
    model, _ = load_ppo_checkpoint(path)
    model.eval()
    return _GuidanceModel(model=model, temperature=temperature)


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
            continue
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
    district_count = len(districts) if isinstance(districts, list) else 0
    district_lead = 0.0
    rank_diff = 0.0
    progress_diff = 0.0
    if isinstance(districts, list):
        for district in districts:
            if not isinstance(district, dict):
                continue
            stacks = district.get("stacks")
            if not isinstance(stacks, dict):
                continue
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
    rank_total = 0.0
    progress_total = 0.0
    for card_id in _as_card_list(stack.get("developed")):
        rank_total += float(_card_rank(card_id))

    deed = stack.get("deed")
    if isinstance(deed, dict):
        rank_total += float(_card_rank(str(deed.get("cardId", ""))))
        progress = _as_int(deed.get("progress"))
        progress_total += float(progress)
        rank_total += 0.35 * float(progress)
    return rank_total, progress_total


def _resource_total(player_state: Mapping[str, Any]) -> int:
    resources = player_state.get("resources")
    if not isinstance(resources, dict):
        return 0
    total = 0
    for suit in ("Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots"):
        total += _as_int(resources.get(suit))
    return total


def _as_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected object mapping, got {type(value).__name__}.")


def _as_card_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for entry in value:
        if isinstance(entry, str):
            out.append(entry)
    return out


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _safe_div(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / float(count)
