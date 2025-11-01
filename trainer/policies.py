from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .behavior_cloning import BehaviorCloningModel, load_behavior_cloning_checkpoint
from .bridge_client import BridgeClient
from .encoding import _card_rank, encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .ppo_model import CandidateActorCritic, load_ppo_checkpoint
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
    worlds: int = 8
    rollouts: int = 1
    depth: int = 16
    max_root_actions: int = 6
    rollout_epsilon: float = 0.12

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


@dataclass
class DeterminizedSearchPolicy(Policy):
    config: SearchConfig
    name: str = "search"

    def __post_init__(self) -> None:
        self._heuristic_policy = HeuristicPolicy()
        self._random_policy = RandomLegalPolicy()
        self._sim_client: BridgeClient | None = None
        self._sim_env: MagnateBridgeEnv | None = None

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
            return legal_actions[0].action_key
        if state is None:
            raise ValueError("Search policy requires a serialized state payload.")

        root_player = _active_player_id(view)
        candidates = self._root_candidates(legal_actions)
        worlds = self._sample_worlds(state=state, view=view, root_player=root_player, rng=rng)

        best_action = candidates[0]
        best_score = float("-inf")
        best_prior = self._heuristic_policy.score_action(best_action)
        for action in candidates:
            score = self._evaluate_action_over_worlds(
                world_states=worlds,
                root_player=root_player,
                root_action_key=action.action_key,
                rng=rng,
            )
            prior = self._heuristic_policy.score_action(action)
            if (
                score > best_score
                or (
                    math.isclose(score, best_score, abs_tol=1e-9)
                    and (
                        prior > best_prior
                        or (
                            math.isclose(prior, best_prior, abs_tol=1e-9)
                            and action.action_key < best_action.action_key
                        )
                    )
                )
            ):
                best_score = score
                best_action = action
                best_prior = prior

        return best_action.action_key

    def close(self) -> None:
        if self._sim_client is not None:
            self._sim_client.close()
            self._sim_client = None
            self._sim_env = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return None

    def _root_candidates(self, legal_actions: Sequence[KeyedAction]) -> list[KeyedAction]:
        ranked = sorted(
            legal_actions,
            key=lambda action: (
                -self._heuristic_policy.score_action(action),
                action.action_key,
            ),
        )
        return ranked[: min(len(ranked), self.config.max_root_actions)]

    def _evaluate_action_over_worlds(
        self,
        world_states: Sequence[Dict[str, Any]],
        root_player: PlayerId,
        root_action_key: str,
        rng: random.Random,
    ) -> float:
        total = 0.0
        count = 0
        for world_state in world_states:
            for _ in range(self.config.rollouts):
                total += self._run_rollout(
                    world_state=world_state,
                    root_player=root_player,
                    root_action_key=root_action_key,
                    rng=rng,
                )
                count += 1
        if count == 0:
            return 0.0
        return total / float(count)

    def _run_rollout(
        self,
        world_state: Dict[str, Any],
        root_player: PlayerId,
        root_action_key: str,
        rng: random.Random,
    ) -> float:
        sim_env = self._simulator_env()
        step_result = sim_env.reset(
            serialized_state=copy.deepcopy(world_state),
            skip_advance_to_decision=True,
        )
        step_result = sim_env.step(action_key=root_action_key)

        depth = 0
        while not step_result.terminal and depth < self.config.depth:
            legal = sim_env.legal_actions()
            action_key = self._rollout_action_key(
                view=step_result.view,
                legal_actions=legal.actions,
                rng=rng,
            )
            step_result = sim_env.step(action_key=action_key)
            depth += 1

        if step_result.terminal:
            return _terminal_value(step_result.state, root_player)
        root_view = sim_env.observation(viewer_id=root_player).view
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

    def _sample_worlds(
        self,
        state: Mapping[str, Any],
        view: Mapping[str, Any],
        root_player: PlayerId,
        rng: random.Random,
    ) -> list[Dict[str, Any]]:
        opponent_player = "PlayerB" if root_player == "PlayerA" else "PlayerA"
        players_by_id = _player_views_by_id(view)
        root_view = players_by_id[root_player]
        opponent_view = players_by_id[opponent_player]
        root_hand = _as_card_list(root_view.get("hand"))
        opponent_hand_count = _as_int(opponent_view.get("handCount"))
        draw_count = _as_int(_as_mapping(view.get("deck")).get("drawCount"))

        known_cards = set(root_hand)
        known_cards.update(_as_card_list(_as_mapping(view.get("deck")).get("discard")))
        known_cards.update(_district_property_cards(view))
        hidden_pool = [card_id for card_id in PROPERTY_CARD_IDS if card_id not in known_cards]

        expected_hidden = opponent_hand_count + draw_count
        if len(hidden_pool) != expected_hidden:
            raise ValueError(
                "Determinization card accounting mismatch. "
                f"expected={expected_hidden}, actual={len(hidden_pool)}"
            )

        worlds: list[Dict[str, Any]] = []
        for _ in range(self.config.worlds):
            shuffled_hidden = list(hidden_pool)
            rng.shuffle(shuffled_hidden)
            opponent_hand = shuffled_hidden[:opponent_hand_count]
            draw_cards = shuffled_hidden[opponent_hand_count : opponent_hand_count + draw_count]

            world_state = copy.deepcopy(dict(state))
            _replace_player_hand(world_state, root_player, root_hand)
            _replace_player_hand(world_state, opponent_player, opponent_hand)
            deck = _as_mapping(world_state.get("deck"))
            deck["draw"] = draw_cards
            worlds.append(world_state)
        return worlds

    def _simulator_env(self) -> MagnateBridgeEnv:
        if self._sim_env is None:
            self._sim_client = BridgeClient()
            self._sim_env = MagnateBridgeEnv(client=self._sim_client)
        return self._sim_env


@dataclass
class BehaviorCloningPolicy(Policy):
    model: BehaviorCloningModel
    checkpoint_path: str = ""
    name: str = "behavior-cloned"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str:
        del rng  # deterministic action selection from the trained checkpoint
        del state
        if not legal_actions:
            raise ValueError("Behavior-cloned policy requires at least one legal action.")

        observation_vector = encode_observation(view)
        action_vectors = encode_action_candidates(legal_actions)
        action_index = self.model.choose_action_index(observation_vector, action_vectors)
        return legal_actions[action_index].action_key


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
) -> Policy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomLegalPolicy()
    if normalized == "heuristic":
        return HeuristicPolicy()
    if normalized == "search":
        return DeterminizedSearchPolicy(config=search_config or SearchConfig())
    if normalized in ("bc", "behavior-cloned", "behavior_cloned"):
        if checkpoint_path is None:
            raise ValueError("Policy 'bc' requires a checkpoint path.")
        path = Path(checkpoint_path)
        model = load_behavior_cloning_checkpoint(path)
        return BehaviorCloningPolicy(
            model=model,
            checkpoint_path=str(path),
            name=f"behavior-cloned:{path.name}",
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
        f"Unknown policy name: {name!r}. Expected one of: random, heuristic, search, bc, ppo."
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
            continue
        player_id = player.get("id")
        if player_id in ("PlayerA", "PlayerB"):
            out[player_id] = player
    if "PlayerA" not in out or "PlayerB" not in out:
        raise ValueError("View payload is missing one or more players.")
    return out


def _district_property_cards(view: Mapping[str, Any]) -> set[str]:
    cards: set[str] = set()
    districts = view.get("districts")
    if not isinstance(districts, list):
        return cards

    for district in districts:
        if not isinstance(district, dict):
            continue
        stacks = district.get("stacks")
        if not isinstance(stacks, dict):
            continue
        for player_id in ("PlayerA", "PlayerB"):
            stack = stacks.get(player_id)
            if not isinstance(stack, dict):
                continue
            for card_id in _as_card_list(stack.get("developed")):
                cards.add(card_id)
            deed = stack.get("deed")
            if isinstance(deed, dict):
                deed_card = deed.get("cardId")
                if isinstance(deed_card, str):
                    cards.add(deed_card)
    return cards


def _replace_player_hand(state: Dict[str, Any], player_id: PlayerId, hand: Sequence[str]) -> None:
    players = state.get("players")
    if not isinstance(players, list):
        raise ValueError("Serialized state is missing players list.")
    for player in players:
        if isinstance(player, dict) and player.get("id") == player_id:
            player["hand"] = list(hand)
            return
    raise ValueError(f"Serialized state is missing player {player_id}.")


def _terminal_value(state: Mapping[str, Any], root_player: PlayerId) -> float:
    final_score = state.get("finalScore")
    if not isinstance(final_score, dict):
        return 0.0
    winner = final_score.get("winner")
    if winner == "Draw":
        return 0.0
    if winner == root_player:
        return 1.0
    if winner in ("PlayerA", "PlayerB"):
        return -1.0
    return 0.0


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
