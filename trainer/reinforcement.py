from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from .behavior_cloning import BehaviorCloningModel
from .encoding import encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .types import PlayerId, Winner


@dataclass(frozen=True)
class ReinforceConfig:
    episodes: int = 200
    learning_rate: float = 0.01
    l2: float = 1e-5
    temperature: float = 1.0
    seed: int = 0
    max_decisions_per_game: int = 2000


@dataclass(frozen=True)
class ReinforceEpisodeResult:
    episode: int
    seed: str
    first_player: PlayerId
    winner: Winner
    turn: int
    decisions: int
    average_entropy: float


@dataclass(frozen=True)
class ReinforceSummary:
    episodes: int
    winners: Dict[Winner, int]
    average_turn: float
    average_decisions: float
    average_entropy: float
    history: List[ReinforceEpisodeResult]


@dataclass(frozen=True)
class _EpisodeDecision:
    observation: List[float]
    action_features: List[List[float]]
    chosen_index: int
    probs: List[float]
    active_player_id: PlayerId


def fine_tune_with_reinforce(
    env: MagnateBridgeEnv,
    model: BehaviorCloningModel,
    config: ReinforceConfig,
    seed_prefix: str,
) -> ReinforceSummary:
    _validate_config(config)

    rng = random.Random(config.seed)
    winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    history: List[ReinforceEpisodeResult] = []

    turn_total = 0
    decision_total = 0
    entropy_total = 0.0

    for episode_index in range(config.episodes):
        seed = f"{seed_prefix}-{episode_index}"
        first_player: PlayerId = "PlayerA" if episode_index % 2 == 0 else "PlayerB"
        result = _run_episode_with_update(
            env=env,
            model=model,
            config=config,
            seed=seed,
            first_player=first_player,
            rng=rng,
            episode_index=episode_index,
        )
        history.append(result)
        winners[result.winner] += 1
        turn_total += result.turn
        decision_total += result.decisions
        entropy_total += result.average_entropy

    episodes = config.episodes
    average_turn = (turn_total / episodes) if episodes > 0 else 0.0
    average_decisions = (decision_total / episodes) if episodes > 0 else 0.0
    average_entropy = (entropy_total / episodes) if episodes > 0 else 0.0

    return ReinforceSummary(
        episodes=episodes,
        winners=winners,
        average_turn=average_turn,
        average_decisions=average_decisions,
        average_entropy=average_entropy,
        history=history,
    )


def apply_reinforce_update(
    model: BehaviorCloningModel,
    observation: Sequence[float],
    action_features: Sequence[Sequence[float]],
    chosen_index: int,
    probs: Sequence[float],
    advantage: float,
    learning_rate: float,
    l2: float,
) -> None:
    if len(observation) != model.observation_dim:
        raise ValueError(
            "Observation length mismatch. "
            f"expected={model.observation_dim}, actual={len(observation)}"
        )
    if not action_features:
        raise ValueError("At least one candidate action is required.")
    if chosen_index < 0 or chosen_index >= len(action_features):
        raise ValueError(
            "Chosen action index out of bounds. "
            f"index={chosen_index}, candidates={len(action_features)}"
        )
    if len(probs) != len(action_features):
        raise ValueError(
            "Probability length mismatch. "
            f"expected={len(action_features)}, actual={len(probs)}"
        )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if l2 < 0:
        raise ValueError("l2 must be >= 0.")

    for features in action_features:
        if len(features) != model.action_feature_dim:
            raise ValueError(
                "Action feature length mismatch. "
                f"expected={model.action_feature_dim}, actual={len(features)}"
            )

    delta = _policy_gradient_delta(
        action_features=action_features,
        chosen_index=chosen_index,
        probs=probs,
        action_feature_dim=model.action_feature_dim,
    )
    scale = learning_rate * advantage
    decay = 1.0 - (learning_rate * l2)

    for feature_index in range(model.action_feature_dim):
        model.action_weights[feature_index] = (
            decay * model.action_weights[feature_index]
        ) + (scale * delta[feature_index])

    for obs_index, obs_value in enumerate(observation):
        row = model.obs_action_weights[obs_index]
        for feature_index in range(model.action_feature_dim):
            row[feature_index] = (decay * row[feature_index]) + (
                scale * obs_value * delta[feature_index]
            )


def sample_action_index(probs: Sequence[float], rng: random.Random) -> int:
    if not probs:
        raise ValueError("Cannot sample action from an empty probability list.")

    total = 0.0
    for value in probs:
        if value < 0.0:
            raise ValueError("Probabilities must be non-negative.")
        total += value
    if total <= 0.0:
        raise ValueError("Probability mass must be positive.")

    draw = rng.random() * total
    cumulative = 0.0
    for index, value in enumerate(probs):
        cumulative += value
        if draw <= cumulative:
            return index
    return len(probs) - 1


def softmax_with_temperature(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        raise ValueError("Softmax requires at least one score.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")

    scaled_scores = [score / temperature for score in scores]
    maximum = max(scaled_scores)
    exps = [math.exp(score - maximum) for score in scaled_scores]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exps]


def _run_episode_with_update(
    env: MagnateBridgeEnv,
    model: BehaviorCloningModel,
    config: ReinforceConfig,
    seed: str,
    first_player: PlayerId,
    rng: random.Random,
    episode_index: int,
) -> ReinforceEpisodeResult:
    step_result = env.reset(seed=seed, first_player=first_player)
    staged: List[_EpisodeDecision] = []
    entropy_total = 0.0

    while not step_result.terminal:
        legal = env.legal_actions()
        observation_vector = encode_observation(step_result.view)
        action_vectors = encode_action_candidates(legal.actions)
        scores = model.score_candidates(observation_vector, action_vectors)
        probs = softmax_with_temperature(scores, config.temperature)
        action_index = sample_action_index(probs, rng)
        action_key = legal.actions[action_index].action_key

        staged.append(
            _EpisodeDecision(
                observation=observation_vector,
                action_features=action_vectors,
                chosen_index=action_index,
                probs=probs,
                active_player_id=legal.active_player_id,
            )
        )
        entropy_total += _entropy(probs)

        step_result = env.step(action_key=action_key)
        if len(staged) > config.max_decisions_per_game:
            raise RuntimeError(
                "Episode exceeded max_decisions_per_game guard "
                f"({config.max_decisions_per_game})."
            )

    winner = _winner_from_state(step_result.state)
    _apply_episode_updates(model, staged, winner, config)

    decisions = len(staged)
    average_entropy = (entropy_total / decisions) if decisions > 0 else 0.0
    return ReinforceEpisodeResult(
        episode=episode_index,
        seed=seed,
        first_player=first_player,
        winner=winner,
        turn=int(step_result.state.get("turn", 0)),
        decisions=decisions,
        average_entropy=average_entropy,
    )


def _apply_episode_updates(
    model: BehaviorCloningModel,
    staged: Sequence[_EpisodeDecision],
    winner: Winner,
    config: ReinforceConfig,
) -> None:
    decisions_by_player: Dict[PlayerId, int] = {
        "PlayerA": 0,
        "PlayerB": 0,
    }
    for decision in staged:
        decisions_by_player[decision.active_player_id] += 1

    for decision in staged:
        reward = _reward_for_player(winner, decision.active_player_id)
        if reward == 0.0:
            continue

        player_steps = decisions_by_player[decision.active_player_id]
        if player_steps <= 0:
            continue
        advantage = reward / float(player_steps)

        apply_reinforce_update(
            model=model,
            observation=decision.observation,
            action_features=decision.action_features,
            chosen_index=decision.chosen_index,
            probs=decision.probs,
            advantage=advantage,
            learning_rate=config.learning_rate,
            l2=config.l2,
        )


def _policy_gradient_delta(
    action_features: Sequence[Sequence[float]],
    chosen_index: int,
    probs: Sequence[float],
    action_feature_dim: int,
) -> List[float]:
    delta = [0.0 for _ in range(action_feature_dim)]
    for candidate_index, features in enumerate(action_features):
        coeff = -probs[candidate_index]
        if candidate_index == chosen_index:
            coeff += 1.0
        for feature_index, value in enumerate(features):
            delta[feature_index] += coeff * value
    return delta


def _winner_from_state(state: Mapping[str, object]) -> Winner:
    final_score = state.get("finalScore")
    if isinstance(final_score, dict):
        winner = final_score.get("winner")
        if winner in ("PlayerA", "PlayerB", "Draw"):
            return winner
    return "Draw"


def _reward_for_player(winner: Winner, player_id: PlayerId) -> float:
    if winner == "Draw":
        return 0.0
    if winner == player_id:
        return 1.0
    return -1.0


def _entropy(probs: Sequence[float]) -> float:
    entropy = 0.0
    for value in probs:
        if value <= 0.0:
            continue
        entropy -= value * math.log(value)
    return entropy


def _validate_config(config: ReinforceConfig) -> None:
    if config.episodes < 0:
        raise ValueError("episodes must be >= 0.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.l2 < 0:
        raise ValueError("l2 must be >= 0.")
    if config.temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if config.max_decisions_per_game <= 0:
        raise ValueError("max_decisions_per_game must be > 0.")
