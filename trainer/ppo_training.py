from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from .encoding import encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .policies import HeuristicPolicy, RandomLegalPolicy
from .ppo_model import CandidateActorCritic
from .types import PlayerId, Winner

UpdateCallback = Callable[[int, "PpoUpdateSummary"], None]
OpponentKind = str


@dataclass(frozen=True)
class PpoConfig:
    total_episodes: int = 1024
    episodes_per_update: int = 32
    seed: int = 0
    gamma: float = 0.995
    clip_ratio: float = 0.2
    train_epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    max_decisions_per_game: int = 2000
    self_play_weight: float = 0.5
    heuristic_weight: float = 0.3
    random_weight: float = 0.2


@dataclass(frozen=True)
class PpoUpdateSummary:
    update: int
    start_episode: int
    end_episode: int
    episodes: int
    winners: Dict[Winner, int]
    opponent_counts: Dict[OpponentKind, int]
    average_turn: float
    average_transitions: float
    policy_loss: float
    value_loss: float
    entropy: float


@dataclass(frozen=True)
class PpoTrainingSummary:
    total_episodes: int
    updates: int
    winners: Dict[Winner, int]
    opponent_counts: Dict[OpponentKind, int]
    average_turn: float
    average_transitions: float
    update_summaries: List[PpoUpdateSummary]


@dataclass
class _Transition:
    observation: List[float]
    action_features: List[List[float]]
    action_index: int
    old_log_prob: float
    old_value: float
    player_id: PlayerId
    ret: float = 0.0
    advantage: float = 0.0


def train_ppo_policy(
    env: MagnateBridgeEnv,
    model: CandidateActorCritic,
    config: PpoConfig,
    seed_prefix: str,
    start_episode_index: int = 0,
    optimizer: Any | None = None,
    on_update_end: UpdateCallback | None = None,
) -> Tuple[Any, PpoTrainingSummary]:
    _validate_config(config)
    model.train()

    optim = optimizer or torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = random.Random(config.seed + start_episode_index)

    updates = max(1, math.ceil(config.total_episodes / config.episodes_per_update))
    update_summaries: List[PpoUpdateSummary] = []

    global_winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    global_opponents: Dict[OpponentKind, int] = {"self": 0, "heuristic": 0, "random": 0}
    global_turn_total = 0
    global_transition_total = 0
    episode_cursor = start_episode_index

    for update_index in range(updates):
        remaining = config.total_episodes - (update_index * config.episodes_per_update)
        episodes_this_update = min(config.episodes_per_update, remaining)
        if episodes_this_update <= 0:
            break

        transitions: List[_Transition] = []
        winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
        opponent_counts: Dict[OpponentKind, int] = {"self": 0, "heuristic": 0, "random": 0}
        turn_total = 0
        transition_total = 0
        start_episode = episode_cursor

        for local_episode in range(episodes_this_update):
            episode_index = episode_cursor + local_episode
            seed = f"{seed_prefix}-{episode_index}"
            first_player: PlayerId = "PlayerA" if episode_index % 2 == 0 else "PlayerB"
            learner_player: PlayerId = "PlayerA" if ((episode_index // 2) % 2 == 0) else "PlayerB"
            opponent = _sample_opponent_kind(config, rng)

            winner, turn, episode_transitions = _collect_episode(
                env=env,
                model=model,
                config=config,
                seed=seed,
                first_player=first_player,
                learner_player=learner_player,
                opponent=opponent,
                rng=rng,
            )
            transitions.extend(episode_transitions)
            winners[winner] += 1
            opponent_counts[opponent] += 1
            turn_total += turn
            transition_total += len(episode_transitions)

        _normalize_advantages(transitions)
        policy_loss, value_loss, entropy = _ppo_optimize(
            model=model,
            optimizer=optim,
            transitions=transitions,
            config=config,
            rng=rng,
        )

        end_episode = start_episode + episodes_this_update - 1
        update_summary = PpoUpdateSummary(
            update=update_index,
            start_episode=start_episode,
            end_episode=end_episode,
            episodes=episodes_this_update,
            winners=winners,
            opponent_counts=opponent_counts,
            average_turn=(turn_total / episodes_this_update) if episodes_this_update > 0 else 0.0,
            average_transitions=(transition_total / episodes_this_update) if episodes_this_update > 0 else 0.0,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
        )
        update_summaries.append(update_summary)
        if on_update_end is not None:
            on_update_end(update_index, update_summary)

        for key in global_winners:
            global_winners[key] += winners[key]
        for key in global_opponents:
            global_opponents[key] += opponent_counts[key]
        global_turn_total += turn_total
        global_transition_total += transition_total
        episode_cursor += episodes_this_update

    total_episodes = config.total_episodes
    summary = PpoTrainingSummary(
        total_episodes=total_episodes,
        updates=len(update_summaries),
        winners=global_winners,
        opponent_counts=global_opponents,
        average_turn=(global_turn_total / total_episodes) if total_episodes > 0 else 0.0,
        average_transitions=(global_transition_total / total_episodes) if total_episodes > 0 else 0.0,
        update_summaries=update_summaries,
    )
    return optim, summary


def _collect_episode(
    env: MagnateBridgeEnv,
    model: CandidateActorCritic,
    config: PpoConfig,
    seed: str,
    first_player: PlayerId,
    learner_player: PlayerId,
    opponent: OpponentKind,
    rng: random.Random,
) -> Tuple[Winner, int, List[_Transition]]:
    step_result = env.reset(seed=seed, first_player=first_player)
    transitions: List[_Transition] = []
    fixed_opponent_policy = None
    if opponent == "heuristic":
        fixed_opponent_policy = HeuristicPolicy()
    elif opponent == "random":
        fixed_opponent_policy = RandomLegalPolicy()

    while not step_result.terminal:
        legal = env.legal_actions()
        active_player = legal.active_player_id
        use_model = (opponent == "self") or (active_player == learner_player)
        if use_model:
            observation = encode_observation(step_result.view)
            action_features = encode_action_candidates(legal.actions)
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            action_tensor = torch.tensor(action_features, dtype=torch.float32)
            with torch.no_grad():
                distribution = model.action_distribution(
                    observation=obs_tensor,
                    action_features=action_tensor,
                    temperature=config.temperature,
                )
                sampled_index = int(distribution.sample().item())
                log_prob = float(
                    distribution.log_prob(torch.tensor(sampled_index, dtype=torch.int64)).item()
                )
                value = float(model.value_tensor(obs_tensor).item())

            transitions.append(
                _Transition(
                    observation=observation,
                    action_features=action_features,
                    action_index=sampled_index,
                    old_log_prob=log_prob,
                    old_value=value,
                    player_id=active_player,
                )
            )
            action_key = legal.actions[sampled_index].action_key
        else:
            if fixed_opponent_policy is None:
                raise RuntimeError(f"Opponent policy was not initialized for kind {opponent!r}.")
            action_key = fixed_opponent_policy.choose_action_key(
                step_result.view,
                legal.actions,
                rng,
            )

        step_result = env.step(action_key=action_key)
        if len(transitions) > config.max_decisions_per_game:
            raise RuntimeError(
                "Episode exceeded max_decisions_per_game guard "
                f"({config.max_decisions_per_game})."
            )

    winner = _winner_from_state(step_result.state)
    _attach_returns_and_advantages(transitions, winner, config.gamma)
    turn = int(step_result.state.get("turn", 0))
    return winner, turn, transitions


def _attach_returns_and_advantages(
    transitions: Sequence[_Transition],
    winner: Winner,
    gamma: float,
) -> None:
    index_by_player: Dict[PlayerId, List[int]] = {"PlayerA": [], "PlayerB": []}
    for index, transition in enumerate(transitions):
        index_by_player[transition.player_id].append(index)

    for player_id, indices in index_by_player.items():
        terminal_reward = _reward_for_player(winner, player_id)
        count = len(indices)
        for order, transition_index in enumerate(indices):
            remaining = (count - order - 1)
            discounted_return = terminal_reward * (gamma ** remaining)
            transition = transitions[transition_index]
            transition.ret = discounted_return
            transition.advantage = discounted_return - transition.old_value


def _normalize_advantages(transitions: Sequence[_Transition]) -> None:
    if not transitions:
        return
    values = [transition.advantage for transition in transitions]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    if std < 1e-8:
        std = 1.0

    for transition in transitions:
        transition.advantage = (transition.advantage - mean) / std


def _ppo_optimize(
    model: CandidateActorCritic,
    optimizer: Any,
    transitions: Sequence[_Transition],
    config: PpoConfig,
    rng: random.Random,
) -> Tuple[float, float, float]:
    if not transitions:
        return 0.0, 0.0, 0.0

    indices = list(range(len(transitions)))
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    for _ in range(config.train_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch_indices = indices[start : start + config.minibatch_size]
            if not batch_indices:
                continue

            batch_policy = []
            batch_value = []
            batch_entropy = []
            for index in batch_indices:
                transition = transitions[index]
                obs_tensor = torch.tensor(transition.observation, dtype=torch.float32)
                action_tensor = torch.tensor(transition.action_features, dtype=torch.float32)
                distribution = model.action_distribution(
                    observation=obs_tensor,
                    action_features=action_tensor,
                    temperature=config.temperature,
                )
                action_index = torch.tensor(transition.action_index, dtype=torch.int64)
                log_prob = distribution.log_prob(action_index)
                old_log_prob = torch.tensor(transition.old_log_prob, dtype=torch.float32)
                ratio = torch.exp(log_prob - old_log_prob)

                advantage = torch.tensor(transition.advantage, dtype=torch.float32)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - config.clip_ratio,
                    1.0 + config.clip_ratio,
                )
                surr_1 = ratio * advantage
                surr_2 = clipped_ratio * advantage
                batch_policy.append(-torch.min(surr_1, surr_2))

                value = model.value_tensor(obs_tensor)
                target_return = torch.tensor(transition.ret, dtype=torch.float32)
                batch_value.append((value - target_return) ** 2)
                batch_entropy.append(distribution.entropy())

            policy_loss = torch.stack(batch_policy).mean()
            value_loss = torch.stack(batch_value).mean()
            entropy = torch.stack(batch_entropy).mean()
            total_loss = (
                policy_loss
                + (config.value_loss_coef * value_loss)
                - (config.entropy_coef * entropy)
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    avg_policy = (sum(policy_losses) / len(policy_losses)) if policy_losses else 0.0
    avg_value = (sum(value_losses) / len(value_losses)) if value_losses else 0.0
    avg_entropy = (sum(entropies) / len(entropies)) if entropies else 0.0
    return avg_policy, avg_value, avg_entropy


def _sample_opponent_kind(config: PpoConfig, rng: random.Random) -> OpponentKind:
    options = [
        ("self", config.self_play_weight),
        ("heuristic", config.heuristic_weight),
        ("random", config.random_weight),
    ]
    total = sum(weight for _, weight in options)
    if total <= 0:
        raise ValueError("At least one opponent mix weight must be > 0.")

    draw = rng.random() * total
    cumulative = 0.0
    for label, weight in options:
        cumulative += weight
        if draw <= cumulative:
            return label
    return options[-1][0]


def _winner_from_state(state: Dict[str, object]) -> Winner:
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


def _validate_config(config: PpoConfig) -> None:
    if config.total_episodes < 0:
        raise ValueError("total_episodes must be >= 0.")
    if config.episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be > 0.")
    if config.gamma <= 0 or config.gamma > 1:
        raise ValueError("gamma must be in (0, 1].")
    if config.clip_ratio <= 0 or config.clip_ratio >= 1:
        raise ValueError("clip_ratio must be in (0, 1).")
    if config.train_epochs <= 0:
        raise ValueError("train_epochs must be > 0.")
    if config.minibatch_size <= 0:
        raise ValueError("minibatch_size must be > 0.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.value_loss_coef < 0:
        raise ValueError("value_loss_coef must be >= 0.")
    if config.entropy_coef < 0:
        raise ValueError("entropy_coef must be >= 0.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0.")
    if config.temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if config.max_decisions_per_game <= 0:
        raise ValueError("max_decisions_per_game must be > 0.")
    if config.self_play_weight < 0 or config.heuristic_weight < 0 or config.random_weight < 0:
        raise ValueError("Opponent mix weights must be >= 0.")
    if (config.self_play_weight + config.heuristic_weight + config.random_weight) <= 0:
        raise ValueError("At least one opponent mix weight must be > 0.")
