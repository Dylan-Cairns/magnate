from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from trainer.encoding import encode_action_candidates, encode_observation
from trainer.env import MagnateBridgeEnv
from trainer.policies import Policy
from trainer.types import KeyedAction, PlayerId, Winner

from .types import OpponentSample, ValueTransition


@dataclass(frozen=True)
class SelfPlayEpisode:
    seed: str
    first_player: PlayerId
    winner: Winner
    turns: int
    value_transitions: list[ValueTransition]
    opponent_samples: list[OpponentSample]


def collect_self_play_episode(
    env: MagnateBridgeEnv,
    policies: Mapping[PlayerId, Policy],
    seed: str,
    first_player: PlayerId,
    rng: random.Random,
) -> SelfPlayEpisode:
    step_result = env.reset(seed=seed, first_player=first_player)

    pending_observation_by_player: Dict[PlayerId, Sequence[float] | None] = {
        "PlayerA": None,
        "PlayerB": None,
    }
    value_transitions: list[ValueTransition] = []
    opponent_samples: list[OpponentSample] = []

    while not step_result.terminal:
        legal = env.legal_actions()
        active_player = legal.active_player_id
        policy = policies[active_player]

        observation_vector = encode_observation(step_result.view)
        prior_observation = pending_observation_by_player[active_player]
        if prior_observation is not None:
            value_transitions.append(
                ValueTransition(
                    observation=prior_observation,
                    reward=0.0,
                    done=False,
                    next_observation=observation_vector,
                    player_id=active_player,
                )
            )

        action_features = encode_action_candidates(legal.actions)
        action_key = policy.choose_action_key(
            step_result.view,
            legal.actions,
            rng,
            state=step_result.state,
        )
        action_index = _find_action_index(legal.actions, action_key)
        opponent_samples.append(
            OpponentSample(
                observation=observation_vector,
                action_features=action_features,
                action_index=action_index,
                player_id=active_player,
            )
        )
        pending_observation_by_player[active_player] = observation_vector
        step_result = env.step(action_key=action_key)

    winner = _winner_from_state(step_result.state)
    for player_id, observation_vector in pending_observation_by_player.items():
        if observation_vector is None:
            continue
        value_transitions.append(
            ValueTransition(
                observation=observation_vector,
                reward=_terminal_reward(winner=winner, player_id=player_id),
                done=True,
                next_observation=None,
                player_id=player_id,
            )
        )

    return SelfPlayEpisode(
        seed=seed,
        first_player=first_player,
        winner=winner,
        turns=_as_int(step_result.state.get("turn")),
        value_transitions=value_transitions,
        opponent_samples=opponent_samples,
    )


def collect_self_play_games(
    env: MagnateBridgeEnv,
    policy_player_a: Policy,
    policy_player_b: Policy,
    games: int,
    seed_prefix: str,
) -> list[SelfPlayEpisode]:
    if games <= 0:
        raise ValueError("games must be > 0.")

    policies: Dict[PlayerId, Policy] = {
        "PlayerA": policy_player_a,
        "PlayerB": policy_player_b,
    }
    rng = random.Random(0)
    episodes: list[SelfPlayEpisode] = []
    for index in range(games):
        first_player: PlayerId = "PlayerA" if index % 2 == 0 else "PlayerB"
        seed = f"{seed_prefix}-{index}"
        episodes.append(
            collect_self_play_episode(
                env=env,
                policies=policies,
                seed=seed,
                first_player=first_player,
                rng=rng,
            )
        )
    return episodes


def flatten_value_transitions(episodes: Sequence[SelfPlayEpisode]) -> list[ValueTransition]:
    out: list[ValueTransition] = []
    for episode in episodes:
        out.extend(episode.value_transitions)
    return out


def flatten_opponent_samples(episodes: Sequence[SelfPlayEpisode]) -> list[OpponentSample]:
    out: list[OpponentSample] = []
    for episode in episodes:
        out.extend(episode.opponent_samples)
    return out


def _find_action_index(actions: Sequence[KeyedAction], action_key: str) -> int:
    for index, action in enumerate(actions):
        if action.action_key == action_key:
            return index
    raise ValueError(f"Selected action key was not present in legal actions: {action_key}")


def _winner_from_state(state: Mapping[str, Any]) -> Winner:
    final_score = state.get("finalScore")
    if isinstance(final_score, dict):
        winner = final_score.get("winner")
        if winner in ("PlayerA", "PlayerB", "Draw"):
            return winner
    return "Draw"


def _terminal_reward(*, winner: Winner, player_id: PlayerId) -> float:
    if winner == "Draw":
        return 0.0
    return 1.0 if winner == player_id else -1.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return 0
