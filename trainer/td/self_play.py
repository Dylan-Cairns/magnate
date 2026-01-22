from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence

from trainer.encoding import encode_action_candidates, encode_observation
from trainer.env import MagnateBridgeEnv
from trainer.types import KeyedAction, PlayerId, Winner

from .types import OpponentSample, ValueTransition

SelfPlayProgressCallback = Callable[[int, int, Mapping[str, int]], None]


class PolicyLike(Protocol):
    def choose_action_key(
        self,
        view: Dict[str, Any],
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: Mapping[str, Any] | None = None,
    ) -> str: ...


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
    policies: Mapping[PlayerId, PolicyLike],
    seed: str,
    first_player: PlayerId,
    rng: random.Random,
) -> SelfPlayEpisode:
    step_result = env.reset(seed=seed, first_player=first_player)

    pending_observation_by_player: Dict[PlayerId, Sequence[float] | None] = {
        "PlayerA": None,
        "PlayerB": None,
    }
    pending_timestep_by_player: Dict[PlayerId, int | None] = {
        "PlayerA": None,
        "PlayerB": None,
    }
    next_timestep_by_player: Dict[PlayerId, int] = {
        "PlayerA": 0,
        "PlayerB": 0,
    }
    value_transitions: list[ValueTransition] = []
    opponent_samples: list[OpponentSample] = []

    while not step_result.terminal:
        legal = env.legal_actions()
        active_player = legal.active_player_id
        policy = policies[active_player]

        observation_vector = encode_observation(step_result.view)
        prior_observation = pending_observation_by_player[active_player]
        prior_timestep = pending_timestep_by_player[active_player]
        if prior_observation is not None:
            if prior_timestep is None:
                raise ValueError(
                    "Missing timestep for pending observation transition. "
                    f"seed={seed} activePlayer={active_player}"
                )
            value_transitions.append(
                ValueTransition(
                    observation=prior_observation,
                    reward=0.0,
                    done=False,
                    next_observation=observation_vector,
                    player_id=active_player,
                    episode_id=seed,
                    timestep=prior_timestep,
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
        pending_timestep_by_player[active_player] = next_timestep_by_player[active_player]
        next_timestep_by_player[active_player] += 1
        step_result = env.step(action_key=action_key)

    winner = _winner_from_state(step_result.state)
    for player_id, observation_vector in pending_observation_by_player.items():
        if observation_vector is None:
            continue
        timestep = pending_timestep_by_player[player_id]
        if timestep is None:
            raise ValueError(
                "Missing timestep for terminal transition. "
                f"seed={seed} playerId={player_id}"
            )
        value_transitions.append(
            ValueTransition(
                observation=observation_vector,
                reward=_terminal_reward(winner=winner, player_id=player_id),
                done=True,
                next_observation=None,
                player_id=player_id,
                episode_id=seed,
                timestep=timestep,
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
    policy_player_a: PolicyLike,
    policy_player_b: PolicyLike,
    games: int,
    seed_prefix: str,
    progress_every_games: int = 0,
    on_progress: SelfPlayProgressCallback | None = None,
) -> list[SelfPlayEpisode]:
    if games <= 0:
        raise ValueError("games must be > 0.")

    policies: Dict[PlayerId, PolicyLike] = {
        "PlayerA": policy_player_a,
        "PlayerB": policy_player_b,
    }
    rng = random.Random(0)
    episodes: list[SelfPlayEpisode] = []
    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    for index in range(games):
        first_player: PlayerId = "PlayerA" if index % 2 == 0 else "PlayerB"
        seed = f"{seed_prefix}-{index}"
        episode = collect_self_play_episode(
            env=env,
            policies=policies,
            seed=seed,
            first_player=first_player,
            rng=rng,
        )
        episodes.append(episode)
        winners[episode.winner] += 1

        completed = index + 1
        if (
            on_progress is not None
            and progress_every_games > 0
            and (completed % progress_every_games == 0 or completed == games)
        ):
            on_progress(completed, games, dict(winners))
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
    if not isinstance(final_score, dict):
        raise ValueError(
            "Terminal state is missing finalScore. "
            f"turn={state.get('turn')} phase={state.get('phase')!r}"
        )
    winner = final_score.get("winner")
    if winner in ("PlayerA", "PlayerB", "Draw"):
        return winner
    raise ValueError(
        "Terminal state has invalid finalScore.winner. "
        f"winner={winner!r} turn={state.get('turn')} phase={state.get('phase')!r}"
    )


def _terminal_reward(*, winner: Winner, player_id: PlayerId) -> float:
    if winner == "Draw":
        return 0.0
    return 1.0 if winner == player_id else -1.0


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Expected integer value, got bool.")
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(f"Expected numeric value, got {type(value).__name__}.")
