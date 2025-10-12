from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .encoding import encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .policies import Policy
from .types import DecisionSample, PlayerId, Winner


def collect_episode_samples(
    env: MagnateBridgeEnv,
    policies: Mapping[PlayerId, Policy],
    seed: str,
    first_player: PlayerId,
    rng: random.Random,
) -> List[DecisionSample]:
    step_result = env.reset(seed=seed, first_player=first_player)
    staged: List[DecisionSample] = []

    while not step_result.terminal:
        legal = env.legal_actions()
        active_player = legal.active_player_id
        policy = policies[active_player]

        observation_vector = encode_observation(step_result.view)
        action_vectors = encode_action_candidates(legal.actions)
        action_key = policy.choose_action_key(step_result.view, legal.actions, rng)
        action_index = _find_action_index(legal.actions, action_key)
        chosen_action = legal.actions[action_index]

        staged.append(
            DecisionSample(
                seed=seed,
                turn=int(step_result.state.get("turn", 0)),
                phase=str(step_result.state.get("phase", "")),
                active_player_id=active_player,
                action_key=action_key,
                action_id=chosen_action.action_id,
                action_index=action_index,
                observation=observation_vector,
                action_features=action_vectors,
                winner="Draw",
                reward=0.0,
            )
        )

        step_result = env.step(action_key=action_key)

    winner = _winner_from_state(step_result.state)
    return [_attach_reward(sample, winner) for sample in staged]


def collect_training_samples(
    env: MagnateBridgeEnv,
    policy_player_a: Policy,
    policy_player_b: Policy,
    games: int,
    seed_prefix: str,
) -> List[DecisionSample]:
    policies: Dict[PlayerId, Policy] = {
        "PlayerA": policy_player_a,
        "PlayerB": policy_player_b,
    }

    all_samples: List[DecisionSample] = []
    rng = random.Random(0)
    for index in range(games):
        seed = f"{seed_prefix}-{index}"
        first_player: PlayerId = "PlayerA" if index % 2 == 0 else "PlayerB"
        all_samples.extend(
            collect_episode_samples(
                env=env,
                policies=policies,
                seed=seed,
                first_player=first_player,
                rng=rng,
            )
        )
    return all_samples


def write_samples_jsonl(samples: Sequence[DecisionSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.as_json()) + "\n")


def _find_action_index(actions, action_key: str) -> int:
    for index, action in enumerate(actions):
        if action.action_key == action_key:
            return index
    raise ValueError(f"Selected action key was not present in legal actions: {action_key}")


def _winner_from_state(state: Mapping[str, object]) -> Winner:
    final_score = state.get("finalScore")
    if isinstance(final_score, dict):
        winner = final_score.get("winner")
        if winner in ("PlayerA", "PlayerB", "Draw"):
            return winner
    return "Draw"


def _attach_reward(sample: DecisionSample, winner: Winner) -> DecisionSample:
    if winner == "Draw":
        reward = 0.0
    elif winner == sample.active_player_id:
        reward = 1.0
    else:
        reward = -1.0
    return replace(sample, winner=winner, reward=reward)

