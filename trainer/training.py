from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

from .basic_policies import Policy
from .bridge_payloads import ActionId, GamePhase, SerializedStatePayload
from .encoding import encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .types import DecisionSample, DecisionSamplePayload, KeyedAction, PlayerId, Winner


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
        action_key = policy.choose_action_key(
            step_result.view,
            legal.actions,
            rng,
            state=step_result.state,
        )
        action_index = _find_action_index(legal.actions, action_key)
        chosen_action = legal.actions[action_index]
        action_probs = _action_probs_from_policy(
            policy=policy,
            legal_actions=legal.actions,
            chosen_action_index=action_index,
        )

        staged.append(
            DecisionSample(
                seed=seed,
                turn=step_result.state["turn"],
                phase=step_result.state["phase"],
                active_player_id=active_player,
                action_key=action_key,
                action_id=chosen_action.action_id,
                action_index=action_index,
                observation=observation_vector,
                action_features=action_vectors,
                winner="Draw",
                reward=0.0,
                action_probs=action_probs,
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


def read_samples_jsonl(input_path: Path) -> List[DecisionSample]:
    samples: List[DecisionSample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            candidate = line.strip()
            if not candidate:
                continue
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}, got {type(payload).__name__}."
                )
            samples.append(_decision_sample_from_json(payload, line_number))
    return samples


def _find_action_index(actions: Sequence[KeyedAction], action_key: str) -> int:
    for index, action in enumerate(actions):
        if action.action_key == action_key:
            return index
    raise ValueError(f"Selected action key was not present in legal actions: {action_key}")


def _winner_from_state(state: SerializedStatePayload) -> Winner:
    final_score = state.get("finalScore")
    if final_score is None:
        raise ValueError(
            "Terminal state is missing finalScore. "
            f"turn={state['turn']} phase={state['phase']!r}"
        )
    return final_score["winner"]


def _attach_reward(sample: DecisionSample, winner: Winner) -> DecisionSample:
    if winner == "Draw":
        reward = 0.0
    elif winner == sample.active_player_id:
        reward = 1.0
    else:
        reward = -1.0
    return replace(sample, winner=winner, reward=reward)


def _decision_sample_from_json(payload: Mapping[str, object], line_number: int) -> DecisionSample:
    active_player_id = _as_player_id(payload.get("activePlayerId"), line_number, "activePlayerId")
    winner = _as_winner(payload.get("winner"), line_number, "winner")

    observation = _float_list(payload.get("observation"), line_number, "observation")
    action_features_raw = payload.get("actionFeatures")
    if not isinstance(action_features_raw, list):
        raise ValueError(f"actionFeatures must be a list on line {line_number}.")

    action_features: List[List[float]] = []
    for index, features in enumerate(action_features_raw):
        action_features.append(
            _float_list(
                features,
                line_number=line_number,
                field=f"actionFeatures[{index}]",
            )
        )

    action_index = _as_int(payload.get("actionIndex"), line_number, "actionIndex")
    if action_index < 0 or action_index >= len(action_features):
        raise ValueError(
            "actionIndex is out of bounds on line "
            f"{line_number}: index={action_index}, candidates={len(action_features)}."
        )
    action_probs = _optional_float_list(payload.get("actionProbs"), line_number, "actionProbs")
    if action_probs is not None and len(action_probs) != len(action_features):
        raise ValueError(
            "actionProbs length mismatch on line "
            f"{line_number}: probs={len(action_probs)}, candidates={len(action_features)}."
        )

    decision_payload: DecisionSamplePayload = {
        "seed": str(payload.get("seed", "")),
        "turn": _as_int(payload.get("turn"), line_number, "turn"),
        "phase": _as_game_phase(payload.get("phase"), line_number, "phase"),
        "activePlayerId": active_player_id,
        "actionKey": str(payload.get("actionKey", "")),
        "actionId": _as_action_id(payload.get("actionId"), line_number, "actionId"),
        "actionIndex": action_index,
        "observation": observation,
        "actionFeatures": action_features,
        "winner": winner,
        "reward": _as_float(payload.get("reward"), line_number, "reward"),
        "actionProbs": action_probs,
    }
    return DecisionSample(
        seed=decision_payload["seed"],
        turn=decision_payload["turn"],
        phase=decision_payload["phase"],
        active_player_id=decision_payload["activePlayerId"],
        action_key=decision_payload["actionKey"],
        action_id=decision_payload["actionId"],
        action_index=decision_payload["actionIndex"],
        observation=decision_payload["observation"],
        action_features=decision_payload["actionFeatures"],
        winner=decision_payload["winner"],
        reward=decision_payload["reward"],
        action_probs=decision_payload["actionProbs"],
    )


def _float_list(value: object, line_number: int, field: str) -> List[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list on line {line_number}.")
    out: List[float] = []
    for entry in value:
        out.append(_as_float(entry, line_number, field))
    return out


def _as_int(value: object, line_number: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer on line {line_number}.")
    return value


def _as_float(value: object, line_number: int, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric on line {line_number}.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field} must be numeric on line {line_number}.")


def _optional_float_list(value: object, line_number: int, field: str) -> List[float] | None:
    if value is None:
        return None
    return _float_list(value, line_number, field)


def _as_player_id(value: object, line_number: int, field: str) -> PlayerId:
    if value not in ("PlayerA", "PlayerB"):
        raise ValueError(f"Invalid {field} on line {line_number}: {value!r}")
    return value


def _as_winner(value: object, line_number: int, field: str) -> Winner:
    if value not in ("PlayerA", "PlayerB", "Draw"):
        raise ValueError(f"Invalid {field} on line {line_number}: {value!r}")
    return value


def _as_game_phase(value: object, line_number: int, field: str) -> GamePhase:
    if value not in (
        "StartTurn",
        "TaxCheck",
        "CollectIncome",
        "ActionWindow",
        "DrawCard",
        "GameOver",
    ):
        raise ValueError(f"Invalid {field} on line {line_number}: {value!r}")
    return value


def _as_action_id(value: object, line_number: int, field: str) -> ActionId:
    if value not in (
        "buy-deed",
        "choose-income-suit",
        "develop-deed",
        "develop-outright",
        "end-turn",
        "sell-card",
        "trade",
    ):
        raise ValueError(f"Invalid {field} on line {line_number}: {value!r}")
    return value


def _action_probs_from_policy(
    *,
    policy: Policy,
    legal_actions: Sequence[KeyedAction],
    chosen_action_index: int,
) -> List[float]:
    by_key = policy.root_action_probs()
    if not by_key:
        raise ValueError(
            "Policy did not provide root_action_probs for training sample. "
            f"policy={getattr(policy, 'name', type(policy).__name__)}"
        )

    raw: List[float] = []
    for action in legal_actions:
        value = float(by_key.get(action.action_key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "Policy returned invalid root action probability. "
                f"policy={getattr(policy, 'name', type(policy).__name__)} "
                f"actionKey={action.action_key!r} value={value}"
            )
        raw.append(value)
    total = sum(raw)
    if total <= 0.0:
        raise ValueError(
            "Policy root_action_probs sum must be > 0 for training sample. "
            f"policy={getattr(policy, 'name', type(policy).__name__)} "
            f"actions={len(legal_actions)} chosenIndex={chosen_action_index}"
        )
    return [value / total for value in raw]
