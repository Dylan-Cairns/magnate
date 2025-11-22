from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from trainer.types import PlayerId

from .types import OpponentSample, ValueTransition


def write_value_transitions_jsonl(
    transitions: Iterable[ValueTransition],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, transition in enumerate(transitions, start=1):
            if transition.done and transition.next_observation is not None:
                raise ValueError(
                    "Invalid value transition while writing JSONL: "
                    "done=True requires next_observation=None. "
                    f"entry={index}"
                )
            if (not transition.done) and transition.next_observation is None:
                raise ValueError(
                    "Invalid value transition while writing JSONL: "
                    "done=False requires next_observation. "
                    f"entry={index}"
                )
            has_episode = transition.episode_id is not None
            has_timestep = transition.timestep is not None
            if has_episode != has_timestep:
                raise ValueError(
                    "Invalid value transition while writing JSONL: "
                    "episode_id and timestep must both be set or both be None. "
                    f"entry={index}"
                )
            if transition.episode_id is not None and transition.episode_id == "":
                raise ValueError(
                    "Invalid value transition while writing JSONL: "
                    "episode_id must be non-empty when set. "
                    f"entry={index}"
                )
            if transition.timestep is not None and transition.timestep < 0:
                raise ValueError(
                    "Invalid value transition while writing JSONL: "
                    "timestep must be >= 0 when set. "
                    f"entry={index}"
                )
            payload = {
                "observation": list(transition.observation),
                "reward": float(transition.reward),
                "done": bool(transition.done),
                "nextObservation": (
                    list(transition.next_observation)
                    if transition.next_observation is not None
                    else None
                ),
                "playerId": transition.player_id,
            }
            if transition.episode_id is not None:
                payload["episodeId"] = transition.episode_id
                payload["timestep"] = int(transition.timestep)
            handle.write(json.dumps(payload) + "\n")


def read_value_transitions_jsonl(input_path: Path) -> list[ValueTransition]:
    transitions: list[ValueTransition] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}, got {type(payload).__name__}."
                )
            transitions.append(_value_transition_from_json(payload, line_number))
    return transitions


def write_opponent_samples_jsonl(
    samples: Iterable[OpponentSample],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "observation": list(sample.observation),
                "actionFeatures": [list(features) for features in sample.action_features],
                "actionIndex": int(sample.action_index),
                "playerId": sample.player_id,
            }
            handle.write(json.dumps(payload) + "\n")


def read_opponent_samples_jsonl(input_path: Path) -> list[OpponentSample]:
    samples: list[OpponentSample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number}, got {type(payload).__name__}."
                )
            samples.append(_opponent_sample_from_json(payload, line_number))
    return samples


def _value_transition_from_json(
    payload: Mapping[str, Any],
    line_number: int,
) -> ValueTransition:
    observation = _float_list(payload.get("observation"), line_number, "observation")
    reward = _as_float(payload.get("reward"), line_number, "reward")
    done = _as_bool(payload.get("done"), line_number, "done")
    next_observation_raw = payload.get("nextObservation")
    next_observation = None
    if next_observation_raw is not None:
        next_observation = _float_list(
            next_observation_raw,
            line_number,
            "nextObservation",
        )
    if done and next_observation is not None:
        raise ValueError(
            "done=True requires nextObservation to be null "
            f"on line {line_number}."
        )
    if (not done) and next_observation is None:
        raise ValueError(
            "done=False requires nextObservation "
            f"on line {line_number}."
        )
    player_id = _as_player_id(payload.get("playerId"), line_number, "playerId")
    episode_id, timestep = _as_episode_step(payload, line_number)
    return ValueTransition(
        observation=observation,
        reward=reward,
        done=done,
        next_observation=next_observation,
        player_id=player_id,
        episode_id=episode_id,
        timestep=timestep,
    )


def _opponent_sample_from_json(
    payload: Mapping[str, Any],
    line_number: int,
) -> OpponentSample:
    observation = _float_list(payload.get("observation"), line_number, "observation")
    action_features_raw = payload.get("actionFeatures")
    if not isinstance(action_features_raw, list):
        raise ValueError(f"actionFeatures must be a list on line {line_number}.")
    action_features: List[List[float]] = []
    for index, entry in enumerate(action_features_raw):
        action_features.append(
            _float_list(entry, line_number, f"actionFeatures[{index}]")
        )
    action_index = _as_int(payload.get("actionIndex"), line_number, "actionIndex")
    if action_index < 0 or action_index >= len(action_features):
        raise ValueError(
            f"actionIndex out of bounds on line {line_number}: "
            f"index={action_index}, candidates={len(action_features)}."
        )
    player_id = _as_player_id(payload.get("playerId"), line_number, "playerId")
    return OpponentSample(
        observation=observation,
        action_features=action_features,
        action_index=action_index,
        player_id=player_id,
    )


def _float_list(value: Any, line_number: int, field: str) -> List[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list on line {line_number}.")
    out: List[float] = []
    for entry in value:
        out.append(_as_float(entry, line_number, field))
    return out


def _as_int(value: Any, line_number: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer on line {line_number}.")
    return value


def _as_float(value: Any, line_number: int, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric on line {line_number}.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{field} must be numeric on line {line_number}.")


def _as_bool(value: Any, line_number: int, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean on line {line_number}.")
    return value


def _as_player_id(value: Any, line_number: int, field: str) -> PlayerId:
    if value not in ("PlayerA", "PlayerB"):
        raise ValueError(f"{field} must be PlayerA|PlayerB on line {line_number}.")
    return value


def _as_episode_step(
    payload: Mapping[str, Any],
    line_number: int,
) -> tuple[str | None, int | None]:
    episode_raw = payload.get("episodeId")
    timestep_raw = payload.get("timestep")
    has_episode = episode_raw is not None
    has_timestep = timestep_raw is not None
    if not has_episode and not has_timestep:
        return None, None
    if has_episode != has_timestep:
        raise ValueError(
            "episodeId and timestep must either both be present or both be absent "
            f"on line {line_number}."
        )
    if not isinstance(episode_raw, str) or episode_raw == "":
        raise ValueError(
            f"episodeId must be a non-empty string on line {line_number}."
        )
    if isinstance(timestep_raw, bool) or not isinstance(timestep_raw, int):
        raise ValueError(
            f"timestep must be an integer on line {line_number}."
        )
    if timestep_raw < 0:
        raise ValueError(
            f"timestep must be >= 0 on line {line_number}."
        )
    return episode_raw, timestep_raw
