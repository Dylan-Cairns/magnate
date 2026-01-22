from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, NotRequired, Optional, Tuple, TypedDict, cast

import torch

from trainer.encoding import ENCODING_VERSION

from .models import OpponentModel, ValueNet

TD_VALUE_CHECKPOINT_TYPE = "magnate_td_value_v1"
TD_OPPONENT_CHECKPOINT_TYPE = "magnate_td_opponent_v1"


class ValueCheckpointPayload(TypedDict):
    checkpointType: Literal["magnate_td_value_v1"]
    encodingVersion: int
    observationDim: int
    hiddenDim: int
    stateDict: dict[str, torch.Tensor]
    metadata: dict[str, object]
    optimizerStateDict: NotRequired[dict[str, object]]


class OpponentCheckpointPayload(TypedDict):
    checkpointType: Literal["magnate_td_opponent_v1"]
    encodingVersion: int
    observationDim: int
    actionFeatureDim: int
    hiddenDim: int
    stateDict: dict[str, torch.Tensor]
    metadata: dict[str, object]
    optimizerStateDict: NotRequired[dict[str, object]]


def save_value_checkpoint(
    *,
    model: ValueNet,
    output_path: Path,
    metadata: Mapping[str, object] | None = None,
    optimizer_state_dict: Optional[dict[str, object]] = None,
) -> None:
    payload: ValueCheckpointPayload = {
        "checkpointType": TD_VALUE_CHECKPOINT_TYPE,
        "encodingVersion": ENCODING_VERSION,
        "observationDim": int(model.observation_dim),
        "hiddenDim": int(model.hidden_dim),
        "stateDict": dict(model.state_dict()),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    if optimizer_state_dict is not None:
        payload["optimizerStateDict"] = optimizer_state_dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_value_checkpoint(
    *,
    path: Path,
    map_location: str = "cpu",
) -> Tuple[ValueNet, ValueCheckpointPayload]:
    raw = _load_mapping(path=path, map_location=map_location)
    payload = _parse_value_checkpoint_payload(raw)

    model = ValueNet(observation_dim=payload["observationDim"], hidden_dim=payload["hiddenDim"])
    model.load_state_dict(payload["stateDict"])
    model.eval()
    return model, payload


def save_opponent_checkpoint(
    *,
    model: OpponentModel,
    output_path: Path,
    metadata: Mapping[str, object] | None = None,
    optimizer_state_dict: Optional[dict[str, object]] = None,
) -> None:
    payload: OpponentCheckpointPayload = {
        "checkpointType": TD_OPPONENT_CHECKPOINT_TYPE,
        "encodingVersion": ENCODING_VERSION,
        "observationDim": int(model.observation_dim),
        "actionFeatureDim": int(model.action_feature_dim),
        "hiddenDim": int(model.hidden_dim),
        "stateDict": dict(model.state_dict()),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    if optimizer_state_dict is not None:
        payload["optimizerStateDict"] = optimizer_state_dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_opponent_checkpoint(
    *,
    path: Path,
    map_location: str = "cpu",
) -> Tuple[OpponentModel, OpponentCheckpointPayload]:
    raw = _load_mapping(path=path, map_location=map_location)
    payload = _parse_opponent_checkpoint_payload(raw)

    model = OpponentModel(
        observation_dim=payload["observationDim"],
        action_feature_dim=payload["actionFeatureDim"],
        hidden_dim=payload["hiddenDim"],
    )
    model.load_state_dict(payload["stateDict"])
    model.eval()
    return model, payload


def _load_mapping(*, path: Path, map_location: str) -> dict[str, object]:
    raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint payload must be a mapping.")
    return cast(dict[str, object], raw)


def _parse_value_checkpoint_payload(raw: Mapping[str, object]) -> ValueCheckpointPayload:
    _assert_checkpoint_type(raw=raw, expected=TD_VALUE_CHECKPOINT_TYPE)
    _assert_encoding_version(raw=raw)
    payload: ValueCheckpointPayload = {
        "checkpointType": TD_VALUE_CHECKPOINT_TYPE,
        "encodingVersion": _as_positive_int(raw.get("encodingVersion"), "encodingVersion"),
        "observationDim": _as_positive_int(raw.get("observationDim"), "observationDim"),
        "hiddenDim": _as_positive_int(raw.get("hiddenDim"), "hiddenDim"),
        "stateDict": _require_state_dict(raw.get("stateDict"), label="stateDict"),
        "metadata": _require_metadata(raw.get("metadata"), label="metadata"),
    }
    if "optimizerStateDict" in raw:
        payload["optimizerStateDict"] = _require_object_dict(
            raw.get("optimizerStateDict"),
            label="optimizerStateDict",
        )
    return payload


def _parse_opponent_checkpoint_payload(raw: Mapping[str, object]) -> OpponentCheckpointPayload:
    _assert_checkpoint_type(raw=raw, expected=TD_OPPONENT_CHECKPOINT_TYPE)
    _assert_encoding_version(raw=raw)
    payload: OpponentCheckpointPayload = {
        "checkpointType": TD_OPPONENT_CHECKPOINT_TYPE,
        "encodingVersion": _as_positive_int(raw.get("encodingVersion"), "encodingVersion"),
        "observationDim": _as_positive_int(raw.get("observationDim"), "observationDim"),
        "actionFeatureDim": _as_positive_int(raw.get("actionFeatureDim"), "actionFeatureDim"),
        "hiddenDim": _as_positive_int(raw.get("hiddenDim"), "hiddenDim"),
        "stateDict": _require_state_dict(raw.get("stateDict"), label="stateDict"),
        "metadata": _require_metadata(raw.get("metadata"), label="metadata"),
    }
    if "optimizerStateDict" in raw:
        payload["optimizerStateDict"] = _require_object_dict(
            raw.get("optimizerStateDict"),
            label="optimizerStateDict",
        )
    return payload


def _assert_checkpoint_type(*, raw: Mapping[str, object], expected: str) -> None:
    checkpoint_type = str(raw.get("checkpointType", ""))
    if checkpoint_type != expected:
        raise ValueError(
            f"Unsupported checkpoint type {checkpoint_type!r}; expected {expected!r}."
        )


def _assert_encoding_version(*, raw: Mapping[str, object]) -> None:
    if "encodingVersion" not in raw:
        raise ValueError("Checkpoint is missing encodingVersion metadata.")
    version = _as_positive_int(raw.get("encodingVersion"), "encodingVersion")
    if version != ENCODING_VERSION:
        raise ValueError(
            f"Encoding version mismatch: checkpoint={version} expected={ENCODING_VERSION}."
        )


def _require_state_dict(value: object, *, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings.")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label}.{key} must be a tensor.")
        state_dict[key] = tensor
    return state_dict


def _require_metadata(value: object, *, label: str) -> dict[str, object]:
    if value is None:
        return {}
    return _require_object_dict(value, label=label)


def _require_object_dict(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    metadata: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings.")
        metadata[key] = item
    return metadata


def _as_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value
