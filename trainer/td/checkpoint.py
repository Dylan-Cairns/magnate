from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from trainer.encoding import ENCODING_VERSION

from .models import OpponentModel, ValueNet

TD_VALUE_CHECKPOINT_TYPE = "magnate_td_value_v1"
TD_OPPONENT_CHECKPOINT_TYPE = "magnate_td_opponent_v1"


def save_value_checkpoint(
    *,
    model: ValueNet,
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpointType": TD_VALUE_CHECKPOINT_TYPE,
        "encodingVersion": ENCODING_VERSION,
        "observationDim": int(model.observation_dim),
        "hiddenDim": int(model.hidden_dim),
        "stateDict": model.state_dict(),
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
) -> Tuple[ValueNet, Dict[str, Any]]:
    raw = _load_mapping(path=path, map_location=map_location)
    _assert_checkpoint_type(raw=raw, expected=TD_VALUE_CHECKPOINT_TYPE)
    _assert_encoding_version(raw=raw)

    observation_dim = _as_positive_int(raw.get("observationDim"), "observationDim")
    hidden_dim = _as_positive_int(raw.get("hiddenDim"), "hiddenDim")
    state_dict = raw.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError("Value checkpoint stateDict must be a mapping.")

    model = ValueNet(observation_dim=observation_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model, raw


def save_opponent_checkpoint(
    *,
    model: OpponentModel,
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpointType": TD_OPPONENT_CHECKPOINT_TYPE,
        "encodingVersion": ENCODING_VERSION,
        "observationDim": int(model.observation_dim),
        "actionFeatureDim": int(model.action_feature_dim),
        "hiddenDim": int(model.hidden_dim),
        "stateDict": model.state_dict(),
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
) -> Tuple[OpponentModel, Dict[str, Any]]:
    raw = _load_mapping(path=path, map_location=map_location)
    _assert_checkpoint_type(raw=raw, expected=TD_OPPONENT_CHECKPOINT_TYPE)
    _assert_encoding_version(raw=raw)

    observation_dim = _as_positive_int(raw.get("observationDim"), "observationDim")
    action_feature_dim = _as_positive_int(raw.get("actionFeatureDim"), "actionFeatureDim")
    hidden_dim = _as_positive_int(raw.get("hiddenDim"), "hiddenDim")
    state_dict = raw.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError("Opponent checkpoint stateDict must be a mapping.")

    model = OpponentModel(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, raw


def _load_mapping(*, path: Path, map_location: str) -> Dict[str, Any]:
    raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint payload must be a mapping.")
    return raw


def _assert_checkpoint_type(*, raw: Mapping[str, Any], expected: str) -> None:
    checkpoint_type = str(raw.get("checkpointType", ""))
    if checkpoint_type != expected:
        raise ValueError(
            f"Unsupported checkpoint type {checkpoint_type!r}; expected {expected!r}."
        )


def _assert_encoding_version(*, raw: Mapping[str, Any]) -> None:
    if "encodingVersion" not in raw:
        raise ValueError("Checkpoint is missing encodingVersion metadata.")
    version = _as_positive_int(raw.get("encodingVersion"), "encodingVersion")
    if version != ENCODING_VERSION:
        raise ValueError(
            f"Encoding version mismatch: checkpoint={version} expected={ENCODING_VERSION}."
        )


def _as_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value
