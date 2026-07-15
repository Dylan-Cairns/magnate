from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypedDict, cast

import torch
import torch.nn.functional as functional

from trainer.encoding import ACTION_FEATURE_DIM, ENCODING_VERSION, OBSERVATION_DIM

from .checkpoint import (
    TD_OPPONENT_CHECKPOINT_TYPE,
    TD_VALUE_CHECKPOINT_TYPE,
    load_opponent_checkpoint,
    load_value_checkpoint,
    save_opponent_checkpoint,
    save_value_checkpoint,
)
from .models import OpponentModel, ValueNet

BROWSER_TD_ROOT_MANIFEST_SCHEMA_VERSION = 1
BROWSER_TD_ROOT_WEIGHTS_SCHEMA_VERSION = 1
BROWSER_TD_ROOT_MODEL_TYPE = "td-root-search-v1"
OUTPUT_PARITY_ABSOLUTE_TOLERANCE = 1e-6


class ModelParityReport(TypedDict):
    tensorCount: int
    parameterCount: int
    maxOutputAbsoluteDifference: float


class BrowserPackReconstructionResult(TypedDict):
    packId: str
    manifest: str
    weights: str
    manifestSha256: str
    weightsSha256: str
    valueCheckpoint: str
    opponentCheckpoint: str
    valueParity: ModelParityReport
    opponentParity: ModelParityReport


class BrowserPackCheckpointError(ValueError):
    """Raised when a browser pack cannot be reconstructed without ambiguity."""


@dataclass(frozen=True)
class _NetworkSpec:
    checkpoint_type: str
    encoding_version: int
    observation_dim: int
    hidden_dim: int
    required_state_dict_keys: tuple[str, ...]
    action_feature_dim: int | None = None


@dataclass(frozen=True)
class _ManifestSpec:
    pack_id: str
    label: str
    created_at_utc: str
    weights_path: Path
    value: _NetworkSpec
    opponent: _NetworkSpec
    source_run_id: str | None
    source_value_checkpoint: str | None
    source_opponent_checkpoint: str | None
    checkpoint_metadata: dict[str, object]


def reconstruct_browser_td_root_checkpoints(
    *,
    manifest_path: Path,
    output_dir: Path,
    value_filename: str = "value.pt",
    opponent_filename: str = "opponent.pt",
    overwrite: bool = False,
) -> BrowserPackReconstructionResult:
    """Reconstruct trainer checkpoints from a static TD-root browser model pack.

    The browser pack is validated against the current trainer architecture. Both
    generated checkpoints are loaded through the canonical checkpoint readers,
    compared tensor-for-tensor with the JSON source, and checked against manual
    forward passes that do not call the model ``forward`` implementations.
    """

    resolved_manifest_path = manifest_path.resolve()
    manifest = _read_json_object(path=resolved_manifest_path, label="browser pack manifest")
    manifest_spec = _parse_manifest(
        manifest=manifest,
        manifest_path=resolved_manifest_path,
    )
    weights = _read_json_object(
        path=manifest_spec.weights_path,
        label="browser pack weights",
    )
    manifest_sha256 = _sha256_file(resolved_manifest_path)
    weights_sha256 = _sha256_file(manifest_spec.weights_path)
    _validate_schema_version(
        value=weights.get("schemaVersion"),
        expected=BROWSER_TD_ROOT_WEIGHTS_SCHEMA_VERSION,
        label="weights.schemaVersion",
    )

    value_model = ValueNet(
        observation_dim=manifest_spec.value.observation_dim,
        hidden_dim=manifest_spec.value.hidden_dim,
    )
    opponent_action_dim = manifest_spec.opponent.action_feature_dim
    if opponent_action_dim is None:
        raise BrowserPackCheckpointError(
            "manifest.model.opponent.actionFeatureDim is required."
        )
    opponent_model = OpponentModel(
        observation_dim=manifest_spec.opponent.observation_dim,
        action_feature_dim=opponent_action_dim,
        hidden_dim=manifest_spec.opponent.hidden_dim,
    )

    value_template = value_model.state_dict()
    opponent_template = opponent_model.state_dict()
    _validate_required_keys(
        declared=manifest_spec.value.required_state_dict_keys,
        expected=tuple(value_template.keys()),
        label="manifest.model.value.requiredStateDictKeys",
    )
    _validate_required_keys(
        declared=manifest_spec.opponent.required_state_dict_keys,
        expected=tuple(opponent_template.keys()),
        label="manifest.model.opponent.requiredStateDictKeys",
    )

    value_state = _parse_tensor_record(
        value=weights.get("valueTensors"),
        template=value_template,
        label="weights.valueTensors",
    )
    opponent_state = _parse_tensor_record(
        value=weights.get("opponentTensors"),
        template=opponent_template,
        label="weights.opponentTensors",
    )
    value_model.load_state_dict(value_state, strict=True)
    opponent_model.load_state_dict(opponent_state, strict=True)
    value_model.eval()
    opponent_model.eval()

    # Verify the in-memory reconstruction before creating any output files.
    _verify_value_parity(model=value_model, source_state=value_state)
    _verify_opponent_parity(model=opponent_model, source_state=opponent_state)

    value_output_path = _resolve_output_path(
        output_dir=output_dir,
        filename=value_filename,
        label="value filename",
    )
    opponent_output_path = _resolve_output_path(
        output_dir=output_dir,
        filename=opponent_filename,
        label="opponent filename",
    )
    if value_output_path == opponent_output_path:
        raise BrowserPackCheckpointError(
            "Value and opponent checkpoint output paths must be different."
        )
    for path in (value_output_path, opponent_output_path):
        if path.exists() and not overwrite:
            raise BrowserPackCheckpointError(
                f"Checkpoint output already exists: {path}. Use overwrite=True to replace it."
            )

    output_dir_resolved = output_dir.resolve()
    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    value_metadata = _checkpoint_metadata(
        manifest_spec=manifest_spec,
        manifest_path=resolved_manifest_path,
        model_kind="value",
        manifest_sha256=manifest_sha256,
        weights_sha256=weights_sha256,
    )
    opponent_metadata = _checkpoint_metadata(
        manifest_spec=manifest_spec,
        manifest_path=resolved_manifest_path,
        model_kind="opponent",
        manifest_sha256=manifest_sha256,
        weights_sha256=weights_sha256,
    )

    with tempfile.TemporaryDirectory(
        prefix=".browser-pack-reconstruction-",
        dir=output_dir_resolved,
    ) as temporary_dir_raw:
        temporary_dir = Path(temporary_dir_raw)
        temporary_value = temporary_dir / "value.pt"
        temporary_opponent = temporary_dir / "opponent.pt"
        save_value_checkpoint(
            model=value_model,
            output_path=temporary_value,
            metadata=value_metadata,
        )
        save_opponent_checkpoint(
            model=opponent_model,
            output_path=temporary_opponent,
            metadata=opponent_metadata,
        )

        loaded_value, value_payload = load_value_checkpoint(path=temporary_value)
        loaded_opponent, opponent_payload = load_opponent_checkpoint(path=temporary_opponent)
        if "optimizerStateDict" in value_payload or "optimizerStateDict" in opponent_payload:
            raise BrowserPackCheckpointError(
                "Reconstructed warm-start checkpoints must not contain optimizer state."
            )

        value_parity = _verify_value_parity(
            model=loaded_value,
            source_state=value_state,
        )
        opponent_parity = _verify_opponent_parity(
            model=loaded_opponent,
            source_state=opponent_state,
        )

        os.replace(temporary_value, value_output_path)
        os.replace(temporary_opponent, opponent_output_path)

    return {
        "packId": manifest_spec.pack_id,
        "manifest": str(resolved_manifest_path),
        "weights": str(manifest_spec.weights_path),
        "manifestSha256": manifest_sha256,
        "weightsSha256": weights_sha256,
        "valueCheckpoint": str(value_output_path),
        "opponentCheckpoint": str(opponent_output_path),
        "valueParity": value_parity,
        "opponentParity": opponent_parity,
    }


def _parse_manifest(*, manifest: Mapping[str, object], manifest_path: Path) -> _ManifestSpec:
    _validate_schema_version(
        value=manifest.get("schemaVersion"),
        expected=BROWSER_TD_ROOT_MANIFEST_SCHEMA_VERSION,
        label="manifest.schemaVersion",
    )
    pack_id = _require_non_empty_string(manifest.get("packId"), "manifest.packId")
    label = _require_non_empty_string(manifest.get("label"), "manifest.label")
    created_at_utc = _require_non_empty_string(
        manifest.get("createdAtUtc"),
        "manifest.createdAtUtc",
    )
    model = _require_mapping(manifest.get("model"), "manifest.model")
    model_type = _require_non_empty_string(model.get("modelType"), "manifest.model.modelType")
    if model_type != BROWSER_TD_ROOT_MODEL_TYPE:
        raise BrowserPackCheckpointError(
            f"Unsupported browser model type {model_type!r}; "
            f"expected {BROWSER_TD_ROOT_MODEL_TYPE!r}."
        )

    weights_relative = Path(
        _require_non_empty_string(model.get("weightsPath"), "manifest.model.weightsPath")
    )
    if weights_relative.is_absolute():
        raise BrowserPackCheckpointError("manifest.model.weightsPath must be relative.")
    pack_dir = manifest_path.parent.resolve()
    weights_path = (pack_dir / weights_relative).resolve()
    try:
        weights_path.relative_to(pack_dir)
    except ValueError as error:
        raise BrowserPackCheckpointError(
            "manifest.model.weightsPath must remain inside the browser pack directory."
        ) from error

    value_spec = _parse_network_spec(
        value=_require_mapping(model.get("value"), "manifest.model.value"),
        label="manifest.model.value",
        include_action_feature_dim=False,
    )
    opponent_spec = _parse_network_spec(
        value=_require_mapping(model.get("opponent"), "manifest.model.opponent"),
        label="manifest.model.opponent",
        include_action_feature_dim=True,
    )
    _validate_network_specs(value=value_spec, opponent=opponent_spec)

    source = _require_mapping(manifest.get("source"), "manifest.source")
    checkpoint_metadata_raw = source.get("checkpointMetadata")
    checkpoint_metadata = (
        {}
        if checkpoint_metadata_raw is None
        else dict(_require_mapping(checkpoint_metadata_raw, "manifest.source.checkpointMetadata"))
    )
    return _ManifestSpec(
        pack_id=pack_id,
        label=label,
        created_at_utc=created_at_utc,
        weights_path=weights_path,
        value=value_spec,
        opponent=opponent_spec,
        source_run_id=_optional_string(source.get("runId"), "manifest.source.runId"),
        source_value_checkpoint=_optional_string(
            source.get("valueCheckpoint"),
            "manifest.source.valueCheckpoint",
        ),
        source_opponent_checkpoint=_optional_string(
            source.get("opponentCheckpoint"),
            "manifest.source.opponentCheckpoint",
        ),
        checkpoint_metadata=checkpoint_metadata,
    )


def _parse_network_spec(
    *,
    value: Mapping[str, object],
    label: str,
    include_action_feature_dim: bool,
) -> _NetworkSpec:
    return _NetworkSpec(
        checkpoint_type=_require_non_empty_string(
            value.get("checkpointType"),
            f"{label}.checkpointType",
        ),
        encoding_version=_require_positive_int(
            value.get("encodingVersion"),
            f"{label}.encodingVersion",
        ),
        observation_dim=_require_positive_int(
            value.get("observationDim"),
            f"{label}.observationDim",
        ),
        hidden_dim=_require_positive_int(value.get("hiddenDim"), f"{label}.hiddenDim"),
        required_state_dict_keys=_require_string_tuple(
            value.get("requiredStateDictKeys"),
            f"{label}.requiredStateDictKeys",
        ),
        action_feature_dim=(
            _require_positive_int(
                value.get("actionFeatureDim"),
                f"{label}.actionFeatureDim",
            )
            if include_action_feature_dim
            else None
        ),
    )


def _validate_network_specs(*, value: _NetworkSpec, opponent: _NetworkSpec) -> None:
    if value.checkpoint_type != TD_VALUE_CHECKPOINT_TYPE:
        raise BrowserPackCheckpointError(
            f"Unsupported value checkpoint type {value.checkpoint_type!r}; "
            f"expected {TD_VALUE_CHECKPOINT_TYPE!r}."
        )
    if opponent.checkpoint_type != TD_OPPONENT_CHECKPOINT_TYPE:
        raise BrowserPackCheckpointError(
            f"Unsupported opponent checkpoint type {opponent.checkpoint_type!r}; "
            f"expected {TD_OPPONENT_CHECKPOINT_TYPE!r}."
        )
    for label, actual in (
        ("value encoding", value.encoding_version),
        ("opponent encoding", opponent.encoding_version),
    ):
        if actual != ENCODING_VERSION:
            raise BrowserPackCheckpointError(
                f"{label} mismatch: pack={actual} trainer={ENCODING_VERSION}."
            )
    for label, actual in (
        ("value observation dimension", value.observation_dim),
        ("opponent observation dimension", opponent.observation_dim),
    ):
        if actual != OBSERVATION_DIM:
            raise BrowserPackCheckpointError(
                f"{label} mismatch: pack={actual} trainer={OBSERVATION_DIM}."
            )
    if opponent.action_feature_dim != ACTION_FEATURE_DIM:
        raise BrowserPackCheckpointError(
            "opponent action feature dimension mismatch: "
            f"pack={opponent.action_feature_dim} trainer={ACTION_FEATURE_DIM}."
        )


def _parse_tensor_record(
    *,
    value: object,
    template: Mapping[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    record = _require_mapping(value, label)
    actual_keys = set(record.keys())
    expected_keys = set(template.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise BrowserPackCheckpointError(
            f"{label} tensor keys do not match the trainer model; "
            f"missing={missing} unexpected={unexpected}."
        )

    parsed: dict[str, torch.Tensor] = {}
    for key, expected_tensor in template.items():
        tensor_label = f"{label}.{key}"
        payload = _require_mapping(record.get(key), tensor_label)
        shape = _require_shape(payload.get("shape"), f"{tensor_label}.shape")
        expected_shape = tuple(expected_tensor.shape)
        if shape != expected_shape:
            raise BrowserPackCheckpointError(
                f"{tensor_label}.shape mismatch: pack={list(shape)} "
                f"trainer={list(expected_shape)}."
            )
        values = _require_sequence(payload.get("values"), f"{tensor_label}.values")
        expected_count = expected_tensor.numel()
        if len(values) != expected_count:
            raise BrowserPackCheckpointError(
                f"{tensor_label}.values length mismatch: "
                f"pack={len(values)} trainer={expected_count}."
            )
        for index, item in enumerate(values):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise BrowserPackCheckpointError(
                    f"{tensor_label}.values[{index}] must be a finite number."
                )
            if not math.isfinite(float(item)):
                raise BrowserPackCheckpointError(
                    f"{tensor_label}.values[{index}] must be finite."
                )
        try:
            tensor = torch.tensor(values, dtype=torch.float32).reshape(shape)
        except (TypeError, ValueError, RuntimeError) as error:
            raise BrowserPackCheckpointError(
                f"{tensor_label} could not be represented as a float32 tensor."
            ) from error
        if not bool(torch.isfinite(tensor).all().item()):
            raise BrowserPackCheckpointError(
                f"{tensor_label} contains a value outside the finite float32 range."
            )
        parsed[key] = tensor
    return parsed


def _verify_value_parity(
    *,
    model: ValueNet,
    source_state: Mapping[str, torch.Tensor],
) -> ModelParityReport:
    _verify_exact_tensors(model_state=model.state_dict(), source_state=source_state, label="value")
    observation = torch.linspace(
        -0.875,
        0.875,
        steps=model.observation_dim,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        actual = model(observation)
        expected = _manual_value_output(observation=observation, state=source_state)
    max_difference = _assert_output_parity(actual=actual, expected=expected, label="value")
    return {
        "tensorCount": len(source_state),
        "parameterCount": sum(tensor.numel() for tensor in source_state.values()),
        "maxOutputAbsoluteDifference": max_difference,
    }


def _verify_opponent_parity(
    *,
    model: OpponentModel,
    source_state: Mapping[str, torch.Tensor],
) -> ModelParityReport:
    _verify_exact_tensors(
        model_state=model.state_dict(),
        source_state=source_state,
        label="opponent",
    )
    observation = torch.linspace(
        -0.75,
        0.75,
        steps=model.observation_dim,
        dtype=torch.float32,
    )
    action_base = torch.linspace(
        -0.625,
        0.625,
        steps=model.action_feature_dim,
        dtype=torch.float32,
    )
    action_features = torch.stack((action_base, -action_base, torch.flip(action_base, (0,))))
    with torch.inference_mode():
        actual = model.logits_tensor(observation, action_features)
        expected = _manual_opponent_output(
            observation=observation,
            action_features=action_features,
            state=source_state,
        )
    max_difference = _assert_output_parity(actual=actual, expected=expected, label="opponent")
    return {
        "tensorCount": len(source_state),
        "parameterCount": sum(tensor.numel() for tensor in source_state.values()),
        "maxOutputAbsoluteDifference": max_difference,
    }


def _manual_value_output(
    *,
    observation: torch.Tensor,
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    hidden_one = torch.tanh(
        functional.linear(
            observation,
            state["encoder.0.weight"],
            state["encoder.0.bias"],
        )
    )
    hidden_two = torch.tanh(
        functional.linear(
            hidden_one,
            state["encoder.2.weight"],
            state["encoder.2.bias"],
        )
    )
    return functional.linear(
        hidden_two,
        state["encoder.4.weight"],
        state["encoder.4.bias"],
    ).reshape(-1)


def _manual_opponent_output(
    *,
    observation: torch.Tensor,
    action_features: torch.Tensor,
    state: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    observation_embedding = torch.tanh(
        functional.linear(
            observation,
            state["obs_encoder.0.weight"],
            state["obs_encoder.0.bias"],
        )
    )
    observation_embedding = torch.tanh(
        functional.linear(
            observation_embedding,
            state["obs_encoder.2.weight"],
            state["obs_encoder.2.bias"],
        )
    )
    action_embedding = torch.tanh(
        functional.linear(
            action_features,
            state["action_encoder.0.weight"],
            state["action_encoder.0.bias"],
        )
    )
    observation_expanded = observation_embedding.unsqueeze(0).expand(
        action_embedding.shape[0],
        -1,
    )
    combined = torch.cat(
        (
            observation_expanded,
            action_embedding,
            observation_expanded * action_embedding,
        ),
        dim=-1,
    )
    hidden = torch.tanh(
        functional.linear(
            combined,
            state["policy_head.0.weight"],
            state["policy_head.0.bias"],
        )
    )
    return functional.linear(
        hidden,
        state["policy_head.2.weight"],
        state["policy_head.2.bias"],
    ).squeeze(-1)


def _verify_exact_tensors(
    *,
    model_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    if set(model_state.keys()) != set(source_state.keys()):
        raise BrowserPackCheckpointError(f"{label} model tensor keys changed during reconstruction.")
    for key, source_tensor in source_state.items():
        actual_tensor = model_state[key].detach().cpu()
        if not torch.equal(actual_tensor, source_tensor.detach().cpu()):
            raise BrowserPackCheckpointError(
                f"{label} tensor {key!r} changed during checkpoint reconstruction."
            )


def _assert_output_parity(
    *,
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> float:
    if actual.shape != expected.shape:
        raise BrowserPackCheckpointError(
            f"{label} output shape mismatch: model={list(actual.shape)} "
            f"reference={list(expected.shape)}."
        )
    max_difference = float(torch.max(torch.abs(actual - expected)).item())
    if not math.isfinite(max_difference) or max_difference > OUTPUT_PARITY_ABSOLUTE_TOLERANCE:
        raise BrowserPackCheckpointError(
            f"{label} output parity failed: maxAbsDiff={max_difference} "
            f"tolerance={OUTPUT_PARITY_ABSOLUTE_TOLERANCE}."
        )
    return max_difference


def _checkpoint_metadata(
    *,
    manifest_spec: _ManifestSpec,
    manifest_path: Path,
    model_kind: str,
    manifest_sha256: str,
    weights_sha256: str,
) -> dict[str, object]:
    nested = manifest_spec.checkpoint_metadata.get(model_kind)
    metadata = (
        dict(_require_mapping(nested, f"manifest.source.checkpointMetadata.{model_kind}"))
        if nested is not None
        else {}
    )
    shared_step = manifest_spec.checkpoint_metadata.get("step")
    if "step" not in metadata and isinstance(shared_step, int) and not isinstance(shared_step, bool):
        metadata["step"] = shared_step
    source_checkpoint = (
        manifest_spec.source_value_checkpoint
        if model_kind == "value"
        else manifest_spec.source_opponent_checkpoint
    )
    metadata["browserPackReconstruction"] = {
        "packId": manifest_spec.pack_id,
        "label": manifest_spec.label,
        "createdAtUtc": manifest_spec.created_at_utc,
        "manifestPath": str(manifest_path),
        "weightsPath": str(manifest_spec.weights_path),
        "sourceRunId": manifest_spec.source_run_id,
        "sourceCheckpoint": source_checkpoint,
        "manifestSha256": manifest_sha256,
        "weightsSha256": weights_sha256,
    }
    return metadata


def _resolve_output_path(*, output_dir: Path, filename: str, label: str) -> Path:
    if not filename.strip():
        raise BrowserPackCheckpointError(f"{label} must be non-empty.")
    relative = Path(filename)
    if (
        relative.is_absolute()
        or relative.name != filename
        or relative.name in {"", ".", ".."}
    ):
        raise BrowserPackCheckpointError(f"{label} must be a plain filename, not a path.")
    return output_dir.resolve() / relative


def _read_json_object(*, path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise BrowserPackCheckpointError(f"Missing {label}: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BrowserPackCheckpointError(f"Could not read valid JSON from {label}: {path}") from error
    if not isinstance(raw, dict):
        raise BrowserPackCheckpointError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, object], raw)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise BrowserPackCheckpointError(f"{label} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise BrowserPackCheckpointError(f"{label} keys must be strings.")
    return cast(Mapping[str, object], value)


def _require_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise BrowserPackCheckpointError(f"{label} must be an array.")
    return cast(Sequence[object], value)


def _require_shape(value: object, label: str) -> tuple[int, ...]:
    raw = _require_sequence(value, label)
    if not raw:
        raise BrowserPackCheckpointError(f"{label} must not be empty.")
    shape: list[int] = []
    for index, dimension in enumerate(raw):
        shape.append(_require_positive_int(dimension, f"{label}[{index}]"))
    return tuple(shape)


def _require_string_tuple(value: object, label: str) -> tuple[str, ...]:
    raw = _require_sequence(value, label)
    result: list[str] = []
    for index, item in enumerate(raw):
        result.append(_require_non_empty_string(item, f"{label}[{index}]"))
    if not result:
        raise BrowserPackCheckpointError(f"{label} must not be empty.")
    return tuple(result)


def _validate_required_keys(
    *,
    declared: tuple[str, ...],
    expected: tuple[str, ...],
    label: str,
) -> None:
    if len(declared) != len(set(declared)):
        raise BrowserPackCheckpointError(f"{label} must not contain duplicate keys.")
    if set(declared) != set(expected):
        missing = sorted(set(expected) - set(declared))
        unexpected = sorted(set(declared) - set(expected))
        raise BrowserPackCheckpointError(
            f"{label} does not match the trainer model; "
            f"missing={missing} unexpected={unexpected}."
        )


def _require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BrowserPackCheckpointError(f"{label} must be a non-empty string.")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, label)


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BrowserPackCheckpointError(f"{label} must be an integer > 0.")
    return value


def _validate_schema_version(*, value: object, expected: int, label: str) -> None:
    actual = _require_positive_int(value, label)
    if actual != expected:
        raise BrowserPackCheckpointError(
            f"Unsupported {label}={actual}; expected {expected}."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
