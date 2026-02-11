from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch
from trainer.encoding import ACTION_FEATURE_DIM, ENCODING_VERSION, OBSERVATION_DIM
from trainer.td.checkpoint import (
    TD_OPPONENT_CHECKPOINT_TYPE,
    TD_VALUE_CHECKPOINT_TYPE,
    load_opponent_checkpoint,
    load_value_checkpoint,
)

from scripts.checkpoint_manifest import (
    default_manifest_path_for_artifact_dir,
    load_default_warm_start,
)

INDEX_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
WEIGHTS_SCHEMA_VERSION = 1
MODEL_TYPE = "td-search-v1"

VALUE_REQUIRED_STATE_DICT_KEYS: Sequence[str] = (
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "encoder.4.weight",
    "encoder.4.bias",
)

OPPONENT_REQUIRED_STATE_DICT_KEYS: Sequence[str] = (
    "obs_encoder.0.weight",
    "obs_encoder.0.bias",
    "obs_encoder.2.weight",
    "obs_encoder.2.bias",
    "action_encoder.0.weight",
    "action_encoder.0.bias",
    "policy_head.0.weight",
    "policy_head.0.bias",
    "policy_head.2.weight",
    "policy_head.2.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export TD value+opponent checkpoints into static browser td-search model-pack files."
        )
    )
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help="Value checkpoint path (*.pt).",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Opponent checkpoint path (*.pt).",
    )
    parser.add_argument(
        "--latest-promoted",
        action="store_true",
        help=(
            "Resolve value+opponent checkpoints from latest promoted loop summary "
            "under --artifact-root."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/td_loops"),
        help="TD loop artifact root used with --latest-promoted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("public/model-packs"),
        help="Output root for browser model packs.",
    )
    parser.add_argument(
        "--pack-id",
        type=str,
        default=None,
        help="Optional stable pack id; generated from source if omitted.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional human-readable label; generated if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite manifest/weights for an existing pack id.",
    )
    parser.set_defaults(set_default=False)
    parser.add_argument(
        "--set-default",
        dest="set_default",
        action="store_true",
        help="Set exported pack as default in index.json (default: disabled).",
    )
    parser.add_argument(
        "--no-set-default",
        dest="set_default",
        action="store_false",
        help="Do not change defaultPackId in index.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)

    resolution = resolve_checkpoints(
        value_checkpoint=args.value_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        latest_promoted=args.latest_promoted,
        artifact_root=args.artifact_root,
    )

    result = export_td_search_checkpoint_pack(
        value_checkpoint_path=resolution["valueCheckpointPath"],
        opponent_checkpoint_path=resolution["opponentCheckpointPath"],
        output_root=args.output_root,
        pack_id=args.pack_id,
        label=args.label,
        overwrite=args.overwrite,
        set_default=args.set_default,
        source_run_id=resolution.get("sourceRunId"),
    )

    print(json.dumps(result, indent=2))
    return 0


def export_td_search_checkpoint_pack(
    *,
    value_checkpoint_path: Path,
    opponent_checkpoint_path: Path,
    output_root: Path,
    pack_id: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
    set_default: bool = False,
    source_run_id: str | None = None,
) -> Dict[str, Any]:
    if not value_checkpoint_path.exists():
        raise SystemExit(f"Value checkpoint not found: {value_checkpoint_path}")
    if not opponent_checkpoint_path.exists():
        raise SystemExit(f"Opponent checkpoint not found: {opponent_checkpoint_path}")

    value_model, value_payload = load_value_checkpoint(path=value_checkpoint_path)
    opponent_model, opponent_payload = load_opponent_checkpoint(path=opponent_checkpoint_path)
    value_state = value_model.state_dict()
    opponent_state = opponent_model.state_dict()

    _validate_value_checkpoint(raw_payload=value_payload, state_dict=value_state)
    _validate_opponent_checkpoint(raw_payload=opponent_payload, state_dict=opponent_state)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    normalized_pack_id = (
        _slug(pack_id)
        if pack_id is not None
        else _default_pack_id(
            source_run_id=source_run_id,
            value_checkpoint_path=value_checkpoint_path,
            stamp=stamp,
            metadata=value_payload.get("metadata"),
        )
    )
    if not normalized_pack_id:
        raise SystemExit("Resolved empty pack id.")

    pack_label = (
        label.strip()
        if isinstance(label, str) and label.strip()
        else _default_label(
            source_run_id=source_run_id,
            value_checkpoint_path=value_checkpoint_path,
            metadata=value_payload.get("metadata"),
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    pack_dir = output_root / normalized_pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pack_dir / "manifest.json"
    weights_path = pack_dir / "weights.json"
    if not overwrite and (manifest_path.exists() or weights_path.exists()):
        raise SystemExit(
            f"Pack already exists for id {normalized_pack_id!r}. "
            "Use --overwrite to replace manifest/weights."
        )

    created_at = datetime.now(timezone.utc).isoformat()
    weights_payload = _weights_payload(value_state=value_state, opponent_state=opponent_state)
    manifest_payload = _manifest_payload(
        pack_id=normalized_pack_id,
        label=pack_label,
        created_at=created_at,
        source_run_id=source_run_id,
        value_checkpoint_path=value_checkpoint_path,
        opponent_checkpoint_path=opponent_checkpoint_path,
        value_payload=value_payload,
        opponent_payload=opponent_payload,
    )

    weights_path.write_text(json.dumps(weights_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    index_path = output_root / "index.json"
    index_payload = _read_or_init_index(index_path=index_path)
    _upsert_index_entry(
        index=index_payload,
        entry={
            "id": normalized_pack_id,
            "label": pack_label,
            "modelType": MODEL_TYPE,
            "manifestPath": f"model-packs/{normalized_pack_id}/manifest.json",
            "createdAtUtc": created_at,
            "sourceRunId": source_run_id,
            "sourceValueCheckpoint": str(value_checkpoint_path),
            "sourceOpponentCheckpoint": str(opponent_checkpoint_path),
        },
    )
    index_payload["generatedAtUtc"] = created_at
    if set_default:
        index_payload["defaultPackId"] = normalized_pack_id
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    return {
        "packId": normalized_pack_id,
        "label": pack_label,
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "index": str(index_path),
        "defaultPackId": index_payload.get("defaultPackId"),
        "sourceRunId": source_run_id,
        "sourceValueCheckpoint": str(value_checkpoint_path),
        "sourceOpponentCheckpoint": str(opponent_checkpoint_path),
    }


def resolve_checkpoints(
    *,
    value_checkpoint: Path | None,
    opponent_checkpoint: Path | None,
    latest_promoted: bool,
    artifact_root: Path,
) -> Dict[str, Any]:
    if value_checkpoint is not None or opponent_checkpoint is not None:
        if value_checkpoint is None or opponent_checkpoint is None:
            raise SystemExit(
                "Both --value-checkpoint and --opponent-checkpoint are required "
                "when providing explicit checkpoint paths."
            )
        return {
            "valueCheckpointPath": value_checkpoint,
            "opponentCheckpointPath": opponent_checkpoint,
            "sourceRunId": None,
        }

    if not latest_promoted:
        raise SystemExit(
            "Provide --value-checkpoint + --opponent-checkpoint "
            "or use --latest-promoted."
        )
    return _latest_promoted_checkpoint_pair(artifact_root=artifact_root)


def _latest_promoted_checkpoint_pair(*, artifact_root: Path) -> Dict[str, Any]:
    manifest_path = default_manifest_path_for_artifact_dir(artifact_root)
    if manifest_path is not None:
        manifest_checkpoint = load_default_warm_start(
            manifest_path=manifest_path,
            require_paths=True,
        )
        if manifest_checkpoint is not None and manifest_checkpoint.status == "promoted":
            return {
                "valueCheckpointPath": manifest_checkpoint.value_path,
                "opponentCheckpointPath": manifest_checkpoint.opponent_path,
                "sourceRunId": manifest_checkpoint.source_run_id,
            }

    if not artifact_root.exists():
        raise SystemExit(f"Artifact root does not exist: {artifact_root}")

    latest: Dict[str, Any] | None = None
    for summary_path in sorted(artifact_root.glob("*/loop.summary.json")):
        payload = _read_json(path=summary_path, label=f"loop summary {summary_path}")
        promotion = payload.get("promotion")
        if not isinstance(promotion, Mapping) or not bool(promotion.get("promoted")):
            continue

        chunks = payload.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            continue
        tail = chunks[-1]
        if not isinstance(tail, Mapping):
            continue
        latest_checkpoint = tail.get("latestCheckpoint")
        if not isinstance(latest_checkpoint, Mapping):
            continue
        value_raw = latest_checkpoint.get("value")
        opponent_raw = latest_checkpoint.get("opponent")
        if (
            not isinstance(value_raw, str)
            or not value_raw.strip()
            or not isinstance(opponent_raw, str)
            or not opponent_raw.strip()
        ):
            continue

        latest = {
            "valueCheckpointPath": Path(value_raw),
            "opponentCheckpointPath": Path(opponent_raw),
            "sourceRunId": summary_path.parent.name,
        }

    if latest is None:
        raise SystemExit(
            f"No promoted loop summary with value+opponent checkpoints found under {artifact_root}."
        )
    return latest


def _validate_args(args: argparse.Namespace) -> None:
    using_explicit = args.value_checkpoint is not None or args.opponent_checkpoint is not None
    if using_explicit and args.latest_promoted:
        raise SystemExit(
            "Use explicit checkpoint paths or --latest-promoted, not both."
        )
    if not using_explicit and not bool(args.latest_promoted):
        raise SystemExit(
            "Provide --value-checkpoint + --opponent-checkpoint "
            "or use --latest-promoted."
        )
    if args.pack_id is not None and not args.pack_id.strip():
        raise SystemExit("--pack-id must be non-empty when provided.")
    if args.label is not None and not args.label.strip():
        raise SystemExit("--label must be non-empty when provided.")


def _validate_value_checkpoint(
    *,
    raw_payload: Mapping[str, Any],
    state_dict: MutableMapping[str, torch.Tensor],
) -> None:
    checkpoint_type = str(raw_payload.get("checkpointType", ""))
    if checkpoint_type != TD_VALUE_CHECKPOINT_TYPE:
        raise SystemExit(
            f"Unsupported value checkpointType {checkpoint_type!r}; "
            f"expected {TD_VALUE_CHECKPOINT_TYPE!r}."
        )

    encoding_version = int(raw_payload.get("encodingVersion"))
    if encoding_version != ENCODING_VERSION:
        raise SystemExit(
            f"Value encoding mismatch: checkpoint={encoding_version} expected={ENCODING_VERSION}."
        )

    observation_dim = int(raw_payload.get("observationDim"))
    if observation_dim != OBSERVATION_DIM:
        raise SystemExit(
            f"Value observation dim mismatch: checkpoint={observation_dim} expected={OBSERVATION_DIM}."
        )

    for key in VALUE_REQUIRED_STATE_DICT_KEYS:
        if key not in state_dict:
            raise SystemExit(f"Value checkpoint stateDict is missing tensor: {key}")


def _validate_opponent_checkpoint(
    *,
    raw_payload: Mapping[str, Any],
    state_dict: MutableMapping[str, torch.Tensor],
) -> None:
    checkpoint_type = str(raw_payload.get("checkpointType", ""))
    if checkpoint_type != TD_OPPONENT_CHECKPOINT_TYPE:
        raise SystemExit(
            f"Unsupported opponent checkpointType {checkpoint_type!r}; "
            f"expected {TD_OPPONENT_CHECKPOINT_TYPE!r}."
        )

    encoding_version = int(raw_payload.get("encodingVersion"))
    if encoding_version != ENCODING_VERSION:
        raise SystemExit(
            f"Opponent encoding mismatch: checkpoint={encoding_version} expected={ENCODING_VERSION}."
        )

    observation_dim = int(raw_payload.get("observationDim"))
    if observation_dim != OBSERVATION_DIM:
        raise SystemExit(
            f"Opponent observation dim mismatch: checkpoint={observation_dim} expected={OBSERVATION_DIM}."
        )
    action_feature_dim = int(raw_payload.get("actionFeatureDim"))
    if action_feature_dim != ACTION_FEATURE_DIM:
        raise SystemExit(
            "Opponent action feature dim mismatch: "
            f"checkpoint={action_feature_dim} expected={ACTION_FEATURE_DIM}."
        )

    for key in OPPONENT_REQUIRED_STATE_DICT_KEYS:
        if key not in state_dict:
            raise SystemExit(f"Opponent checkpoint stateDict is missing tensor: {key}")


def _weights_payload(
    *,
    value_state: Mapping[str, torch.Tensor],
    opponent_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    value_tensors: Dict[str, Any] = {}
    for key in VALUE_REQUIRED_STATE_DICT_KEYS:
        tensor = value_state[key].detach().cpu().to(torch.float32)
        value_tensors[key] = {
            "shape": list(tensor.shape),
            "values": [float(value) for value in tensor.reshape(-1).tolist()],
        }

    opponent_tensors: Dict[str, Any] = {}
    for key in OPPONENT_REQUIRED_STATE_DICT_KEYS:
        tensor = opponent_state[key].detach().cpu().to(torch.float32)
        opponent_tensors[key] = {
            "shape": list(tensor.shape),
            "values": [float(value) for value in tensor.reshape(-1).tolist()],
        }

    return {
        "schemaVersion": WEIGHTS_SCHEMA_VERSION,
        "valueTensors": value_tensors,
        "opponentTensors": opponent_tensors,
    }


def _manifest_payload(
    *,
    pack_id: str,
    label: str,
    created_at: str,
    source_run_id: str | None,
    value_checkpoint_path: Path,
    opponent_checkpoint_path: Path,
    value_payload: Mapping[str, Any],
    opponent_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    value_metadata = value_payload.get("metadata")
    opponent_metadata = opponent_payload.get("metadata")
    step_value = None
    if isinstance(value_metadata, Mapping) and isinstance(value_metadata.get("step"), int):
        step_value = int(value_metadata.get("step"))
    elif isinstance(opponent_metadata, Mapping) and isinstance(
        opponent_metadata.get("step"), int
    ):
        step_value = int(opponent_metadata.get("step"))

    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "packId": pack_id,
        "label": label,
        "createdAtUtc": created_at,
        "model": {
            "modelType": MODEL_TYPE,
            "weightsPath": "weights.json",
            "value": {
                "checkpointType": str(value_payload.get("checkpointType")),
                "encodingVersion": int(value_payload.get("encodingVersion")),
                "observationDim": int(value_payload.get("observationDim")),
                "hiddenDim": int(value_payload.get("hiddenDim")),
                "requiredStateDictKeys": list(VALUE_REQUIRED_STATE_DICT_KEYS),
            },
            "opponent": {
                "checkpointType": str(opponent_payload.get("checkpointType")),
                "encodingVersion": int(opponent_payload.get("encodingVersion")),
                "observationDim": int(opponent_payload.get("observationDim")),
                "actionFeatureDim": int(opponent_payload.get("actionFeatureDim")),
                "hiddenDim": int(opponent_payload.get("hiddenDim")),
                "requiredStateDictKeys": list(OPPONENT_REQUIRED_STATE_DICT_KEYS),
            },
        },
        "source": {
            "runId": source_run_id,
            "valueCheckpoint": str(value_checkpoint_path),
            "opponentCheckpoint": str(opponent_checkpoint_path),
            "checkpointMetadata": {
                "step": step_value,
                "value": dict(value_metadata) if isinstance(value_metadata, Mapping) else {},
                "opponent": (
                    dict(opponent_metadata) if isinstance(opponent_metadata, Mapping) else {}
                ),
            },
        },
    }


def _read_or_init_index(*, index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        return {
            "schemaVersion": INDEX_SCHEMA_VERSION,
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "defaultPackId": None,
            "packs": [],
        }

    payload = _read_json(path=index_path, label=f"model-pack index {index_path}")
    if int(payload.get("schemaVersion", -1)) != INDEX_SCHEMA_VERSION:
        raise SystemExit(
            f"Unsupported index schema version in {index_path}: "
            f"{payload.get('schemaVersion')}"
        )
    packs = payload.get("packs")
    if not isinstance(packs, list):
        raise SystemExit(f"model-pack index packs must be a list: {index_path}")
    payload["packs"] = packs
    return payload


def _upsert_index_entry(*, index: Dict[str, Any], entry: Dict[str, Any]) -> None:
    packs = index.get("packs")
    if not isinstance(packs, list):
        raise SystemExit("Index packs payload must be a list.")

    for idx, existing in enumerate(packs):
        if isinstance(existing, Mapping) and existing.get("id") == entry["id"]:
            packs[idx] = entry
            break
    else:
        packs.append(entry)

    packs.sort(key=lambda row: str((row or {}).get("createdAtUtc", "")), reverse=True)


def _default_pack_id(
    *,
    source_run_id: str | None,
    value_checkpoint_path: Path,
    stamp: str,
    metadata: Any,
) -> str:
    step_value: str | None = None
    if isinstance(metadata, Mapping):
        raw_step = metadata.get("step")
        if isinstance(raw_step, int):
            step_value = f"step-{raw_step:07d}"
    parts = [source_run_id or value_checkpoint_path.stem, "td-search", step_value or "step", stamp]
    return _slug("-".join(part for part in parts if part))


def _default_label(
    *,
    source_run_id: str | None,
    value_checkpoint_path: Path,
    metadata: Any,
) -> str:
    run_label = source_run_id or value_checkpoint_path.parent.name
    step_text = ""
    if isinstance(metadata, Mapping):
        raw_step = metadata.get("step")
        if isinstance(raw_step, int):
            step_text = f" step {raw_step}"
    return f"TD Search {run_label}{step_text}".strip()


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-")


def _read_json(*, path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
