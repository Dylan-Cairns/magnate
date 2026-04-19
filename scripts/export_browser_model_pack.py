from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import torch
from trainer.encoding import ENCODING_VERSION, OBSERVATION_DIM
from trainer.td.checkpoint import TD_VALUE_CHECKPOINT_TYPE, load_value_checkpoint

INDEX_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
WEIGHTS_SCHEMA_VERSION = 1
MODEL_TYPE = "td-value-v1"

REQUIRED_STATE_DICT_KEYS: Sequence[str] = (
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "encoder.4.weight",
    "encoder.4.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a TD value checkpoint into static browser model-pack files "
            "(manifest + JSON weights) for web app use."
        )
    )
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        default=None,
        help="Explicit value checkpoint path (*.pt).",
    )
    parser.add_argument(
        "--latest-promoted",
        action="store_true",
        help=(
            "Resolve checkpoint from latest promoted loop summary under --artifact-root "
            "(same promotion selection semantics as overnight warm-start lookup)."
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
    parser.set_defaults(set_default=True)
    parser.add_argument(
        "--set-default",
        dest="set_default",
        action="store_true",
        help="Set exported pack as default in index.json (default: enabled).",
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

    resolution = resolve_checkpoint(
        explicit_checkpoint=args.value_checkpoint,
        latest_promoted=args.latest_promoted,
        artifact_root=args.artifact_root,
    )

    result = export_value_checkpoint_pack(
        checkpoint_path=resolution["checkpointPath"],
        output_root=args.output_root,
        pack_id=args.pack_id,
        label=args.label,
        overwrite=args.overwrite,
        set_default=args.set_default,
        source_run_id=resolution.get("sourceRunId"),
    )

    print(json.dumps(result, indent=2))
    return 0


def export_value_checkpoint_pack(
    *,
    checkpoint_path: Path,
    output_root: Path,
    pack_id: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
    set_default: bool = True,
    source_run_id: str | None = None,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise SystemExit(f"Value checkpoint not found: {checkpoint_path}")

    model, raw_payload = load_value_checkpoint(path=checkpoint_path)
    state_dict = model.state_dict()
    _validate_checkpoint_payload(raw_payload=raw_payload, state_dict=state_dict)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    normalized_pack_id = (
        _slug(pack_id)
        if pack_id is not None
        else _default_pack_id(
            checkpoint_path=checkpoint_path,
            source_run_id=source_run_id,
            stamp=stamp,
            metadata=raw_payload.get("metadata"),
        )
    )
    if not normalized_pack_id:
        raise SystemExit("Resolved empty pack id.")

    pack_label = (
        label.strip()
        if isinstance(label, str) and label.strip()
        else _default_label(
            checkpoint_path=checkpoint_path,
            source_run_id=source_run_id,
            metadata=raw_payload.get("metadata"),
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
    weights_payload = _weights_payload(state_dict)
    manifest_payload = _manifest_payload(
        pack_id=normalized_pack_id,
        label=pack_label,
        created_at=created_at,
        checkpoint_path=checkpoint_path,
        source_run_id=source_run_id,
        raw_payload=raw_payload,
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
            "sourceValueCheckpoint": str(checkpoint_path),
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
        "sourceValueCheckpoint": str(checkpoint_path),
    }


def resolve_checkpoint(
    *,
    explicit_checkpoint: Path | None,
    latest_promoted: bool,
    artifact_root: Path,
) -> Dict[str, Any]:
    if explicit_checkpoint is not None:
        return {
            "checkpointPath": explicit_checkpoint,
            "sourceRunId": None,
        }

    if not latest_promoted:
        raise SystemExit("Either --value-checkpoint or --latest-promoted is required.")

    return _latest_promoted_value_checkpoint(artifact_root=artifact_root)


def _latest_promoted_value_checkpoint(*, artifact_root: Path) -> Dict[str, Any]:
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
        if not isinstance(value_raw, str) or not value_raw.strip():
            continue

        value_path = Path(value_raw)
        latest = {
            "checkpointPath": value_path,
            "sourceRunId": summary_path.parent.name,
        }

    if latest is None:
        raise SystemExit(
            f"No promoted loop summary with value checkpoint found under {artifact_root}."
        )
    return latest


def _validate_args(args: argparse.Namespace) -> None:
    if args.value_checkpoint is None and not bool(args.latest_promoted):
        raise SystemExit("Either --value-checkpoint or --latest-promoted is required.")
    if args.value_checkpoint is not None and bool(args.latest_promoted):
        raise SystemExit("Use either --value-checkpoint or --latest-promoted, not both.")
    if args.pack_id is not None and not args.pack_id.strip():
        raise SystemExit("--pack-id must be non-empty when provided.")
    if args.label is not None and not args.label.strip():
        raise SystemExit("--label must be non-empty when provided.")


def _validate_checkpoint_payload(
    *,
    raw_payload: Mapping[str, Any],
    state_dict: MutableMapping[str, torch.Tensor],
) -> None:
    checkpoint_type = str(raw_payload.get("checkpointType", ""))
    if checkpoint_type != TD_VALUE_CHECKPOINT_TYPE:
        raise SystemExit(
            f"Unsupported checkpointType {checkpoint_type!r}; "
            f"expected {TD_VALUE_CHECKPOINT_TYPE!r}."
        )

    encoding_version = int(raw_payload.get("encodingVersion"))
    if encoding_version != ENCODING_VERSION:
        raise SystemExit(
            "Encoding version mismatch. "
            f"checkpoint={encoding_version} expected={ENCODING_VERSION}."
        )

    observation_dim = int(raw_payload.get("observationDim"))
    if observation_dim != OBSERVATION_DIM:
        raise SystemExit(
            "Observation dim mismatch against browser encoding. "
            f"checkpoint={observation_dim} expected={OBSERVATION_DIM}."
        )

    for key in REQUIRED_STATE_DICT_KEYS:
        if key not in state_dict:
            raise SystemExit(f"Checkpoint stateDict is missing required tensor: {key}")


def _weights_payload(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    tensors: Dict[str, Any] = {}
    for key in REQUIRED_STATE_DICT_KEYS:
        tensor = state_dict[key].detach().cpu().to(torch.float32)
        flat = tensor.reshape(-1).tolist()
        tensors[key] = {
            "shape": list(tensor.shape),
            "values": [float(value) for value in flat],
        }
    return {
        "schemaVersion": WEIGHTS_SCHEMA_VERSION,
        "tensors": tensors,
    }


def _manifest_payload(
    *,
    pack_id: str,
    label: str,
    created_at: str,
    checkpoint_path: Path,
    source_run_id: str | None,
    raw_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = raw_payload.get("metadata")
    safe_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
    return {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "packId": pack_id,
        "label": label,
        "createdAtUtc": created_at,
        "model": {
            "modelType": MODEL_TYPE,
            "checkpointType": str(raw_payload.get("checkpointType")),
            "encodingVersion": int(raw_payload.get("encodingVersion")),
            "observationDim": int(raw_payload.get("observationDim")),
            "hiddenDim": int(raw_payload.get("hiddenDim")),
            "weightsPath": "weights.json",
            "requiredStateDictKeys": list(REQUIRED_STATE_DICT_KEYS),
        },
        "source": {
            "runId": source_run_id,
            "valueCheckpoint": str(checkpoint_path),
            "checkpointMetadata": safe_metadata,
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
    checkpoint_path: Path,
    source_run_id: str | None,
    stamp: str,
    metadata: Any,
) -> str:
    step_value: str | None = None
    if isinstance(metadata, Mapping):
        raw_step = metadata.get("step")
        if isinstance(raw_step, int):
            step_value = f"step-{raw_step:07d}"
    parts = [source_run_id or checkpoint_path.stem, step_value or "value", stamp]
    return _slug("-".join(part for part in parts if part))


def _default_label(
    *,
    checkpoint_path: Path,
    source_run_id: str | None,
    metadata: Any,
) -> str:
    run_label = source_run_id or checkpoint_path.parent.name
    step_text = ""
    if isinstance(metadata, Mapping):
        raw_step = metadata.get("step")
        if isinstance(raw_step, int):
            step_text = f" step {raw_step}"
    return f"TD Value {run_label}{step_text}".strip()


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
