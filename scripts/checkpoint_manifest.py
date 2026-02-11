from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

MANIFEST_SCHEMA_VERSION = 2
DEFAULT_MANIFEST_RELATIVE_PATH = Path("models/td_checkpoints/manifest.json")


@dataclass(frozen=True)
class ManifestCheckpoint:
    key: str
    status: str
    label: str
    source_run_id: str
    source_loop_summary: str | None
    source_chunk: str | None
    step: int | None
    generated_at_utc: str
    value_path: Path
    opponent_path: Path


def default_manifest_path_for_artifact_dir(artifact_dir: Path) -> Path | None:
    """Return the repo manifest path when artifact_dir is the standard artifacts/td_loops."""
    normalized = Path(artifact_dir)
    if normalized.name != "td_loops" or normalized.parent.name != "artifacts":
        return None
    return normalized.parent.parent / DEFAULT_MANIFEST_RELATIVE_PATH


def load_manifest_checkpoints(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_RELATIVE_PATH,
    keys: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
    require_paths: bool = True,
) -> list[ManifestCheckpoint]:
    payload = _load_manifest_payload(manifest_path)
    if payload is None:
        return []

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        return []

    key_filter = set(keys) if keys is not None else None
    status_filter = {status.strip().lower() for status in statuses} if statuses is not None else None
    out: list[ManifestCheckpoint] = []
    for key, raw in checkpoints.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            continue
        if key_filter is not None and key not in key_filter:
            continue
        checkpoint = _checkpoint_from_manifest_entry(
            manifest_path=manifest_path,
            manifest_payload=payload,
            key=key,
            raw=raw,
        )
        if checkpoint is None:
            continue
        if status_filter is not None and checkpoint.status.strip().lower() not in status_filter:
            continue
        if require_paths and (
            not checkpoint.value_path.exists() or not checkpoint.opponent_path.exists()
        ):
            continue
        out.append(checkpoint)
    return out


def load_default_warm_start(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_RELATIVE_PATH,
    require_paths: bool = True,
) -> ManifestCheckpoint | None:
    payload = _load_manifest_payload(manifest_path)
    if payload is None:
        return None
    default_key = _default_warm_start_key(payload)
    if default_key is None:
        return None
    rows = load_manifest_checkpoints(
        manifest_path=manifest_path,
        keys=[default_key],
        require_paths=require_paths,
    )
    return rows[0] if rows else None


def load_manifest_opponent_pool(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_RELATIVE_PATH,
    max_entries: int | None = None,
    require_paths: bool = True,
) -> list[ManifestCheckpoint]:
    payload = _load_manifest_payload(manifest_path)
    if payload is None:
        return []

    pool_keys = _opponent_pool_keys(payload)
    if not pool_keys:
        default_key = _default_warm_start_key(payload)
        if default_key is not None:
            pool_keys = [default_key]

    rows: list[ManifestCheckpoint] = []
    seen: set[str] = set()
    for key in pool_keys:
        if key in seen:
            continue
        seen.add(key)
        matches = load_manifest_checkpoints(
            manifest_path=manifest_path,
            keys=[key],
            require_paths=require_paths,
        )
        if not matches:
            continue
        checkpoint = matches[0]
        if checkpoint.status != "promoted":
            continue
        rows.append(checkpoint)

    if max_entries is not None:
        rows = rows[:max_entries]
    return rows


def update_manifest_promoted_checkpoint(
    *,
    manifest_path: Path,
    key: str,
    value_path: Path,
    opponent_path: Path,
    source_run_id: str,
    source_loop_summary: Path | None = None,
    source_chunk: str | None = None,
    source_eval_artifacts: Sequence[Path] | None = None,
    step: int | None = None,
    label: str | None = None,
    generated_at_utc: str | None = None,
    set_default: bool = False,
    add_to_opponent_pool: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    key = _normalize_key(key)
    manifest_path = Path(manifest_path)
    repo_root = _repo_root_from_manifest_path(manifest_path)
    payload = _load_manifest_payload(manifest_path) or {}
    existing = payload.get("checkpoints")
    checkpoints: Dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    if key in checkpoints and not force:
        raise SystemExit(
            f"Manifest checkpoint key already exists: {key}. Pass --force to overwrite."
        )

    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "status": "promoted",
        "label": label or f"Promoted checkpoint {key}",
        "sourceRunId": source_run_id,
        "step": step,
        "generatedAtUtc": generated,
        "value": _project_relative_path(repo_root=repo_root, path=value_path),
        "opponent": _project_relative_path(repo_root=repo_root, path=opponent_path),
    }
    if source_loop_summary is not None:
        entry["sourceLoopSummary"] = _project_relative_path(
            repo_root=repo_root,
            path=source_loop_summary,
        )
    if source_chunk:
        entry["sourceChunk"] = source_chunk
    if source_eval_artifacts:
        entry["sourceEvalArtifacts"] = [
            _project_relative_path(repo_root=repo_root, path=path)
            for path in source_eval_artifacts
        ]

    checkpoints[key] = {item_key: item_value for item_key, item_value in entry.items() if item_value is not None}

    payload["schemaVersion"] = MANIFEST_SCHEMA_VERSION
    payload.setdefault("generatedAtUtc", generated)
    payload["updatedAtUtc"] = generated
    payload["checkpoints"] = checkpoints
    if set_default:
        payload["defaultWarmStart"] = key
    elif "defaultWarmStart" not in payload:
        payload["defaultWarmStart"] = key

    pool = _opponent_pool_keys(payload)
    if add_to_opponent_pool:
        pool = [key] + [existing_key for existing_key in pool if existing_key != key]
    payload["opponentPool"] = pool

    _write_json_atomic(manifest_path, payload)
    return payload


def normalized_checkpoint_key(value: str) -> str:
    return _normalize_key(value)


def _load_manifest_payload(path: Path) -> Dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid checkpoint manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Checkpoint manifest must be a JSON object: {path}")
    return payload


def _checkpoint_from_manifest_entry(
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    key: str,
    raw: Mapping[str, Any],
) -> ManifestCheckpoint | None:
    value_raw = raw.get("value")
    opponent_raw = raw.get("opponent")
    if not isinstance(value_raw, str) or not value_raw.strip():
        return None
    if not isinstance(opponent_raw, str) or not opponent_raw.strip():
        return None

    status = raw.get("status")
    if isinstance(status, str) and status.strip():
        normalized_status = status.strip().lower()
    elif key == "promoted" or key == _default_warm_start_key(manifest_payload):
        normalized_status = "promoted"
    else:
        normalized_status = "experimental"

    return ManifestCheckpoint(
        key=key,
        status=normalized_status,
        label=str(raw.get("label") or key),
        source_run_id=str(raw.get("sourceRunId") or key),
        source_loop_summary=(
            str(raw.get("sourceLoopSummary"))
            if isinstance(raw.get("sourceLoopSummary"), str)
            else None
        ),
        source_chunk=(
            str(raw.get("sourceChunk")) if isinstance(raw.get("sourceChunk"), str) else None
        ),
        step=_optional_int(raw.get("step")),
        generated_at_utc=str(raw.get("generatedAtUtc") or manifest_payload.get("generatedAtUtc") or ""),
        value_path=_resolve_manifest_path(manifest_path=manifest_path, path_value=value_raw),
        opponent_path=_resolve_manifest_path(manifest_path=manifest_path, path_value=opponent_raw),
    )


def _default_warm_start_key(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("defaultWarmStart")
    if isinstance(value, str) and value.strip():
        return value.strip()
    checkpoints = payload.get("checkpoints")
    if isinstance(checkpoints, Mapping) and "promoted" in checkpoints:
        return "promoted"
    return None


def _opponent_pool_keys(payload: Mapping[str, Any]) -> list[str]:
    raw = payload.get("opponentPool")
    if not isinstance(raw, list):
        return []
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


def _resolve_manifest_path(*, manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _repo_root_from_manifest_path(manifest_path) / path


def _repo_root_from_manifest_path(manifest_path: Path) -> Path:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.parent.parent.parent
    except IndexError:
        return Path.cwd()


def _project_relative_path(*, repo_root: Path, path: Path) -> str:
    raw = Path(path)
    if not raw.is_absolute():
        raw = (Path.cwd() / raw).resolve()
    else:
        raw = raw.resolve()
    try:
        return raw.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(raw)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise SystemExit("Manifest checkpoint key must not be empty.")
    return normalized


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)
