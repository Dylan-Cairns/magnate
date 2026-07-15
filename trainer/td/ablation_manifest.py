from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenReplaySplit:
    shards_dir: Path
    training_keys: tuple[str, ...]
    validation_keys: tuple[str, ...]
    value_paths: dict[str, Path]
    opponent_paths: dict[str, Path]
    inventory_sha256: str
    membership_sha256: str
    value_content_sha256: str
    opponent_content_sha256: str
    training_value_content_sha256: str
    training_opponent_content_sha256: str
    validation_value_content_sha256: str
    validation_opponent_content_sha256: str
    value_bytes: int
    opponent_bytes: int


def resolve_frozen_replay_split(
    *,
    shards_dir: Path,
    salt: str,
    validation_shards: int,
) -> FrozenReplaySplit:
    if not shards_dir.exists() or not shards_dir.is_dir():
        raise ValueError(f"Replay shards directory not found: {shards_dir}")
    if not salt:
        raise ValueError("Replay split salt must not be empty.")

    value_paths = {
        path.name.removesuffix(".value.jsonl"): path.resolve()
        for path in shards_dir.glob("*.value.jsonl")
    }
    opponent_paths = {
        path.name.removesuffix(".opponent.jsonl"): path.resolve()
        for path in shards_dir.glob("*.opponent.jsonl")
    }
    if not value_paths:
        raise ValueError(f"No value replay shards found: {shards_dir}")
    if value_paths.keys() != opponent_paths.keys():
        missing_value = sorted(opponent_paths.keys() - value_paths.keys())
        missing_opponent = sorted(value_paths.keys() - opponent_paths.keys())
        raise ValueError(
            "Value/opponent replay shards are not paired. "
            f"missingValue={missing_value} missingOpponent={missing_opponent}"
        )
    if validation_shards <= 0 or validation_shards >= len(value_paths):
        raise ValueError("validation_shards must leave non-empty train and validation sets.")

    keys = sorted(value_paths)
    ranked = sorted(
        keys,
        key=lambda key: (
            hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest(),
            key,
        ),
    )
    validation_keys = tuple(sorted(ranked[:validation_shards]))
    training_keys = tuple(sorted(ranked[validation_shards:]))

    value_file_sha256 = {key: sha256_file(value_paths[key]) for key in keys}
    opponent_file_sha256 = {key: sha256_file(opponent_paths[key]) for key in keys}
    inventory_payload = [
        {
            "key": key,
            "opponentBytes": opponent_paths[key].stat().st_size,
            "valueBytes": value_paths[key].stat().st_size,
        }
        for key in keys
    ]
    membership_payload = {
        "training": training_keys,
        "validation": validation_keys,
    }
    return FrozenReplaySplit(
        shards_dir=shards_dir.resolve(),
        training_keys=training_keys,
        validation_keys=validation_keys,
        value_paths=value_paths,
        opponent_paths=opponent_paths,
        inventory_sha256=_canonical_sha256(inventory_payload),
        membership_sha256=_canonical_sha256(membership_payload),
        value_content_sha256=_replay_content_sha256(
            keys=keys,
            file_sha256=value_file_sha256,
        ),
        opponent_content_sha256=_replay_content_sha256(
            keys=keys,
            file_sha256=opponent_file_sha256,
        ),
        training_value_content_sha256=_replay_content_sha256(
            keys=training_keys,
            file_sha256=value_file_sha256,
        ),
        training_opponent_content_sha256=_replay_content_sha256(
            keys=training_keys,
            file_sha256=opponent_file_sha256,
        ),
        validation_value_content_sha256=_replay_content_sha256(
            keys=validation_keys,
            file_sha256=value_file_sha256,
        ),
        validation_opponent_content_sha256=_replay_content_sha256(
            keys=validation_keys,
            file_sha256=opponent_file_sha256,
        ),
        value_bytes=sum(path.stat().st_size for path in value_paths.values()),
        opponent_bytes=sum(path.stat().st_size for path in opponent_paths.values()),
    )


def count_jsonl_rows(paths: tuple[Path, ...] | list[Path]) -> int:
    rows = 0
    for path in paths:
        with path.open("rb") as handle:
            rows += sum(1 for line in handle if line.strip())
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def replay_content_sha256(paths: tuple[Path, ...] | list[Path]) -> str:
    """Fingerprint ordered replay shard contents without binding absolute paths."""
    if not paths:
        raise ValueError("Replay content fingerprint requires at least one path.")
    keyed_paths: list[tuple[str, Path]] = []
    for path in paths:
        key = _replay_key(path)
        keyed_paths.append((key, path))
    keys = [key for key, _path in keyed_paths]
    if len(set(keys)) != len(keys):
        raise ValueError("Replay content fingerprint paths contain duplicate shard keys.")
    if keys != sorted(keys):
        raise ValueError("Replay content fingerprint paths must be ordered by shard key.")
    return _replay_content_sha256(
        keys=keys,
        file_sha256={key: sha256_file(path) for key, path in keyed_paths},
    )


def named_files_content_sha256(
    *,
    repo_root: Path,
    relative_paths: tuple[str, ...] | list[str],
) -> str:
    """Fingerprint an explicitly ordered set of repository files."""
    if not relative_paths:
        raise ValueError("Named-file fingerprint requires at least one path.")
    normalized = [Path(raw).as_posix() for raw in relative_paths]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Named-file fingerprint paths contain duplicates.")
    if normalized != sorted(normalized):
        raise ValueError("Named-file fingerprint paths must be sorted.")
    payload = []
    for relative_path in normalized:
        path = (repo_root / relative_path).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Named-file fingerprint path escapes repo root: {relative_path}"
            ) from exc
        if not path.exists() or not path.is_file():
            raise ValueError(f"Named-file fingerprint path not found: {path}")
        payload.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
            }
        )
    return _canonical_sha256(payload)


def write_replay_path_lists(
    *,
    split: FrozenReplaySplit,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "trainingValue": output_dir / "training.value.paths.txt",
        "trainingOpponent": output_dir / "training.opponent.paths.txt",
        "validationValue": output_dir / "validation.value.paths.txt",
        "validationOpponent": output_dir / "validation.opponent.paths.txt",
    }
    path_sets = {
        "trainingValue": [split.value_paths[key] for key in split.training_keys],
        "trainingOpponent": [split.opponent_paths[key] for key in split.training_keys],
        "validationValue": [split.value_paths[key] for key in split.validation_keys],
        "validationOpponent": [split.opponent_paths[key] for key in split.validation_keys],
    }
    for name, output_path in outputs.items():
        payload = "".join(f"{path}\n" for path in path_sets[name])
        output_path.write_text(payload, encoding="utf-8")
    return {name: path.resolve() for name, path in outputs.items()}


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _replay_content_sha256(
    *,
    keys: tuple[str, ...] | list[str],
    file_sha256: dict[str, str],
) -> str:
    return _canonical_sha256([{"key": key, "sha256": file_sha256[key]} for key in keys])


def _replay_key(path: Path) -> str:
    for suffix in (".value.jsonl", ".opponent.jsonl"):
        if path.name.endswith(suffix):
            return path.name.removesuffix(suffix)
    raise ValueError(
        f"Replay content fingerprint paths must end in .value.jsonl or .opponent.jsonl: {path}"
    )
