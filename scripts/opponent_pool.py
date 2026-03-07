from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class PoolCheckpoint:
    run_id: str
    generated_at_utc: str
    value_path: Path
    opponent_path: Path


def load_promoted_checkpoints(
    *,
    artifact_dir: Path,
    max_entries: int | None = None,
    require_paths: bool = True,
) -> List[PoolCheckpoint]:
    if not artifact_dir.exists():
        return []

    rows: List[Tuple[datetime, PoolCheckpoint]] = []
    for summary_path in sorted(artifact_dir.glob("*/loop.summary.json")):
        payload = _read_json_object(summary_path)
        promotion = payload.get("promotion")
        if not isinstance(promotion, dict) or not bool(promotion.get("promoted")):
            continue

        chunks = payload.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            continue
        latest_chunk = chunks[-1]
        if not isinstance(latest_chunk, dict):
            continue
        latest_checkpoint = latest_chunk.get("latestCheckpoint")
        if not isinstance(latest_checkpoint, dict):
            continue

        value_raw = latest_checkpoint.get("value")
        opponent_raw = latest_checkpoint.get("opponent")
        if not isinstance(value_raw, str) or not isinstance(opponent_raw, str):
            continue

        value_path = Path(value_raw)
        opponent_path = Path(opponent_raw)
        if require_paths and (not value_path.exists() or not opponent_path.exists()):
            continue

        run_id = str(payload.get("runId") or summary_path.parent.name)
        generated = str(payload.get("generatedAtUtc") or "")
        generated_dt = _parse_datetime_or_min(generated)

        rows.append(
            (
                generated_dt,
                PoolCheckpoint(
                    run_id=run_id,
                    generated_at_utc=generated,
                    value_path=value_path,
                    opponent_path=opponent_path,
                ),
            )
        )

    rows.sort(key=lambda item: item[0], reverse=True)
    checkpoints = [row[1] for row in rows]
    if max_entries is not None:
        checkpoints = checkpoints[:max_entries]
    return checkpoints


def filter_pool_excluding_checkpoint(
    *,
    checkpoints: Sequence[PoolCheckpoint],
    value_path: Path | None,
    opponent_path: Path | None,
) -> List[PoolCheckpoint]:
    if value_path is None or opponent_path is None:
        return list(checkpoints)
    value_abs = value_path.resolve()
    opponent_abs = opponent_path.resolve()
    return [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.value_path.resolve() != value_abs
        or checkpoint.opponent_path.resolve() != opponent_abs
    ]


def weighted_game_split(total_games: int, weights: Dict[str, float]) -> Dict[str, int]:
    if total_games <= 0:
        raise ValueError("total_games must be > 0.")
    if not weights:
        raise ValueError("weights must not be empty.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must be >= 0.")
    positive = {key: value for key, value in weights.items() if value > 0}
    if not positive:
        raise ValueError("at least one weight must be > 0.")

    total_weight = float(sum(positive.values()))
    raw_alloc: Dict[str, float] = {
        key: (float(total_games) * value / total_weight) for key, value in positive.items()
    }
    base_alloc: Dict[str, int] = {key: int(raw_alloc[key]) for key in positive}
    assigned = sum(base_alloc.values())
    remainder = total_games - assigned
    if remainder > 0:
        fractional_order = sorted(
            positive.keys(),
            key=lambda key: (raw_alloc[key] - float(base_alloc[key]), key),
            reverse=True,
        )
        for index in range(remainder):
            key = fractional_order[index % len(fractional_order)]
            base_alloc[key] += 1

    result = {key: 0 for key in weights}
    result.update(base_alloc)
    return result


def split_evenly(total_games: int, entries: Iterable[str]) -> Dict[str, int]:
    keys = list(entries)
    if total_games < 0:
        raise ValueError("total_games must be >= 0.")
    if not keys:
        return {}
    base = total_games // len(keys)
    remainder = total_games % len(keys)
    return {
        key: base + (1 if index < remainder else 0)
        for index, key in enumerate(keys)
    }


def _read_json_object(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _parse_datetime_or_min(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
