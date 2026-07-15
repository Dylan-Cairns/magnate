from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from trainer.td import (
    evaluate_opponent_holdout,
    evaluate_value_holdout,
    load_opponent_checkpoint,
    load_value_checkpoint,
    read_opponent_samples_jsonl_many,
    read_value_transitions_jsonl_many,
    replay_content_sha256,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen TD value and opponent checkpoints on a complete "
            "sequence-aware replay holdout."
        )
    )
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--value-replay-list", type=Path, required=True)
    parser.add_argument("--opponent-replay-list", type=Path, required=True)
    parser.add_argument("--expected-value-replay-content-sha256", required=True)
    parser.add_argument("--expected-opponent-replay-content-sha256", required=True)
    parser.add_argument("--expected-value-checkpoint-sha256", required=True)
    parser.add_argument("--expected-opponent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-step", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime()
    _validate_args(args)
    torch.set_num_threads(args.num_threads)

    value_paths = _read_path_list(args.value_replay_list, label="value")
    opponent_paths = _read_path_list(args.opponent_replay_list, label="opponent")
    value_replay_sha = replay_content_sha256(value_paths)
    opponent_replay_sha = replay_content_sha256(opponent_paths)
    _expect_sha256(
        label="value replay content",
        actual=value_replay_sha,
        expected=args.expected_value_replay_content_sha256,
    )
    _expect_sha256(
        label="opponent replay content",
        actual=opponent_replay_sha,
        expected=args.expected_opponent_replay_content_sha256,
    )

    value_checkpoint_sha = sha256_file(args.value_checkpoint)
    opponent_checkpoint_sha = sha256_file(args.opponent_checkpoint)
    _expect_sha256(
        label="value checkpoint",
        actual=value_checkpoint_sha,
        expected=args.expected_value_checkpoint_sha256,
    )
    _expect_sha256(
        label="opponent checkpoint",
        actual=opponent_checkpoint_sha,
        expected=args.expected_opponent_checkpoint_sha256,
    )
    value_model, value_payload = load_value_checkpoint(path=args.value_checkpoint)
    opponent_model, opponent_payload = load_opponent_checkpoint(path=args.opponent_checkpoint)
    _expect_checkpoint_step(
        label="value",
        payload=value_payload,
        expected=args.expected_checkpoint_step,
    )
    _expect_checkpoint_step(
        label="opponent",
        payload=opponent_payload,
        expected=args.expected_checkpoint_step,
    )

    transitions = read_value_transitions_jsonl_many(value_paths)
    samples = read_opponent_samples_jsonl_many(opponent_paths)
    result = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "valueCheckpoint": str(args.value_checkpoint.resolve()),
            "valueCheckpointSha256": value_checkpoint_sha,
            "opponentCheckpoint": str(args.opponent_checkpoint.resolve()),
            "opponentCheckpointSha256": opponent_checkpoint_sha,
            "checkpointStep": args.expected_checkpoint_step,
            "valueReplayList": str(args.value_replay_list.resolve()),
            "valueReplayContentSha256": value_replay_sha,
            "opponentReplayList": str(args.opponent_replay_list.resolve()),
            "opponentReplayContentSha256": opponent_replay_sha,
            "gamma": args.gamma,
            "pythonVersion": sys.version,
            "torchVersion": torch.__version__,
        },
        "value": evaluate_value_holdout(
            model=value_model,
            transitions=transitions,
            gamma=args.gamma,
            batch_size=args.value_batch_size,
        ),
        "opponent": evaluate_opponent_holdout(
            model=opponent_model,
            samples=samples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output)}, indent=2))
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("value checkpoint", args.value_checkpoint),
        ("opponent checkpoint", args.opponent_checkpoint),
        ("value replay list", args.value_replay_list),
        ("opponent replay list", args.opponent_replay_list),
    ):
        if not path.exists() or not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
    for flag, value in (
        (
            "--expected-value-replay-content-sha256",
            args.expected_value_replay_content_sha256,
        ),
        (
            "--expected-opponent-replay-content-sha256",
            args.expected_opponent_replay_content_sha256,
        ),
        ("--expected-value-checkpoint-sha256", args.expected_value_checkpoint_sha256),
        (
            "--expected-opponent-checkpoint-sha256",
            args.expected_opponent_checkpoint_sha256,
        ),
    ):
        if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise SystemExit(f"{flag} must be a 64-character hexadecimal SHA-256.")
    if args.expected_checkpoint_step <= 0:
        raise SystemExit("--expected-checkpoint-step must be > 0.")
    if args.gamma < 0.0 or args.gamma > 1.0:
        raise SystemExit("--gamma must be in [0, 1].")
    if args.value_batch_size <= 0:
        raise SystemExit("--value-batch-size must be > 0.")
    if args.num_threads <= 0:
        raise SystemExit("--num-threads must be > 0.")


def _read_path_list(path: Path, *, label: str) -> list[Path]:
    paths: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        replay_path = Path(entry)
        if not replay_path.is_absolute():
            replay_path = path.parent / replay_path
        replay_path = replay_path.resolve()
        if not replay_path.exists() or not replay_path.is_file():
            raise SystemExit(f"{label} replay path not found: {replay_path}")
        paths.append(replay_path)
    if not paths:
        raise SystemExit(f"{label} replay list is empty: {path}")
    if len(set(paths)) != len(paths):
        raise SystemExit(f"{label} replay list contains duplicates: {path}")
    return paths


def _expect_checkpoint_step(*, label: str, payload: dict[str, Any], expected: int) -> None:
    metadata = payload.get("metadata")
    actual = metadata.get("step") if isinstance(metadata, dict) else None
    if actual != expected:
        raise SystemExit(f"{label} checkpoint step mismatch: expected={expected} actual={actual}")


def _expect_sha256(*, label: str, actual: str, expected: str) -> None:
    if actual.lower() != expected.lower():
        raise SystemExit(
            f"{label} SHA-256 mismatch: expected={expected.lower()} actual={actual.lower()}"
        )


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.12+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


if __name__ == "__main__":
    raise SystemExit(main())
