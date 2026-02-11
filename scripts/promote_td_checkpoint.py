from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.checkpoint_manifest import (
    DEFAULT_MANIFEST_RELATIVE_PATH,
    normalized_checkpoint_key,
    update_manifest_promoted_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a promoted TD value/opponent checkpoint pair into models/td_checkpoints "
            "and register it in the checkpoint manifest."
        )
    )
    parser.add_argument("--key", required=True, help="Manifest key for this checkpoint pair.")
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-chunk", default=None)
    parser.add_argument("--source-loop-summary", type=Path, default=None)
    parser.add_argument("--source-eval-artifact", type=Path, action="append", default=[])
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_RELATIVE_PATH)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("models/td_checkpoints"),
        help="Directory containing manifest checkpoint subdirectories.",
    )
    parser.add_argument("--set-default", action="store_true")
    parser.add_argument("--add-to-opponent-pool", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_args(args)
    result = promote_checkpoint_pair(
        manifest_path=args.manifest_path,
        checkpoint_root=args.checkpoint_root,
        key=args.key,
        value_checkpoint=args.value_checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        source_run_id=args.source_run_id,
        source_loop_summary=args.source_loop_summary,
        source_chunk=args.source_chunk,
        source_eval_artifacts=args.source_eval_artifact,
        step=args.step,
        label=args.label,
        set_default=bool(args.set_default),
        add_to_opponent_pool=bool(args.add_to_opponent_pool),
        force=bool(args.force),
    )

    print(
        json.dumps(
            {
                "manifest": str(args.manifest_path),
                "key": result["key"],
                "value": str(result["value"]),
                "opponent": str(result["opponent"]),
                "defaultWarmStart": result["manifestPayload"].get("defaultWarmStart"),
                "opponentPool": result["manifestPayload"].get("opponentPool"),
            },
            indent=2,
        )
    )
    return 0


def promote_checkpoint_pair(
    *,
    manifest_path: Path,
    checkpoint_root: Path,
    key: str,
    value_checkpoint: Path,
    opponent_checkpoint: Path,
    source_run_id: str,
    source_loop_summary: Path | None = None,
    source_chunk: str | None = None,
    source_eval_artifacts: list[Path] | None = None,
    step: int | None = None,
    label: str | None = None,
    set_default: bool = False,
    add_to_opponent_pool: bool = False,
    force: bool = False,
) -> dict[str, object]:
    normalized_key = normalized_checkpoint_key(key)
    destination_dir = checkpoint_root / normalized_key
    if destination_dir.exists() and any(destination_dir.iterdir()) and not force:
        raise SystemExit(
            f"Destination checkpoint directory already exists: {destination_dir}. "
            "Pass --force to overwrite files for this key."
        )
    destination_dir.mkdir(parents=True, exist_ok=True)

    value_dest = destination_dir / value_checkpoint.name
    opponent_dest = destination_dir / opponent_checkpoint.name
    _copy_checkpoint(value_checkpoint, value_dest, force=force)
    _copy_checkpoint(opponent_checkpoint, opponent_dest, force=force)

    payload = update_manifest_promoted_checkpoint(
        manifest_path=manifest_path,
        key=normalized_key,
        value_path=value_dest,
        opponent_path=opponent_dest,
        source_run_id=source_run_id,
        source_loop_summary=source_loop_summary,
        source_chunk=source_chunk,
        source_eval_artifacts=source_eval_artifacts or [],
        step=step,
        label=label,
        set_default=set_default,
        add_to_opponent_pool=add_to_opponent_pool,
        force=force,
    )
    return {
        "key": normalized_key,
        "value": value_dest,
        "opponent": opponent_dest,
        "manifestPayload": payload,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not args.key.strip():
        raise SystemExit("--key must not be empty.")
    if not args.value_checkpoint.exists():
        raise SystemExit(f"--value-checkpoint does not exist: {args.value_checkpoint}")
    if not args.opponent_checkpoint.exists():
        raise SystemExit(f"--opponent-checkpoint does not exist: {args.opponent_checkpoint}")
    if args.step is not None and args.step <= 0:
        raise SystemExit("--step must be > 0 when provided.")
    if args.source_loop_summary is not None and not args.source_loop_summary.exists():
        raise SystemExit(f"--source-loop-summary does not exist: {args.source_loop_summary}")
    for artifact in args.source_eval_artifact:
        if not artifact.exists():
            raise SystemExit(f"--source-eval-artifact does not exist: {artifact}")


def _copy_checkpoint(source: Path, destination: Path, *, force: bool) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists() and not force:
        raise SystemExit(f"Destination checkpoint already exists: {destination}")
    shutil.copy2(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
