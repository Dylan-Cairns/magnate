from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.td_loop_common import concat_jsonl_files, join_command, read_json, run_step

ALL_RUN_ROOTS = (
    "artifacts/imported/usb-20260706/td_replay/20260619T134512Z-v2-40w180d-teacher-1400",
    "artifacts/imported/usb-20260706/artifacts/td_replay/20260619T003038Z-v2-40w180d-teacher-1000",
    "artifacts/imported/usb-20260706/artifacts/td_replay/20260618T220338Z-v2-40w180d-benchmark-4",
    "artifacts/imported/usb-20260706/artifacts/td_replay/20260618T011656Z-v2-hard-teacher-100",
    "artifacts/td_replay/20260618T231515Z-v2-hard-laptop-700",
    "artifacts/td_replay/20260619T135628Z-v2-hard-laptop-900",
)
HARD_RUN_ROOTS = (
    "artifacts/imported/usb-20260706/artifacts/td_replay/20260618T011656Z-v2-hard-teacher-100",
    "artifacts/td_replay/20260618T231515Z-v2-hard-laptop-700",
    "artifacts/td_replay/20260619T135628Z-v2-hard-laptop-900",
)


@dataclass(frozen=True)
class ReplayRunInventory:
    run_root: Path
    shards: int
    games: int
    value_rows: int
    opponent_rows: int


@dataclass(frozen=True)
class ReplaySetInventory:
    set_name: str
    runs: Sequence[ReplayRunInventory]
    value_paths: Sequence[Path]
    opponent_paths: Sequence[Path]
    shards: int
    games: int
    value_rows: int
    opponent_rows: int


@dataclass(frozen=True)
class RuntimeProfile:
    logical_processors: int
    cpu_target_percent: float
    reserve_logical_cores: int
    train_threads: int
    train_interop_threads: int


@dataclass(frozen=True)
class WarmStartPair:
    source_label: str
    run_id: str
    value_path: Path
    opponent_path: Path
    generated_at: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--stage1-steps", type=int, default=60000)
    parser.add_argument("--stage2-steps", type=int, default=30000)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--progress-every-steps", type=int, default=100)
    parser.add_argument("--cpu-target-percent", type=float, default=80.0)
    parser.add_argument("--reserve-logical-cores", type=int, default=1)
    parser.add_argument("--heartbeat-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    python_bin = args.python_bin.resolve()
    _validate_args(args)
    _configure_local_caches(repo_root)

    runtime = _runtime_profile(
        cpu_target_percent=args.cpu_target_percent,
        reserve_logical_cores=args.reserve_logical_cores,
    )
    warm_start = _warm_start_pair(repo_root)
    if warm_start is None:
        raise SystemExit(
            "No warm-start checkpoint pair found. Restore "
            "models/td_checkpoints/manifest.json plus promoted checkpoints first."
        )

    input_root = repo_root / "artifacts/training_inputs/two_stage_imported_20260706"
    checkpoint_root = repo_root / "artifacts/td_checkpoints/two_stage_imported_20260706"
    progress_path = checkpoint_root / "two_stage.progress.json"
    all_value_path = input_root / "all.value.jsonl"
    all_opponent_path = input_root / "all.opponent.jsonl"
    hard_value_path = input_root / "hard-only.value.jsonl"
    hard_opponent_path = input_root / "hard-only.opponent.jsonl"
    manifest_path = input_root / "manifest.json"

    stage1_value_summary = checkpoint_root / "stage1-all-value.summary.json"
    stage1_opponent_summary = checkpoint_root / "stage1-all-opponent.summary.json"
    stage2_value_summary = checkpoint_root / "stage2-hard-only-value.summary.json"
    stage2_opponent_summary = checkpoint_root / "stage2-hard-only-opponent.summary.json"

    all_inventory = _replay_set_inventory(
        repo_root=repo_root,
        run_roots=ALL_RUN_ROOTS,
        set_name="all",
    )
    hard_inventory = _replay_set_inventory(
        repo_root=repo_root,
        run_roots=HARD_RUN_ROOTS,
        set_name="hard-only",
    )

    print(f"[two-stage] python={python_bin}", flush=True)
    print(f"[two-stage] warmStartSource={warm_start.source_label}", flush=True)
    print(f"[two-stage] warmStartRunId={warm_start.run_id}", flush=True)
    print(
        "[two-stage] threadCaps="
        f"OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS')} "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}",
        flush=True,
    )
    print(
        "[two-stage] cpuProfile="
        f"logical={runtime.logical_processors} "
        f"targetPercent={runtime.cpu_target_percent:.1f} "
        f"reserveLogicalCores={runtime.reserve_logical_cores} "
        f"trainThreads={runtime.train_threads} "
        f"trainInteropThreads={runtime.train_interop_threads}",
        flush=True,
    )
    print(
        f"[two-stage] allData games={all_inventory.games} "
        f"shards={all_inventory.shards} rows={all_inventory.value_rows}",
        flush=True,
    )
    print(
        f"[two-stage] hardOnly games={hard_inventory.games} "
        f"shards={hard_inventory.shards} rows={hard_inventory.value_rows}",
        flush=True,
    )
    print(f"[two-stage] inputRoot={input_root}", flush=True)
    print(f"[two-stage] checkpointRoot={checkpoint_root}", flush=True)

    if args.dry_run:
        print("[two-stage] dry run: replay concatenation skipped.", flush=True)
    else:
        _prepare_replay_file(
            inputs=all_inventory.value_paths,
            output=all_value_path,
            force=args.force_prepare,
        )
        _prepare_replay_file(
            inputs=all_inventory.opponent_paths,
            output=all_opponent_path,
            force=args.force_prepare,
        )
        _prepare_replay_file(
            inputs=hard_inventory.value_paths,
            output=hard_value_path,
            force=args.force_prepare,
        )
        _prepare_replay_file(
            inputs=hard_inventory.opponent_paths,
            output=hard_opponent_path,
            force=args.force_prepare,
        )
        _write_manifest(
            manifest_path=manifest_path,
            all_inventory=all_inventory,
            hard_inventory=hard_inventory,
            all_value_path=all_value_path,
            all_opponent_path=all_opponent_path,
            hard_value_path=hard_value_path,
            hard_opponent_path=hard_opponent_path,
        )

    common_train_args = [
        "--value-target-mode",
        "td-lambda",
        "--td-lambda",
        "0.7",
        "--save-every-steps",
        str(args.save_every_steps),
        "--progress-every-steps",
        str(args.progress_every_steps),
        "--num-threads",
        str(runtime.train_threads),
        "--num-interop-threads",
        str(runtime.train_interop_threads),
        "--out-dir",
        str(checkpoint_root),
    ]

    stage1_value_command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--run-label",
        "td-two-stage-20260706-stage1-all-value",
        "--steps",
        str(args.stage1_steps),
        "--seed",
        "20260706",
        "--value-replay",
        str(all_value_path),
        "--warm-start-value-checkpoint",
        str(warm_start.value_path),
        "--disable-opponent",
        "--summary-out",
        str(stage1_value_summary),
        *common_train_args,
    ]
    _run_or_print(
        name="stage1-all-value",
        command=stage1_value_command,
        dry_run=args.dry_run,
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_path,
    )

    stage1_opponent_command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--run-label",
        "td-two-stage-20260706-stage1-all-opponent",
        "--steps",
        str(args.stage1_steps),
        "--seed",
        "20260706",
        "--opponent-replay",
        str(all_opponent_path),
        "--warm-start-opponent-checkpoint",
        str(warm_start.opponent_path),
        "--disable-value",
        "--summary-out",
        str(stage1_opponent_summary),
        *common_train_args,
    ]
    _run_or_print(
        name="stage1-all-opponent",
        command=stage1_opponent_command,
        dry_run=args.dry_run,
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_path,
    )

    if args.dry_run:
        print(
            "[two-stage] dry run: stage 2 would warm-start from stage 1 summaries "
            f"under {checkpoint_root}.",
            flush=True,
        )
        return 0

    stage1_value_checkpoint = _final_checkpoint_path(stage1_value_summary, "value")
    stage1_opponent_checkpoint = _final_checkpoint_path(stage1_opponent_summary, "opponent")
    print(f"[two-stage] stage1Value={stage1_value_checkpoint}", flush=True)
    print(f"[two-stage] stage1Opponent={stage1_opponent_checkpoint}", flush=True)

    stage2_value_command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--run-label",
        "td-two-stage-20260706-stage2-hard-only-value",
        "--steps",
        str(args.stage2_steps),
        "--seed",
        "20260707",
        "--value-replay",
        str(hard_value_path),
        "--warm-start-value-checkpoint",
        str(stage1_value_checkpoint),
        "--disable-opponent",
        "--summary-out",
        str(stage2_value_summary),
        *common_train_args,
    ]
    _run_or_print(
        name="stage2-hard-only-value",
        command=stage2_value_command,
        dry_run=False,
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_path,
    )

    stage2_opponent_command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--run-label",
        "td-two-stage-20260706-stage2-hard-only-opponent",
        "--steps",
        str(args.stage2_steps),
        "--seed",
        "20260707",
        "--opponent-replay",
        str(hard_opponent_path),
        "--warm-start-opponent-checkpoint",
        str(stage1_opponent_checkpoint),
        "--disable-value",
        "--summary-out",
        str(stage2_opponent_summary),
        *common_train_args,
    ]
    _run_or_print(
        name="stage2-hard-only-opponent",
        command=stage2_opponent_command,
        dry_run=False,
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_path,
    )

    stage2_value_checkpoint = _final_checkpoint_path(stage2_value_summary, "value")
    stage2_opponent_checkpoint = _final_checkpoint_path(stage2_opponent_summary, "opponent")
    print("[two-stage] completed", flush=True)
    print(f"[two-stage] finalValue={stage2_value_checkpoint}", flush=True)
    print(f"[two-stage] finalOpponent={stage2_opponent_checkpoint}", flush=True)
    print(f"[two-stage] stage1ValueSummary={stage1_value_summary}", flush=True)
    print(f"[two-stage] stage1OpponentSummary={stage1_opponent_summary}", flush=True)
    print(f"[two-stage] stage2ValueSummary={stage2_value_summary}", flush=True)
    print(f"[two-stage] stage2OpponentSummary={stage2_opponent_summary}", flush=True)
    print(f"[two-stage] progressPath={progress_path}", flush=True)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("stage1_steps", "stage2_steps", "save_every_steps", "progress_every_steps"):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0.")
    if args.cpu_target_percent < 25.0 or args.cpu_target_percent > 100.0:
        raise SystemExit("--cpu-target-percent must be in [25, 100].")
    if args.reserve_logical_cores < 0:
        raise SystemExit("--reserve-logical-cores must be >= 0.")
    if args.heartbeat_minutes < 0.0:
        raise SystemExit("--heartbeat-minutes must be >= 0.")


def _configure_local_caches(repo_root: Path) -> None:
    temp_dir = repo_root / ".tmp"
    pip_cache_dir = repo_root / ".pip-cache"
    npm_cache_dir = repo_root / ".npm-cache"
    yarn_cache_dir = repo_root / ".yarn-cache"
    for path in (temp_dir, pip_cache_dir, npm_cache_dir, yarn_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache_dir)
    os.environ["NPM_CONFIG_CACHE"] = str(npm_cache_dir)
    os.environ["YARN_CACHE_FOLDER"] = str(yarn_cache_dir)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    print(f"[two-stage] tempDir={temp_dir}", flush=True)


def _runtime_profile(*, cpu_target_percent: float, reserve_logical_cores: int) -> RuntimeProfile:
    logical_processors = max(1, os.cpu_count() or 1)
    max_reservable = max(0, logical_processors - 1)
    reserved = min(reserve_logical_cores, max_reservable)
    available_budget = max(1, logical_processors - reserved)
    target_slots = max(1, round((logical_processors * cpu_target_percent) / 100.0))
    train_threads = min(available_budget, target_slots)
    return RuntimeProfile(
        logical_processors=logical_processors,
        cpu_target_percent=round(cpu_target_percent, 1),
        reserve_logical_cores=reserved,
        train_threads=train_threads,
        train_interop_threads=1,
    )


def _warm_start_pair(repo_root: Path) -> WarmStartPair | None:
    manifest_path = repo_root / "models/td_checkpoints/manifest.json"
    if manifest_path.exists():
        payload = read_json(manifest_path, label="checkpoint manifest")
        default_key = str(payload.get("defaultWarmStart") or "promoted")
        checkpoints = payload.get("checkpoints")
        if isinstance(checkpoints, dict):
            entry = checkpoints.get(default_key)
            resolved_key = default_key
            if not isinstance(entry, dict):
                entry = checkpoints.get("promoted")
                resolved_key = "promoted"
            if isinstance(entry, dict):
                status = str(entry.get("status") or ("promoted" if resolved_key == "promoted" else "experimental"))
                if status == "promoted":
                    value_path = _resolve_project_path(repo_root, entry.get("value"))
                    opponent_path = _resolve_project_path(repo_root, entry.get("opponent"))
                    if value_path is not None and opponent_path is not None:
                        if value_path.exists() and opponent_path.exists():
                            return WarmStartPair(
                                source_label=f"checkpoint manifest ({resolved_key})",
                                run_id=str(entry.get("sourceRunId") or ""),
                                value_path=value_path,
                                opponent_path=opponent_path,
                                generated_at=datetime.min.replace(tzinfo=timezone.utc),
                            )

    artifact_root = repo_root / "artifacts/td_loops"
    latest: WarmStartPair | None = None
    if artifact_root.exists():
        for summary_path in artifact_root.rglob("loop.summary.json"):
            try:
                payload = read_json(summary_path, label=f"loop summary {summary_path}")
            except SystemExit:
                continue
            promotion = payload.get("promotion")
            if not isinstance(promotion, dict) or not bool(promotion.get("promoted")):
                continue
            chunks = payload.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                continue
            latest_checkpoint = chunks[-1].get("latestCheckpoint")
            if not isinstance(latest_checkpoint, dict):
                continue
            value_path = _resolve_project_path(repo_root, latest_checkpoint.get("value"))
            opponent_path = _resolve_project_path(repo_root, latest_checkpoint.get("opponent"))
            if value_path is None or opponent_path is None:
                continue
            if not value_path.exists() or not opponent_path.exists():
                continue
            generated_at = _parse_datetime(str(payload.get("generatedAtUtc") or ""))
            candidate = WarmStartPair(
                source_label="latest promoted loop summary",
                run_id=str(payload.get("runId") or summary_path.parent.name),
                value_path=value_path,
                opponent_path=opponent_path,
                generated_at=generated_at,
            )
            if latest is None or candidate.generated_at > latest.generated_at:
                latest = candidate
    return latest


def _resolve_project_path(repo_root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _parse_datetime(raw: str) -> datetime:
    if not raw:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        normalized = raw.replace("Z", "+00:00")
        value = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _replay_set_inventory(
    *,
    repo_root: Path,
    run_roots: Sequence[str],
    set_name: str,
) -> ReplaySetInventory:
    runs: list[ReplayRunInventory] = []
    value_paths: list[Path] = []
    opponent_paths: list[Path] = []
    total_games = 0
    total_value_rows = 0
    total_opponent_rows = 0

    for run_root_raw in run_roots:
        run_root = (repo_root / run_root_raw).resolve()
        if not run_root.exists():
            raise SystemExit(f"Missing {set_name} replay run: {run_root}")
        shards_dir = run_root / "shards"
        if not shards_dir.exists():
            raise SystemExit(f"Missing {set_name} shards directory: {shards_dir}")

        run_value_paths = sorted(shards_dir.glob("*.value.jsonl"))
        run_opponent_paths = sorted(shards_dir.glob("*.opponent.jsonl"))
        run_summary_paths = sorted(shards_dir.glob("*.summary.json"))
        if not run_value_paths:
            raise SystemExit(f"No value replay shards found for {set_name} run: {run_root}")
        if len(run_value_paths) != len(run_opponent_paths):
            raise SystemExit(f"Value/opponent shard count mismatch for {set_name} run: {run_root}")
        if len(run_value_paths) != len(run_summary_paths):
            raise SystemExit(f"Value/summary shard count mismatch for {set_name} run: {run_root}")

        run_games = 0
        run_value_rows = 0
        run_opponent_rows = 0
        for summary_path in run_summary_paths:
            summary = read_json(summary_path, label=f"replay shard summary {summary_path}")
            results = summary.get("results")
            if not isinstance(results, dict):
                raise SystemExit(f"Replay shard summary missing results: {summary_path}")
            run_games += _json_int(results, "games", summary_path)
            run_value_rows += _json_int(results, "valueTransitions", summary_path)
            run_opponent_rows += _json_int(results, "opponentSamples", summary_path)

        if run_value_rows != run_opponent_rows:
            raise SystemExit(f"Value/opponent row count mismatch for {set_name} run: {run_root}")

        value_paths.extend(run_value_paths)
        opponent_paths.extend(run_opponent_paths)
        total_games += run_games
        total_value_rows += run_value_rows
        total_opponent_rows += run_opponent_rows
        runs.append(
            ReplayRunInventory(
                run_root=run_root,
                shards=len(run_value_paths),
                games=run_games,
                value_rows=run_value_rows,
                opponent_rows=run_opponent_rows,
            )
        )

    if len(value_paths) != len(opponent_paths):
        raise SystemExit(f"Value/opponent total shard count mismatch for {set_name}.")
    if total_value_rows != total_opponent_rows:
        raise SystemExit(f"Value/opponent total row count mismatch for {set_name}.")

    return ReplaySetInventory(
        set_name=set_name,
        runs=runs,
        value_paths=value_paths,
        opponent_paths=opponent_paths,
        shards=len(value_paths),
        games=total_games,
        value_rows=total_value_rows,
        opponent_rows=total_opponent_rows,
    )


def _json_int(payload: dict[str, Any], key: str, source_path: Path) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"Expected integer results.{key} in {source_path}: {value!r}")
    return value


def _prepare_replay_file(*, inputs: Sequence[Path], output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[two-stage] using existing replay file: {output}", flush=True)
        return
    temp_path = output.with_suffix(output.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    print(f"[two-stage] writing replay file: {output}", flush=True)
    concat_jsonl_files(inputs=inputs, output=temp_path)
    temp_path.replace(output)


def _write_manifest(
    *,
    manifest_path: Path,
    all_inventory: ReplaySetInventory,
    hard_inventory: ReplaySetInventory,
    all_value_path: Path,
    all_opponent_path: Path,
    hard_value_path: Path,
    hard_opponent_path: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "curriculum": "stage1-all-data-stage2-hard-only",
        "generatedReplayFiles": {
            "allValue": str(all_value_path),
            "allOpponent": str(all_opponent_path),
            "hardValue": str(hard_value_path),
            "hardOpponent": str(hard_opponent_path),
        },
        "allData": _inventory_manifest(all_inventory),
        "hardOnly": _inventory_manifest(hard_inventory),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _inventory_manifest(inventory: ReplaySetInventory) -> dict[str, Any]:
    return {
        "games": inventory.games,
        "shards": inventory.shards,
        "valueRows": inventory.value_rows,
        "opponentRows": inventory.opponent_rows,
        "runs": [
            {
                "runRoot": str(run.run_root),
                "shards": run.shards,
                "games": run.games,
                "valueRows": run.value_rows,
                "opponentRows": run.opponent_rows,
            }
            for run in inventory.runs
        ],
    }


def _run_or_print(
    *,
    name: str,
    command: Sequence[str],
    dry_run: bool,
    heartbeat_minutes: float,
    progress_path: Path,
) -> None:
    if dry_run:
        print(f"[two-stage] dry-run step {name}: {join_command(command)}", flush=True)
        return
    run_step(
        name=name,
        command=command,
        heartbeat_minutes=heartbeat_minutes,
        progress_path=progress_path,
        log_prefix="[two-stage]",
    )


def _final_checkpoint_path(summary_path: Path, kind: str) -> Path:
    summary = read_json(summary_path, label=f"{kind} training summary")
    results = summary.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Training summary missing results: {summary_path}")
    checkpoints = results.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise SystemExit(f"Training summary has no checkpoints: {summary_path}")
    last = checkpoints[-1]
    if not isinstance(last, dict):
        raise SystemExit(f"Invalid checkpoint entry in summary: {summary_path}")
    raw = last.get(kind)
    if not isinstance(raw, str) or not raw:
        raise SystemExit(f"Final {kind} checkpoint is missing in summary: {summary_path}")
    checkpoint_path = Path(raw).resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Final {kind} checkpoint not found: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    raise SystemExit(main())
