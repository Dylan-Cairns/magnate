from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_PROFILE_SPECS = (
    "current:6:14",
    "balanced:4:12",
    "lighter:4:10",
    "aggressive:3:8",
)


@dataclass(frozen=True)
class Checkpoints:
    source_run_id: str
    value: Path
    opponent: Path


@dataclass(frozen=True)
class SearchProfile:
    label: str
    worlds: int
    depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark td-search self-play collection throughput across a small "
            "search-worlds/search-depth profile matrix."
        )
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--td-loops-dir", type=Path, default=Path("artifacts/td_loops"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/benchmarks/collect_search_profiles"),
    )
    parser.add_argument("--value-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Repeated profile spec in label:worlds:depth format. "
            "Defaults to current:6:14, balanced:4:12, lighter:4:10, aggressive:3:8."
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--progress-every-games", type=int, default=1)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument(
        "--estimate-selfplay-games",
        type=int,
        default=510,
        help="Optional self-play-profile game count used for wall-time extrapolation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_supported_runtime(args.python_bin)
    _validate_args(args)

    profiles = _parse_profiles(args.profile or list(DEFAULT_PROFILE_SPECS))
    checkpoints = _resolve_checkpoints(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = args.artifact_dir / f"{stamp}-collect-search-profiles"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print(
        "[bench] thread caps: "
        f"OMP={env['OMP_NUM_THREADS']} MKL={env['MKL_NUM_THREADS']} "
        f"OPENBLAS={env['OPENBLAS_NUM_THREADS']} NUMEXPR={env['NUMEXPR_NUM_THREADS']}"
    )
    print(f"[bench] checkpoints: {checkpoints.source_run_id}")
    print(f"[bench] output dir: {run_dir}")
    print(f"[bench] workers={args.workers} games={args.games}")

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    for profile in profiles:
        row = _run_profile(
            args=args,
            env=env,
            checkpoints=checkpoints,
            profile=profile,
            run_dir=run_dir,
        )
        rows.append(row)
        if best_row is None or float(row["gamesPerMinute"]) > float(best_row["gamesPerMinute"]):
            best_row = row

    summary = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runDir": str(run_dir),
        "checkpoints": {
            "sourceRunId": checkpoints.source_run_id,
            "value": str(checkpoints.value),
            "opponent": str(checkpoints.opponent),
        },
        "env": {
            "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
            "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
            "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
            "NUMEXPR_NUM_THREADS": env["NUMEXPR_NUM_THREADS"],
            "PYTHONUNBUFFERED": env["PYTHONUNBUFFERED"],
        },
        "config": {
            "workers": args.workers,
            "games": args.games,
            "searchRollouts": args.search_rollouts,
            "searchMaxRootActions": args.search_max_root_actions,
            "searchRolloutEpsilon": args.search_rollout_epsilon,
            "estimateSelfplayGames": args.estimate_selfplay_games,
        },
        "profiles": rows,
        "bestProfile": best_row,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[bench] -------- profile summary --------")
    for row in rows:
        print(
            "[bench] "
            f"{row['label']}: "
            f"worlds={row['worlds']} depth={row['depth']} "
            f"elapsed={row['elapsedSeconds']:.1f}s "
            f"games/min={row['gamesPerMinute']:.3f} "
            f"estSelfplayMin={row['estimatedSelfplayMinutes']:.1f}"
        )
    if best_row is not None:
        print(
            "[bench] best profile: "
            f"{best_row['label']} "
            f"(worlds={best_row['worlds']} depth={best_row['depth']})"
        )
    print(f"[bench] summary: {summary_path}")
    print("[bench] --------------------------------")
    return 0


def _run_profile(
    *,
    args: argparse.Namespace,
    env: Dict[str, str],
    checkpoints: Checkpoints,
    profile: SearchProfile,
    run_dir: Path,
) -> Dict[str, Any]:
    worker_count = min(args.workers, args.games)
    shard_games = _split_count(args.games, worker_count)
    profile_dir = run_dir / profile.label
    logs_dir = profile_dir / "logs"
    profile_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    shard_jobs: List[tuple[int, List[str], Path, Path]] = []
    for index, games in enumerate(shard_games):
        shard_id = f"s{index + 1:02d}"
        shard_dir = profile_dir / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{shard_id}.log"
        summary_path = shard_dir / "summary.json"
        shard_jobs.append(
            (
                index + 1,
                _collect_command(
                    args=args,
                    games=games,
                    python_bin=args.python_bin,
                    profile=profile,
                    run_label=f"{profile.label}-{shard_id}",
                    seed_prefix=f"bench-{profile.label}-{shard_id}",
                    summary_out=summary_path,
                    value_out=shard_dir / "value.jsonl",
                    opponent_out=shard_dir / "opponent.jsonl",
                    checkpoints=checkpoints,
                ),
                log_path,
                summary_path,
            )
        )

    print(
        "[bench] profile="
        f"{profile.label} worlds={profile.worlds} depth={profile.depth} "
        f"workers={worker_count} shardGames={','.join(str(value) for value in shard_games)}"
    )

    started = time.perf_counter()
    processes: List[tuple[int, subprocess.Popen[Any], Path, Path, Any]] = []
    try:
        for shard_index, command, log_path, summary_path in shard_jobs:
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )
            processes.append((shard_index, process, log_path, summary_path, handle))

        first_failure: tuple[int, int] | None = None
        for shard_index, process, _log_path, _summary_path, _handle in processes:
            return_code = process.wait()
            if return_code != 0 and first_failure is None:
                first_failure = (shard_index, return_code)
                for _, other, _, _, _ in processes:
                    if other.poll() is None:
                        other.terminate()

        elapsed = time.perf_counter() - started
        if first_failure is not None:
            shard_index, code = first_failure
            raise SystemExit(
                "collect profile shard failed: "
                f"profile={profile.label} shard={shard_index} returnCode={code}"
            )
    finally:
        for _shard_index, _process, _log_path, _summary_path, handle in processes:
            handle.close()

    total_games = 0
    total_turn_weighted = 0.0
    total_value_transitions = 0
    for _, _, _, summary_path, _ in processes:
        payload = _read_collect_summary(summary_path)
        shard_games = int(payload["games"])
        total_games += shard_games
        total_turn_weighted += float(payload["averageTurn"]) * float(shard_games)
        total_value_transitions += int(payload["valueTransitions"])

    games_per_minute = (total_games / elapsed) * 60.0 if elapsed > 0 else 0.0
    estimated_selfplay_minutes = (
        float(args.estimate_selfplay_games) / games_per_minute
        if args.estimate_selfplay_games > 0 and games_per_minute > 0
        else None
    )
    return {
        "label": profile.label,
        "worlds": profile.worlds,
        "depth": profile.depth,
        "workers": worker_count,
        "games": total_games,
        "elapsedSeconds": elapsed,
        "gamesPerMinute": games_per_minute,
        "averageTurn": (total_turn_weighted / float(total_games)) if total_games > 0 else 0.0,
        "valueTransitions": total_value_transitions,
        "estimatedSelfplayMinutes": estimated_selfplay_minutes,
        "logs": [str(log_path) for _, _, log_path, _, _ in processes],
        "summaries": [str(summary_path) for _, _, _, summary_path, _ in processes],
    }


def _collect_command(
    *,
    args: argparse.Namespace,
    games: int,
    python_bin: Path,
    profile: SearchProfile,
    run_label: str,
    seed_prefix: str,
    summary_out: Path,
    value_out: Path,
    opponent_out: Path,
    checkpoints: Checkpoints,
) -> List[str]:
    return [
        str(python_bin),
        "-m",
        "scripts.collect_td_self_play",
        "--games",
        str(games),
        "--seed-prefix",
        seed_prefix,
        "--player-a-policy",
        "td-search",
        "--player-b-policy",
        "td-search",
        "--search-worlds",
        str(profile.worlds),
        "--search-rollouts",
        str(args.search_rollouts),
        "--search-depth",
        str(profile.depth),
        "--search-max-root-actions",
        str(args.search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.search_rollout_epsilon),
        "--run-label",
        run_label,
        "--value-out",
        str(value_out),
        "--opponent-out",
        str(opponent_out),
        "--summary-out",
        str(summary_out),
        "--progress-every-games",
        str(args.progress_every_games),
        "--player-a-td-search-value-checkpoint",
        str(checkpoints.value),
        "--player-a-td-search-opponent-checkpoint",
        str(checkpoints.opponent),
        "--player-b-td-search-value-checkpoint",
        str(checkpoints.value),
        "--player-b-td-search-opponent-checkpoint",
        str(checkpoints.opponent),
    ]


def _read_collect_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing collect summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"invalid collect summary results payload: {path}")
    return {
        "games": int(results["games"]),
        "averageTurn": float(results["averageTurn"]),
        "valueTransitions": int(results["valueTransitions"]),
    }


def _resolve_checkpoints(args: argparse.Namespace) -> Checkpoints:
    if (args.value_checkpoint is None) != (args.opponent_checkpoint is None):
        raise SystemExit(
            "provide both --value-checkpoint and --opponent-checkpoint, or neither"
        )
    if args.value_checkpoint is not None and args.opponent_checkpoint is not None:
        if not args.value_checkpoint.exists():
            raise SystemExit(f"--value-checkpoint does not exist: {args.value_checkpoint}")
        if not args.opponent_checkpoint.exists():
            raise SystemExit(
                f"--opponent-checkpoint does not exist: {args.opponent_checkpoint}"
            )
        return Checkpoints(
            source_run_id="explicit",
            value=args.value_checkpoint,
            opponent=args.opponent_checkpoint,
        )

    latest: Checkpoints | None = None
    if not args.td_loops_dir.exists():
        raise SystemExit(
            f"--td-loops-dir does not exist and no explicit checkpoints provided: {args.td_loops_dir}"
        )

    for summary_path in sorted(args.td_loops_dir.glob("*/loop.summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        promotion = payload.get("promotion") or {}
        if not bool(promotion.get("promoted")):
            continue
        chunks = payload.get("chunks") or []
        if not chunks:
            continue
        checkpoint = (chunks[-1] or {}).get("latestCheckpoint") or {}
        value_raw = checkpoint.get("value")
        opponent_raw = checkpoint.get("opponent")
        if not isinstance(value_raw, str) or not isinstance(opponent_raw, str):
            continue
        value_path = Path(value_raw)
        opponent_path = Path(opponent_raw)
        if not value_path.exists() or not opponent_path.exists():
            continue
        latest = Checkpoints(
            source_run_id=summary_path.parent.name,
            value=value_path,
            opponent=opponent_path,
        )

    if latest is None:
        raise SystemExit(
            "no promoted checkpoints found in --td-loops-dir; provide explicit checkpoint paths"
        )
    return latest


def _parse_profiles(raw_profiles: Sequence[str]) -> List[SearchProfile]:
    profiles: List[SearchProfile] = []
    for raw in raw_profiles:
        parts = [part.strip() for part in raw.split(":")]
        if len(parts) != 3:
            raise SystemExit(
                f"invalid --profile spec {raw!r}; expected label:worlds:depth"
            )
        label, worlds_raw, depth_raw = parts
        if not label:
            raise SystemExit(f"invalid --profile spec {raw!r}; label must be non-empty")
        try:
            worlds = int(worlds_raw)
            depth = int(depth_raw)
        except ValueError as exc:
            raise SystemExit(
                f"invalid --profile spec {raw!r}; worlds/depth must be integers"
            ) from exc
        if worlds <= 0:
            raise SystemExit(f"profile {label!r} worlds must be > 0")
        if depth <= 0:
            raise SystemExit(f"profile {label!r} depth must be > 0")
        profiles.append(SearchProfile(label=label, worlds=worlds, depth=depth))
    return profiles


def _split_count(total: int, workers: int) -> List[int]:
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _require_supported_runtime(python_bin: Path) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("run from the project virtual environment (.venv).")
    if not python_bin.exists():
        raise SystemExit(f"--python-bin does not exist: {python_bin}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.games <= 0:
        raise SystemExit("--games must be > 0.")
    if args.progress_every_games <= 0:
        raise SystemExit("--progress-every-games must be > 0.")
    if args.search_rollouts <= 0:
        raise SystemExit("--search-rollouts must be > 0.")
    if args.search_max_root_actions <= 0:
        raise SystemExit("--search-max-root-actions must be > 0.")
    if args.search_rollout_epsilon < 0.0 or args.search_rollout_epsilon > 1.0:
        raise SystemExit("--search-rollout-epsilon must be in [0, 1].")


if __name__ == "__main__":
    raise SystemExit(main())
