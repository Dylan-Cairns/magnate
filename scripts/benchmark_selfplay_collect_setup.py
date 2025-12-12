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
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Checkpoints:
    source_run_id: str
    value: Path
    opponent: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark self-play collect throughput on this machine and recommend "
            "a safe --collect-workers setting."
        )
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--td-loops-dir", type=Path, default=Path("artifacts/td_loops"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/benchmarks/selfplay_collect_setup"),
    )
    parser.add_argument("--value-checkpoint", type=Path, default=None)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--single-games", type=int, default=40)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--speedup-threshold", type=float, default=1.2)
    parser.add_argument("--progress-every-games", type=int, default=10)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast apples-to-apples timing check (small game count + lighter search). "
            "Uses the same config for single and parallel phases."
        ),
    )

    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--td-worlds", type=int, default=8)
    parser.add_argument("--td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--td-search-sample-opponent-actions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _apply_smoke_profile(args)
    _require_supported_runtime(args.python_bin)
    _validate_args(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_dir = args.artifact_dir / f"{stamp}-selfplay-collect-setup-bench"
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _resolve_checkpoints(args)

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
    if args.smoke:
        print("[bench] mode=smoke")
    print(f"[bench] using checkpoints from run: {checkpoints.source_run_id}")
    print(f"[bench] output dir: {run_dir}")

    single_report = _run_single_phase(
        args=args,
        checkpoints=checkpoints,
        out_dir=out_dir,
        logs_dir=logs_dir,
        env=env,
    )
    parallel_report = _run_parallel_phase(
        args=args,
        checkpoints=checkpoints,
        out_dir=out_dir,
        logs_dir=logs_dir,
        env=env,
    )

    speedup = single_report["elapsedSeconds"] / parallel_report["elapsedSeconds"]
    recommendation = (
        args.parallel_workers if speedup >= args.speedup_threshold else 1
    )
    verdict = "pass" if speedup >= args.speedup_threshold else "fail"

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
            "singleGames": args.single_games,
            "parallelWorkers": args.parallel_workers,
            "parallelGamesTotal": args.single_games,
            "speedupThreshold": args.speedup_threshold,
        },
        "results": {
            "single": single_report,
            "parallel": parallel_report,
            "speedup": speedup,
            "verdict": verdict,
            "recommendedCollectWorkers": recommendation,
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[bench] -------- result --------")
    print(f"[bench] single elapsed:   {single_report['elapsedSeconds']:.1f}s")
    print(f"[bench] parallel elapsed: {parallel_report['elapsedSeconds']:.1f}s")
    print(f"[bench] speedup:          {speedup:.3f}x (threshold {args.speedup_threshold:.3f}x)")
    print(f"[bench] verdict:          {verdict}")
    print(f"[bench] recommended --collect-workers: {recommendation}")
    print(f"[bench] summary: {summary_path}")
    print("[bench] ------------------------")
    return 0


def _run_single_phase(
    *,
    args: argparse.Namespace,
    checkpoints: Checkpoints,
    out_dir: Path,
    logs_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    phase = "single"
    phase_out = out_dir / phase
    phase_out.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{phase}.log"

    cmd = _collect_command(
        args=args,
        python_bin=args.python_bin,
        games=args.single_games,
        seed_prefix=f"bench-selfplay-{phase}",
        run_label=f"bench-selfplay-{phase}",
        value_out=phase_out / "value.jsonl",
        opponent_out=phase_out / "opponent.jsonl",
        summary_out=phase_out / "summary.json",
        value_ckpt=checkpoints.value,
        opponent_ckpt=checkpoints.opponent,
    )
    print(f"[bench] phase={phase} games={args.single_games}")
    elapsed = _run_command_logged(cmd=cmd, log_path=log_path, env=env)
    summary = _read_collect_summary(phase_out / "summary.json")
    return {
        "games": summary["games"],
        "elapsedSeconds": elapsed,
        "gamesPerMinute": (summary["games"] / elapsed) * 60.0,
        "valueTransitions": summary["valueTransitions"],
        "avgTurn": summary["averageTurn"],
        "log": str(log_path),
        "summary": str(phase_out / "summary.json"),
    }


def _run_parallel_phase(
    *,
    args: argparse.Namespace,
    checkpoints: Checkpoints,
    out_dir: Path,
    logs_dir: Path,
    env: Dict[str, str],
) -> Dict[str, Any]:
    phase = "parallel"
    phase_out = out_dir / phase
    phase_out.mkdir(parents=True, exist_ok=True)

    worker_count = min(args.parallel_workers, args.single_games)
    shard_games = _split_count(args.single_games, worker_count)
    shard_cmds: List[Tuple[int, List[str], Path, Path]] = []
    for index, games in enumerate(shard_games):
        shard_id = f"s{index + 1:02d}"
        shard_out = phase_out / shard_id
        shard_out.mkdir(parents=True, exist_ok=True)
        shard_log = logs_dir / f"{phase}-{shard_id}.log"
        shard_summary = shard_out / "summary.json"
        cmd = _collect_command(
            args=args,
            python_bin=args.python_bin,
            games=games,
            seed_prefix=f"bench-selfplay-{phase}-{shard_id}",
            run_label=f"bench-selfplay-{phase}-{shard_id}",
            value_out=shard_out / "value.jsonl",
            opponent_out=shard_out / "opponent.jsonl",
            summary_out=shard_summary,
            value_ckpt=checkpoints.value,
            opponent_ckpt=checkpoints.opponent,
        )
        shard_cmds.append((index + 1, cmd, shard_log, shard_summary))

    print(
        f"[bench] phase={phase} workers={worker_count} "
        f"shardGames={','.join(str(g) for g in shard_games)}"
    )
    started = time.perf_counter()
    procs: List[Tuple[int, subprocess.Popen[Any], Path, Path]] = []
    for shard_index, cmd, shard_log, shard_summary in shard_cmds:
        log_file = shard_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        procs.append((shard_index, proc, shard_log, shard_summary))

    first_failure: Tuple[int, int] | None = None
    for shard_index, proc, shard_log, _ in procs:
        code = proc.wait()
        if code != 0 and first_failure is None:
            first_failure = (shard_index, code)
            for _, other, _, _ in procs:
                if other.poll() is None:
                    other.terminate()

    elapsed = time.perf_counter() - started
    if first_failure is not None:
        shard_index, code = first_failure
        raise SystemExit(
            "parallel benchmark shard failed: "
            f"shard={shard_index} returnCode={code} "
            f"log={logs_dir / f'parallel-s{shard_index:02d}.log'}"
        )

    total_games = 0
    total_transitions = 0
    weighted_turn_sum = 0.0
    for _, _, _, shard_summary in procs:
        payload = _read_collect_summary(shard_summary)
        total_games += int(payload["games"])
        total_transitions += int(payload["valueTransitions"])
        weighted_turn_sum += float(payload["averageTurn"]) * int(payload["games"])

    return {
        "games": total_games,
        "elapsedSeconds": elapsed,
        "gamesPerMinute": (total_games / elapsed) * 60.0,
        "valueTransitions": total_transitions,
        "avgTurn": (weighted_turn_sum / float(total_games)) if total_games > 0 else 0.0,
        "logs": [str(log_path) for _, _, log_path, _ in procs],
        "summaries": [str(summary_path) for _, _, _, summary_path in procs],
    }


def _collect_command(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    games: int,
    seed_prefix: str,
    run_label: str,
    value_out: Path,
    opponent_out: Path,
    summary_out: Path,
    value_ckpt: Path,
    opponent_ckpt: Path,
) -> List[str]:
    cmd = [
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
        str(args.search_worlds),
        "--search-rollouts",
        str(args.search_rollouts),
        "--search-depth",
        str(args.search_depth),
        "--search-max-root-actions",
        str(args.search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.search_rollout_epsilon),
        "--td-worlds",
        str(args.td_worlds),
        "--td-search-opponent-temperature",
        str(args.td_search_opponent_temperature),
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
        str(value_ckpt),
        "--player-a-td-search-opponent-checkpoint",
        str(opponent_ckpt),
        "--player-b-td-search-value-checkpoint",
        str(value_ckpt),
        "--player-b-td-search-opponent-checkpoint",
        str(opponent_ckpt),
    ]
    if args.td_search_sample_opponent_actions:
        cmd.append("--td-search-sample-opponent-actions")
    return cmd


def _run_command_logged(*, cmd: List[str], log_path: Path, env: Dict[str, str]) -> float:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise SystemExit(
            f"command failed returnCode={process.returncode} log={log_path}"
        )
    return elapsed


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
        latest = Checkpoints(
            source_run_id=summary_path.parent.name,
            value=Path(value_raw),
            opponent=Path(opponent_raw),
        )

    if latest is None:
        raise SystemExit(
            "no promoted checkpoints found in --td-loops-dir; provide explicit checkpoint paths"
        )
    return latest


def _apply_smoke_profile(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.single_games = 8
    args.parallel_workers = min(args.parallel_workers, 2)
    args.speedup_threshold = min(args.speedup_threshold, 1.1)
    args.progress_every_games = 1
    args.search_worlds = 2
    args.search_rollouts = 1
    args.search_depth = 8
    args.search_max_root_actions = 4
    args.search_rollout_epsilon = 0.04
    args.td_worlds = 2


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
    if args.single_games <= 0:
        raise SystemExit("--single-games must be > 0.")
    if args.parallel_workers <= 0:
        raise SystemExit("--parallel-workers must be > 0.")
    if args.speedup_threshold <= 0.0:
        raise SystemExit("--speedup-threshold must be > 0.")
    if args.progress_every_games <= 0:
        raise SystemExit("--progress-every-games must be > 0.")


if __name__ == "__main__":
    raise SystemExit(main())
