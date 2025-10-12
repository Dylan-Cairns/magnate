from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


_CHECKPOINT_REQUIRED_POLICIES = {
    "bc",
    "behavior-cloned",
    "behavior_cloned",
    "ppo",
}


@dataclass(frozen=True)
class BenchmarkRow:
    seed: int
    artifact: Path
    selection_score: float
    heuristic_win_rate: float
    random_win_rate: float


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Queue multiple canonical benchmark runs (sequential) by seed. "
            "Extra args are forwarded to scripts.benchmark."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Ordered seed list to benchmark sequentially (default: 2 3 4).",
    )
    parser.add_argument(
        "--candidate-policy",
        type=str,
        default="ppo",
        help="Candidate policy name (random|heuristic|bc|ppo).",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="artifacts/ppo_checkpoint_seed{seed}.pt",
        help="Checkpoint input template. Must include '{seed}'.",
    )
    parser.add_argument(
        "--out-template",
        type=str,
        default="artifacts/benchmarks/ppo_seed{seed}.json",
        help="Benchmark artifact output template. Must include '{seed}'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args, forward_args = parser.parse_known_args()
    return args, forward_args


def main() -> int:
    args, forward_args = parse_args()
    normalized_policy = args.candidate_policy.strip().lower()

    _validate_templates(args.checkpoint_template, args.out_template)
    _validate_forward_args(forward_args)
    _validate_seeds(args.seeds)

    queue_start = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    print(
        f"[queue] start utc={start_ts} seeds={args.seeds} "
        f"candidatePolicy={args.candidate_policy}"
    )
    if forward_args:
        joined_args = " ".join(shlex.quote(item) for item in forward_args)
        print(f"[queue] forwarding args: {joined_args}")

    rows: List[BenchmarkRow] = []
    total = len(args.seeds)
    for index, seed in enumerate(args.seeds, start=1):
        checkpoint = args.checkpoint_template.format(seed=seed)
        out_path = Path(args.out_template.format(seed=seed))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "scripts.benchmark",
            "--candidate-policy",
            args.candidate_policy,
            "--out",
            str(out_path),
            *forward_args,
        ]
        if _policy_requires_checkpoint(normalized_policy):
            command.extend(["--candidate-checkpoint", checkpoint])

        print(
            f"[queue] run {index}/{total} seed={seed} out={out_path} "
            f"checkpoint={checkpoint if _policy_requires_checkpoint(normalized_policy) else 'n/a'}"
        )
        print(f"[queue] command: {' '.join(shlex.quote(item) for item in command)}")
        if args.dry_run:
            continue

        run_start = time.perf_counter()
        completed = subprocess.run(command)
        run_elapsed_minutes = (time.perf_counter() - run_start) / 60.0
        if completed.returncode != 0:
            print(
                f"[queue] failed seed={seed} returnCode={completed.returncode} "
                f"elapsedMin={run_elapsed_minutes:.1f}"
            )
            return completed.returncode

        row = _load_row(seed=seed, artifact_path=out_path)
        rows.append(row)
        print(
            f"[queue] finished seed={seed} elapsedMin={run_elapsed_minutes:.1f} "
            f"score={row.selection_score:.3f} "
            f"heuristicWinRate={row.heuristic_win_rate:.3f} "
            f"randomWinRate={row.random_win_rate:.3f}"
        )

    total_elapsed_minutes = (time.perf_counter() - queue_start) / 60.0
    if not args.dry_run:
        _print_summary(rows)
    print(f"[queue] done elapsedMin={total_elapsed_minutes:.1f}")
    return 0


def _policy_requires_checkpoint(policy_name: str) -> bool:
    return policy_name in _CHECKPOINT_REQUIRED_POLICIES


def _validate_templates(checkpoint_template: str, out_template: str) -> None:
    if "{seed}" not in checkpoint_template:
        raise SystemExit("--checkpoint-template must include '{seed}'.")
    if "{seed}" not in out_template:
        raise SystemExit("--out-template must include '{seed}'.")


def _validate_forward_args(forward_args: List[str]) -> None:
    disallowed = {
        "--candidate-policy",
        "--candidate-checkpoint",
        "--out",
    }
    for item in forward_args:
        if item in disallowed:
            raise SystemExit(
                "Do not pass "
                f"{item!r} to benchmark_queue; it is set by the queue runner."
            )


def _validate_seeds(seeds: List[int]) -> None:
    seen = set()
    for seed in seeds:
        if seed in seen:
            raise SystemExit(f"Duplicate seed in --seeds: {seed}")
        seen.add(seed)


def _load_row(seed: int, artifact_path: Path) -> BenchmarkRow:
    if not artifact_path.exists():
        raise SystemExit(f"Benchmark output not found: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid benchmark JSON in {artifact_path}: {exc}") from exc

    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Missing 'results' object in {artifact_path}")

    return BenchmarkRow(
        seed=seed,
        artifact=artifact_path,
        selection_score=_read_float(results, "selectionScore", artifact_path),
        heuristic_win_rate=_read_float(results, "heuristicWinRate", artifact_path),
        random_win_rate=_read_float(results, "randomWinRate", artifact_path),
    )


def _read_float(payload: dict[str, Any], key: str, artifact_path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise SystemExit(f"Missing numeric field 'results.{key}' in {artifact_path}")
    return float(value)


def _print_summary(rows: List[BenchmarkRow]) -> None:
    if not rows:
        return

    ordered = sorted(rows, key=lambda row: row.selection_score, reverse=True)
    print("[queue] summary sorted by selectionScore (desc):")
    print("[queue] seed  score  heuristic  random  artifact")
    for row in ordered:
        print(
            f"[queue] {row.seed:>4}  {row.selection_score:>5.3f}  "
            f"{row.heuristic_win_rate:>9.3f}  {row.random_win_rate:>6.3f}  {row.artifact}"
        )

    best = ordered[0]
    print(
        f"[queue] best seed={best.seed} score={best.selection_score:.3f} "
        f"artifact={best.artifact}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
