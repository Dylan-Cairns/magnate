from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import List


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Queue multiple PPO training runs (sequential) by seed. "
            "Extra args are forwarded to scripts.train_ppo."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Ordered seed list to train sequentially (default: 2 3 4).",
    )
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="artifacts/ppo_checkpoint_seed{seed}.pt",
        help="Checkpoint output template. Must include '{seed}'.",
    )
    parser.add_argument(
        "--seed-prefix-template",
        type=str,
        default="ppo-seed{seed}",
        help="Seed prefix template. Must include '{seed}'.",
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
    _validate_templates(args.checkpoint_template, args.seed_prefix_template)
    _validate_forward_args(forward_args)
    _validate_seeds(args.seeds)

    start_ts = datetime.now(timezone.utc).isoformat()
    queue_start = time.perf_counter()
    print(f"[queue] start utc={start_ts} seeds={args.seeds}")
    if forward_args:
        print(f"[queue] forwarding args: {' '.join(shlex.quote(item) for item in forward_args)}")

    total = len(args.seeds)
    for index, seed in enumerate(args.seeds, start=1):
        checkpoint_out = args.checkpoint_template.format(seed=seed)
        seed_prefix = args.seed_prefix_template.format(seed=seed)

        command = [
            sys.executable,
            "-m",
            "scripts.train_ppo",
            "--checkpoint-out",
            checkpoint_out,
            "--seed",
            str(seed),
            "--seed-prefix",
            seed_prefix,
            *forward_args,
        ]

        print(
            f"[queue] run {index}/{total} seed={seed} "
            f"checkpoint={checkpoint_out} seedPrefix={seed_prefix}"
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
        print(f"[queue] finished seed={seed} elapsedMin={run_elapsed_minutes:.1f}")

    total_elapsed_minutes = (time.perf_counter() - queue_start) / 60.0
    print(f"[queue] done elapsedMin={total_elapsed_minutes:.1f}")
    return 0


def _validate_templates(checkpoint_template: str, seed_prefix_template: str) -> None:
    if "{seed}" not in checkpoint_template:
        raise SystemExit("--checkpoint-template must include '{seed}'.")
    if "{seed}" not in seed_prefix_template:
        raise SystemExit("--seed-prefix-template must include '{seed}'.")


def _validate_forward_args(forward_args: List[str]) -> None:
    disallowed = {
        "--seed",
        "--seed-prefix",
        "--checkpoint-out",
    }
    for item in forward_args:
        if item in disallowed:
            raise SystemExit(
                "Do not pass "
                f"{item!r} to train_ppo_queue; it is set per-seed by the queue runner."
            )


def _validate_seeds(seeds: List[int]) -> None:
    seen = set()
    for seed in seeds:
        if seed in seen:
            raise SystemExit(f"Duplicate seed in --seeds: {seed}")
        seen.add(seed)


if __name__ == "__main__":
    raise SystemExit(main())
