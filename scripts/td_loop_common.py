from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class LoopCheckpoint:
    step: int
    value_path: Path | None
    opponent_path: Path | None


def build_train_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    value_replay: Path,
    opponent_replay: Path,
    train_summary_path: Path,
    train_checkpoint_root: Path,
    run_id: str,
    warm_start_value: Path | None,
    warm_start_opponent: Path | None,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.train_td",
        "--value-replay",
        str(value_replay),
        "--opponent-replay",
        str(opponent_replay),
        "--steps",
        str(args.train_steps),
        "--value-batch-size",
        str(args.train_value_batch_size),
        "--opponent-batch-size",
        str(args.train_opponent_batch_size),
        "--seed",
        str(args.train_seed),
        "--hidden-dim",
        str(args.train_hidden_dim),
        "--gamma",
        str(args.train_gamma),
        "--value-learning-rate",
        str(args.train_value_learning_rate),
        "--value-weight-decay",
        str(args.train_value_weight_decay),
        "--opponent-learning-rate",
        str(args.train_opponent_learning_rate),
        "--opponent-weight-decay",
        str(args.train_opponent_weight_decay),
        "--max-grad-norm",
        str(args.train_max_grad_norm),
        "--target-sync-interval",
        str(args.train_target_sync_interval),
        "--value-target-mode",
        args.train_value_target_mode,
        "--td-lambda",
        str(args.train_td_lambda),
        "--save-every-steps",
        str(args.train_save_every_steps),
        "--progress-every-steps",
        str(args.train_progress_every_steps),
        "--out-dir",
        str(train_checkpoint_root),
        "--run-label",
        run_id,
        "--summary-out",
        str(train_summary_path),
    ]
    if args.train_num_threads is not None:
        command.extend(["--num-threads", str(args.train_num_threads)])
    if args.train_num_interop_threads is not None:
        command.extend(["--num-interop-threads", str(args.train_num_interop_threads)])
    if args.train_use_mse_loss:
        command.append("--use-mse-loss")
    if args.train_disable_value:
        command.append("--disable-value")
    if args.train_disable_opponent:
        command.append("--disable-opponent")
    if warm_start_value is not None:
        command.extend(["--warm-start-value-checkpoint", str(warm_start_value)])
    if warm_start_opponent is not None:
        command.extend(["--warm-start-opponent-checkpoint", str(warm_start_opponent)])
    return command


def checkpoints_from_train_summary(payload: Dict[str, Any]) -> List[LoopCheckpoint]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit("Train summary is missing results payload.")
    checkpoints = results.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise SystemExit("Train summary is missing results.checkpoints list.")

    out: List[LoopCheckpoint] = []
    for entry in checkpoints:
        if not isinstance(entry, dict):
            raise SystemExit("Train summary checkpoint entry must be an object.")
        step = entry.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise SystemExit(f"Train summary checkpoint has invalid step: {step!r}")
        value_raw = entry.get("value")
        opponent_raw = entry.get("opponent")
        value_path = Path(str(value_raw)) if isinstance(value_raw, str) else None
        opponent_path = Path(str(opponent_raw)) if isinstance(opponent_raw, str) else None
        out.append(
            LoopCheckpoint(
                step=step,
                value_path=value_path,
                opponent_path=opponent_path,
            )
        )
    if not out:
        raise SystemExit("No checkpoints were emitted by training.")
    return sorted(out, key=lambda checkpoint: checkpoint.step)


def select_latest_checkpoint(
    *,
    checkpoints: Sequence[LoopCheckpoint],
    candidate_policy: str,
) -> LoopCheckpoint:
    if candidate_policy == "td-search":
        eligible = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.value_path is not None and checkpoint.opponent_path is not None
        ]
    elif candidate_policy == "td-value":
        eligible = [checkpoint for checkpoint in checkpoints if checkpoint.value_path is not None]
    else:
        raise SystemExit(f"Unsupported eval candidate policy: {candidate_policy!r}")

    if not eligible:
        raise SystemExit(f"No eligible checkpoints for candidate policy {candidate_policy}.")
    return eligible[-1]


def concat_jsonl_files(
    *,
    inputs: Sequence[Path],
    output: Path,
    delete_inputs_after_merge: bool = False,
) -> None:
    if not inputs:
        raise SystemExit("No JSONL shard inputs were provided for merge.")
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"Missing collect shard artifact: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if not delete_inputs_after_merge:
        with output.open("w", encoding="utf-8") as target:
            for path in inputs:
                with path.open("r", encoding="utf-8") as source:
                    for line in source:
                        target.write(line)
        return

    first = inputs[0]
    if first.resolve() != output.resolve():
        if output.exists():
            output.unlink()
        first.replace(output)
    remaining = inputs[1:]
    with output.open("a", encoding="utf-8") as target:
        for path in remaining:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    target.write(line)
            path.unlink()


def run_step(
    *,
    name: str,
    command: Sequence[str],
    heartbeat_minutes: float = 0.0,
    progress_path: Path | None = None,
    log_prefix: str = "[td-loop]",
) -> None:
    print(f"{log_prefix} step {name}: {join_command(command)}")
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_thread = _start_output_pump(
        process=process,
        output_prefix=f"{log_prefix} output step={name} | ",
    )

    heartbeat_seconds = max(0.0, heartbeat_minutes * 60.0)
    next_heartbeat = started + heartbeat_seconds if heartbeat_seconds > 0 else float("inf")

    if progress_path is not None:
        write_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "step": name,
                "status": "running",
                "elapsedMinutes": 0.0,
            },
        )

    while True:
        return_code = process.poll()
        if return_code is not None:
            break

        now = time.perf_counter()
        if now >= next_heartbeat:
            elapsed_minutes = (now - started) / 60.0
            print(f"{log_prefix} heartbeat step={name} elapsedMin={elapsed_minutes:.1f}")
            if progress_path is not None:
                write_progress(
                    progress_path,
                    {
                        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                        "step": name,
                        "status": "running",
                        "elapsedMinutes": round(elapsed_minutes, 3),
                    },
                )
            next_heartbeat = now + heartbeat_seconds
        time.sleep(2.0)

    if output_thread is not None:
        output_thread.join(timeout=5.0)
    if process.stdout is not None:
        process.stdout.close()

    elapsed_minutes = (time.perf_counter() - started) / 60.0
    if return_code != 0:
        if progress_path is not None:
            write_progress(
                progress_path,
                {
                    "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                    "step": name,
                    "status": "failed",
                    "elapsedMinutes": round(elapsed_minutes, 3),
                    "returnCode": int(return_code),
                },
            )
        raise SystemExit(
            f"{log_prefix} failed step={name} returnCode={return_code} "
            f"elapsedMin={elapsed_minutes:.1f}"
        )

    if progress_path is not None:
        write_progress(
            progress_path,
            {
                "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "step": name,
                "status": "completed",
                "elapsedMinutes": round(elapsed_minutes, 3),
                "returnCode": int(return_code),
            },
        )
    print(f"{log_prefix} completed step={name} elapsedMin={elapsed_minutes:.1f}")


def _start_output_pump(
    *,
    process: subprocess.Popen[str],
    output_prefix: str,
) -> threading.Thread | None:
    if process.stdout is None:
        return None

    def pump() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                sys.stdout.write(f"{output_prefix}{line}\n")
            else:
                sys.stdout.write(f"{output_prefix.rstrip()}\n")
            sys.stdout.flush()

    thread = threading.Thread(
        target=pump,
        name=f"magnate-step-output-{process.pid}",
        daemon=True,
    )
    thread.start()
    return thread


def write_progress(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path, *, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def join_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)
