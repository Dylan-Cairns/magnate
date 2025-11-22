from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class LoopCheckpoint:
    step: int
    value_path: Path | None
    opponent_path: Path | None


@dataclass(frozen=True)
class EvalRow:
    artifact: Path
    opponent_policy: str
    candidate_win_rate: float
    ci_low: float
    ci_high: float
    side_gap: float
    candidate_wins: int
    opponent_wins: int
    draws: int
    total_games: int


@dataclass(frozen=True)
class CertifyComparison:
    row: EvalRow
    checks: Dict[str, bool]
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TD loop in gate-first mode: multiple collect/train chunks, "
            "then gate eval and optional certify panel."
        )
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run stage scripts.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/td_loops"),
        help="Root output directory for loop artifacts.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="td-loop",
        help="Run label used in artifact naming.",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help=(
            "Apply preset worker/thread profile for cloud machines."
        ),
    )
    parser.add_argument(
        "--cloud-vcpus",
        type=int,
        default=8,
        choices=(8, 16, 32),
        help="vCPU preset used when --cloud is set (8|16|32).",
    )

    parser.add_argument(
        "--chunks-per-gate",
        type=int,
        default=3,
        help="Number of collect/train chunks before a gate decision.",
    )

    parser.add_argument("--collect-games", type=int, default=1500)
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="Parallel collection shards. 1 disables sharding.",
    )
    parser.add_argument("--collect-seed-prefix", type=str, default="td-loop-collect")
    parser.add_argument("--collect-player-a-policy", type=str, default="search")
    parser.add_argument("--collect-player-b-policy", type=str, default="search")
    parser.add_argument("--collect-progress-every-games", type=int, default=50)
    parser.add_argument("--collect-search-worlds", type=int, default=6)
    parser.add_argument("--collect-search-rollouts", type=int, default=1)
    parser.add_argument("--collect-search-depth", type=int, default=14)
    parser.add_argument("--collect-search-max-root-actions", type=int, default=6)
    parser.add_argument("--collect-search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--collect-td-value-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-worlds", type=int, default=8)
    parser.add_argument("--collect-td-search-value-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-search-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--collect-td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--collect-td-search-sample-opponent-actions", action="store_true")

    parser.add_argument("--train-steps", type=int, default=15000)
    parser.add_argument("--train-value-batch-size", type=int, default=128)
    parser.add_argument("--train-opponent-batch-size", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--train-hidden-dim", type=int, default=256)
    parser.add_argument("--train-gamma", type=float, default=0.995)
    parser.add_argument("--train-value-learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-value-weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-opponent-learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-opponent-weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-target-sync-interval", type=int, default=200)
    parser.add_argument(
        "--train-value-target-mode",
        type=str,
        choices=("td0", "td-lambda"),
        default="td0",
    )
    parser.add_argument("--train-td-lambda", type=float, default=0.7)
    parser.add_argument("--train-use-mse-loss", action="store_true")
    parser.add_argument("--train-disable-value", action="store_true")
    parser.add_argument("--train-disable-opponent", action="store_true")
    parser.add_argument("--train-warm-start-value-checkpoint", type=Path, default=None)
    parser.add_argument("--train-warm-start-opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--train-save-every-steps", type=int, default=1000)
    parser.add_argument("--train-progress-every-steps", type=int, default=50)
    parser.add_argument(
        "--train-num-threads",
        type=int,
        default=None,
        help=(
            "Optional torch intra-op CPU thread count for scripts.train_td. "
            "If --cloud is set and this is omitted, defaults to --cloud-vcpus."
        ),
    )
    parser.add_argument(
        "--train-num-interop-threads",
        type=int,
        default=None,
        help=(
            "Optional torch inter-op CPU thread count for scripts.train_td. "
            "If --cloud is set and this is omitted, defaults to 1."
        ),
    )

    parser.add_argument(
        "--eval-candidate-policy",
        type=str,
        choices=("td-search", "td-value"),
        default="td-search",
    )
    parser.add_argument("--eval-search-worlds", type=int, default=6)
    parser.add_argument("--eval-search-rollouts", type=int, default=1)
    parser.add_argument("--eval-search-depth", type=int, default=14)
    parser.add_argument("--eval-search-max-root-actions", type=int, default=6)
    parser.add_argument("--eval-search-rollout-epsilon", type=float, default=0.04)
    parser.add_argument("--eval-td-worlds", type=int, default=8)
    parser.add_argument("--eval-td-search-opponent-temperature", type=float, default=1.0)
    parser.add_argument("--eval-td-search-sample-opponent-actions", action="store_true")

    parser.add_argument("--gate-workers", type=int, default=1)
    parser.add_argument("--gate-seed-prefix", type=str, default="td-loop-gate")
    parser.add_argument("--gate-seed-start-index", type=int, default=0)
    parser.add_argument("--gate-h0-win-rate", type=float, default=0.50)
    parser.add_argument("--gate-h1-win-rate", type=float, default=0.55)
    parser.add_argument("--gate-alpha", type=float, default=0.05)
    parser.add_argument("--gate-beta", type=float, default=0.10)
    parser.add_argument("--gate-batch-games-per-side", type=int, default=25)
    parser.add_argument("--gate-max-games-per-side", type=int, default=400)
    parser.add_argument("--gate-max-side-gap", type=float, default=0.08)

    parser.add_argument("--certify-workers", type=int, default=1)
    parser.add_argument("--certify-games-per-side", type=int, default=400)
    parser.add_argument(
        "--certify-opponents",
        type=str,
        nargs="+",
        default=["search", "heuristic"],
        choices=("random", "heuristic", "search"),
        help="Opponent panel for certify mode.",
    )
    parser.add_argument("--certify-seed-prefix", type=str, default="td-loop-certify")
    parser.add_argument(
        "--certify-seed-start-index",
        type=int,
        default=100000,
        help="Seed index offset for certify panel (disjoint from gate seeds).",
    )
    parser.add_argument("--certify-min-win-rate", type=float, default=0.55)
    parser.add_argument("--certify-max-side-gap", type=float, default=0.08)
    parser.add_argument("--certify-min-ci-low", type=float, default=0.50)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cloud:
        _apply_cloud_profile(args)

    _require_supported_runtime(args.python_bin)
    _validate_args(args)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_id = f"{stamp}-{_slug(args.run_label)}"
    run_dir = args.artifact_dir / run_id
    chunks_dir = run_dir / "chunks"
    eval_dir = run_dir / "evals"
    loop_summary_path = run_dir / "loop.summary.json"
    for path in (run_dir, chunks_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)

    loop_started = time.perf_counter()
    commands: Dict[str, Any] = {"chunks": []}
    chunk_rows: List[Dict[str, Any]] = []

    latest_checkpoint: LoopCheckpoint | None = None
    warm_value = args.train_warm_start_value_checkpoint
    warm_opponent = args.train_warm_start_opponent_checkpoint

    for chunk_index in range(1, args.chunks_per_gate + 1):
        chunk_label = f"chunk-{chunk_index:03d}"
        chunk_dir = chunks_dir / chunk_label
        replay_dir = chunk_dir / "replay"
        train_dir = chunk_dir / "train"
        for path in (chunk_dir, replay_dir, train_dir):
            path.mkdir(parents=True, exist_ok=True)

        collect_value_path = replay_dir / "self_play.value.jsonl"
        collect_opponent_path = replay_dir / "self_play.opponent.jsonl"
        collect_summary_path = replay_dir / "self_play.summary.json"

        collect_stage = _run_collect_stage(
            python_bin=args.python_bin,
            args=args,
            replay_dir=replay_dir,
            collect_value_path=collect_value_path,
            collect_opponent_path=collect_opponent_path,
            collect_summary_path=collect_summary_path,
            run_id=f"{run_id}-{chunk_label}",
            seed_prefix=f"{args.collect_seed_prefix}-{chunk_label}",
        )

        train_summary_path = train_dir / "summary.json"
        train_checkpoint_root = train_dir / "checkpoints"
        train_command = _build_train_command(
            python_bin=args.python_bin,
            args=args,
            value_replay=collect_value_path,
            opponent_replay=collect_opponent_path,
            train_summary_path=train_summary_path,
            train_checkpoint_root=train_checkpoint_root,
            run_id=f"{run_id}-{chunk_label}",
            warm_start_value=warm_value,
            warm_start_opponent=warm_opponent,
        )
        _run_step(name=f"train[{chunk_label}]", command=train_command)

        train_summary = _read_json(train_summary_path, label=f"train summary {chunk_label}")
        checkpoints = _checkpoints_from_train_summary(train_summary)
        latest_checkpoint = _select_latest_checkpoint(
            checkpoints=checkpoints,
            candidate_policy=args.eval_candidate_policy,
        )
        if latest_checkpoint.value_path is not None:
            warm_value = latest_checkpoint.value_path
        if latest_checkpoint.opponent_path is not None:
            warm_opponent = latest_checkpoint.opponent_path

        commands["chunks"].append(
            {
                "chunk": chunk_label,
                "collect": collect_stage,
                "train": train_command,
            }
        )
        chunk_rows.append(
            {
                "chunk": chunk_label,
                "collectSummary": str(collect_summary_path),
                "trainSummary": str(train_summary_path),
                "latestCheckpoint": {
                    "step": latest_checkpoint.step,
                    "value": str(latest_checkpoint.value_path) if latest_checkpoint.value_path else None,
                    "opponent": (
                        str(latest_checkpoint.opponent_path)
                        if latest_checkpoint.opponent_path
                        else None
                    ),
                },
            }
        )

    if latest_checkpoint is None:
        raise SystemExit("No latest checkpoint available after chunk training.")

    gate_artifact = eval_dir / "gate.json"
    gate_command = _build_eval_command(
        python_bin=args.python_bin,
        args=args,
        checkpoint=latest_checkpoint,
        mode="gate",
        opponent_policy="search",
        seed_prefix=args.gate_seed_prefix,
        seed_start_index=args.gate_seed_start_index,
        workers=args.gate_workers,
        games_per_side=args.gate_max_games_per_side,
        out_path=gate_artifact,
    )
    commands["gate"] = gate_command
    _run_step(name="gate", command=gate_command)
    gate_payload = _read_json(gate_artifact, label="gate artifact")
    gate_outcome = _read_gate_outcome(gate_payload=gate_payload, artifact=gate_artifact)

    certify_payload: Dict[str, Any] = {
        "ran": False,
        "overallPassed": False,
        "comparisons": [],
    }
    commands["certify"] = []

    if gate_outcome["decision"] == "accepted":
        certify_payload["ran"] = True
        seed_index = args.certify_seed_start_index
        comparisons: List[CertifyComparison] = []
        certify_commands: List[List[str]] = []

        for opponent_policy in args.certify_opponents:
            certify_artifact = eval_dir / f"certify-{opponent_policy}.json"
            certify_command = _build_eval_command(
                python_bin=args.python_bin,
                args=args,
                checkpoint=latest_checkpoint,
                mode="certify",
                opponent_policy=opponent_policy,
                seed_prefix=f"{args.certify_seed_prefix}-{opponent_policy}",
                seed_start_index=seed_index,
                workers=args.certify_workers,
                games_per_side=args.certify_games_per_side,
                out_path=certify_artifact,
            )
            certify_commands.append(certify_command)
            _run_step(name=f"certify[{opponent_policy}]", command=certify_command)
            row = _read_eval_row(
                path=certify_artifact,
                opponent_policy=opponent_policy,
            )
            comparisons.append(_evaluate_certify_result(row=row, args=args))
            seed_index += args.certify_games_per_side

        commands["certify"] = certify_commands
        certify_payload["comparisons"] = [
            {
                "opponentPolicy": comparison.row.opponent_policy,
                "artifact": str(comparison.row.artifact),
                "candidateWinRate": comparison.row.candidate_win_rate,
                "candidateWinRateCi95": {
                    "low": comparison.row.ci_low,
                    "high": comparison.row.ci_high,
                },
                "sideGap": comparison.row.side_gap,
                "candidateWins": comparison.row.candidate_wins,
                "opponentWins": comparison.row.opponent_wins,
                "draws": comparison.row.draws,
                "totalGames": comparison.row.total_games,
                "checks": dict(comparison.checks),
                "passed": comparison.passed,
            }
            for comparison in comparisons
        ]
        certify_payload["overallPassed"] = all(comparison.passed for comparison in comparisons)

    promotion = _promotion_decision(
        gate_decision=gate_outcome["decision"],
        gate_reason=gate_outcome["reason"],
        certify=certify_payload,
    )

    loop_elapsed = time.perf_counter() - loop_started
    payload: Dict[str, Any] = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "elapsedSeconds": round(loop_elapsed, 3),
        "config": _config_payload(args),
        "commands": commands,
        "artifacts": {
            "runDir": str(run_dir),
            "loopSummary": str(loop_summary_path),
            "chunksDir": str(chunks_dir),
            "evalDir": str(eval_dir),
            "gateArtifact": str(gate_artifact),
        },
        "chunks": chunk_rows,
        "gate": gate_outcome,
        "certify": certify_payload,
        "promotion": promotion,
    }
    loop_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "runId": run_id,
                "runDir": str(run_dir),
                "loopSummary": str(loop_summary_path),
                "gateDecision": gate_outcome["decision"],
                "gateReason": gate_outcome["reason"],
                "certifyRan": bool(certify_payload["ran"]),
                "certifyPassed": bool(certify_payload["overallPassed"]),
                "promoted": bool(promotion["promoted"]),
                "promotionReason": promotion["reason"],
            },
            indent=2,
        )
    )
    return 0


def _build_collect_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    games: int,
    seed_prefix: str,
    run_label: str,
    value_out: Path,
    opponent_out: Path,
    summary_out: Path,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.collect_td_self_play",
        "--games",
        str(games),
        "--seed-prefix",
        seed_prefix,
        "--player-a-policy",
        args.collect_player_a_policy,
        "--player-b-policy",
        args.collect_player_b_policy,
        "--search-worlds",
        str(args.collect_search_worlds),
        "--search-rollouts",
        str(args.collect_search_rollouts),
        "--search-depth",
        str(args.collect_search_depth),
        "--search-max-root-actions",
        str(args.collect_search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.collect_search_rollout_epsilon),
        "--td-worlds",
        str(args.collect_td_worlds),
        "--td-search-opponent-temperature",
        str(args.collect_td_search_opponent_temperature),
        "--run-label",
        run_label,
        "--value-out",
        str(value_out),
        "--opponent-out",
        str(opponent_out),
        "--summary-out",
        str(summary_out),
        "--progress-every-games",
        str(args.collect_progress_every_games),
    ]
    if args.collect_td_value_checkpoint is not None:
        command.extend(["--td-value-checkpoint", str(args.collect_td_value_checkpoint)])
    if args.collect_td_search_value_checkpoint is not None:
        command.extend(
            ["--td-search-value-checkpoint", str(args.collect_td_search_value_checkpoint)]
        )
    if args.collect_td_search_opponent_checkpoint is not None:
        command.extend(
            [
                "--td-search-opponent-checkpoint",
                str(args.collect_td_search_opponent_checkpoint),
            ]
        )
    if args.collect_td_search_sample_opponent_actions:
        command.append("--td-search-sample-opponent-actions")
    return command


def _run_collect_stage(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    replay_dir: Path,
    collect_value_path: Path,
    collect_opponent_path: Path,
    collect_summary_path: Path,
    run_id: str,
    seed_prefix: str,
) -> Dict[str, Any]:
    if args.collect_workers == 1:
        command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            games=args.collect_games,
            seed_prefix=seed_prefix,
            run_label=run_id,
            value_out=collect_value_path,
            opponent_out=collect_opponent_path,
            summary_out=collect_summary_path,
        )
        _run_step(name=f"collect[{run_id}]", command=command)
        return {"mode": "single", "commands": [command]}

    worker_count = min(args.collect_workers, args.collect_games)
    shard_sizes = _split_count(args.collect_games, worker_count)
    shards_dir = replay_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_commands: List[List[str]] = []
    shard_value_paths: List[Path] = []
    shard_opponent_paths: List[Path] = []
    shard_summary_paths: List[Path] = []

    for shard_index, shard_games in enumerate(shard_sizes):
        shard_id = f"s{shard_index + 1:02d}"
        shard_value = shards_dir / f"{shard_id}.value.jsonl"
        shard_opponent = shards_dir / f"{shard_id}.opponent.jsonl"
        shard_summary = shards_dir / f"{shard_id}.summary.json"
        shard_command = _build_collect_command(
            python_bin=python_bin,
            args=args,
            games=shard_games,
            seed_prefix=f"{seed_prefix}-{shard_id}",
            run_label=f"{run_id}-{shard_id}",
            value_out=shard_value,
            opponent_out=shard_opponent,
            summary_out=shard_summary,
        )
        shard_commands.append(shard_command)
        shard_value_paths.append(shard_value)
        shard_opponent_paths.append(shard_opponent)
        shard_summary_paths.append(shard_summary)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_step, name=f"collect[{run_id} {index + 1}/{worker_count}]", command=command): index
            for index, command in enumerate(shard_commands)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                raise SystemExit(f"[td-loop] failed collect shard: {exc}") from exc

    _concat_jsonl_files(
        inputs=shard_value_paths,
        output=collect_value_path,
        delete_inputs_after_merge=True,
    )
    _concat_jsonl_files(
        inputs=shard_opponent_paths,
        output=collect_opponent_path,
        delete_inputs_after_merge=True,
    )
    _write_merged_collect_summary(
        args=args,
        shard_summaries=shard_summary_paths,
        merged_summary_path=collect_summary_path,
        merged_value_path=collect_value_path,
        merged_opponent_path=collect_opponent_path,
    )

    return {
        "mode": "sharded",
        "workers": worker_count,
        "commands": shard_commands,
        "shards": [
            {
                "value": str(shard_value_paths[index]),
                "opponent": str(shard_opponent_paths[index]),
                "summary": str(shard_summary_paths[index]),
                "games": shard_sizes[index],
            }
            for index in range(worker_count)
        ],
        "merged": {
            "value": str(collect_value_path),
            "opponent": str(collect_opponent_path),
            "summary": str(collect_summary_path),
        },
    }


def _concat_jsonl_files(
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


def _write_merged_collect_summary(
    *,
    args: argparse.Namespace,
    shard_summaries: Sequence[Path],
    merged_summary_path: Path,
    merged_value_path: Path,
    merged_opponent_path: Path,
) -> None:
    winners = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    total_games = 0
    total_turn_weighted = 0.0
    total_value_transitions = 0
    total_opponent_samples = 0
    shard_rows: List[Dict[str, Any]] = []

    for shard_path in shard_summaries:
        payload = _read_json(shard_path, label=f"collect shard summary {shard_path.name}")
        results = payload.get("results")
        if not isinstance(results, dict):
            raise SystemExit(f"collect shard summary missing results payload: {shard_path}")
        shard_games = int(results["games"])
        shard_avg_turn = float(results["averageTurn"])
        shard_winners = results.get("winners")
        if not isinstance(shard_winners, dict):
            raise SystemExit(f"collect shard summary missing winners payload: {shard_path}")
        shard_value_count = int(results["valueTransitions"])
        shard_opponent_count = int(results["opponentSamples"])

        total_games += shard_games
        total_turn_weighted += shard_avg_turn * shard_games
        total_value_transitions += shard_value_count
        total_opponent_samples += shard_opponent_count
        for key in ("PlayerA", "PlayerB", "Draw"):
            winners[key] += int(shard_winners.get(key, 0))

        shard_rows.append(
            {
                "path": str(shard_path),
                "games": shard_games,
                "averageTurn": shard_avg_turn,
                "winners": {
                    "PlayerA": int(shard_winners.get("PlayerA", 0)),
                    "PlayerB": int(shard_winners.get("PlayerB", 0)),
                    "Draw": int(shard_winners.get("Draw", 0)),
                },
                "valueTransitions": shard_value_count,
                "opponentSamples": shard_opponent_count,
            }
        )

    if total_games != args.collect_games:
        raise SystemExit(
            "Merged collect games mismatch. "
            f"expected={args.collect_games} actual={total_games}"
        )

    merged_payload = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "games": args.collect_games,
            "seedPrefix": args.collect_seed_prefix,
            "playerAPolicy": args.collect_player_a_policy,
            "playerBPolicy": args.collect_player_b_policy,
            "collectWorkers": args.collect_workers,
            "search": {
                "worlds": args.collect_search_worlds,
                "rollouts": args.collect_search_rollouts,
                "depth": args.collect_search_depth,
                "maxRootActions": args.collect_search_max_root_actions,
                "rolloutEpsilon": args.collect_search_rollout_epsilon,
            },
        },
        "results": {
            "games": total_games,
            "winners": winners,
            "averageTurn": (total_turn_weighted / float(total_games)) if total_games > 0 else 0.0,
            "valueTransitions": total_value_transitions,
            "opponentSamples": total_opponent_samples,
        },
        "artifacts": {
            "valueTransitions": str(merged_value_path),
            "opponentSamples": str(merged_opponent_path),
            "summary": str(merged_summary_path),
        },
        "shards": shard_rows,
    }
    merged_summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged_summary_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")


def _split_count(total: int, workers: int) -> List[int]:
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _build_train_command(
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


def _build_eval_command(
    *,
    python_bin: Path,
    args: argparse.Namespace,
    checkpoint: LoopCheckpoint,
    mode: str,
    opponent_policy: str,
    seed_prefix: str,
    seed_start_index: int,
    workers: int,
    games_per_side: int,
    out_path: Path,
) -> List[str]:
    command = [
        str(python_bin),
        "-m",
        "scripts.eval_suite",
        "--mode",
        mode,
        "--workers",
        str(workers),
        "--seed-prefix",
        seed_prefix,
        "--seed-start-index",
        str(seed_start_index),
        "--candidate-policy",
        args.eval_candidate_policy,
        "--opponent-policy",
        opponent_policy,
        "--search-worlds",
        str(args.eval_search_worlds),
        "--search-rollouts",
        str(args.eval_search_rollouts),
        "--search-depth",
        str(args.eval_search_depth),
        "--search-max-root-actions",
        str(args.eval_search_max_root_actions),
        "--search-rollout-epsilon",
        str(args.eval_search_rollout_epsilon),
        "--td-worlds",
        str(args.eval_td_worlds),
        "--out",
        str(out_path),
    ]
    if mode == "certify":
        command.extend(["--games-per-side", str(games_per_side)])
    else:
        command.extend(
            [
                "--gate-h0-win-rate",
                str(args.gate_h0_win_rate),
                "--gate-h1-win-rate",
                str(args.gate_h1_win_rate),
                "--gate-alpha",
                str(args.gate_alpha),
                "--gate-beta",
                str(args.gate_beta),
                "--gate-batch-games-per-side",
                str(args.gate_batch_games_per_side),
                "--gate-max-games-per-side",
                str(args.gate_max_games_per_side),
                "--gate-max-side-gap",
                str(args.gate_max_side_gap),
            ]
        )

    if args.eval_candidate_policy == "td-search":
        if checkpoint.value_path is None or checkpoint.opponent_path is None:
            raise SystemExit("td-search evaluation requires both value and opponent checkpoints.")
        command.extend(["--td-search-value-checkpoint", str(checkpoint.value_path)])
        command.extend(["--td-search-opponent-checkpoint", str(checkpoint.opponent_path)])
        command.extend(
            [
                "--td-search-opponent-temperature",
                str(args.eval_td_search_opponent_temperature),
            ]
        )
        if args.eval_td_search_sample_opponent_actions:
            command.append("--td-search-sample-opponent-actions")
    else:
        if checkpoint.value_path is None:
            raise SystemExit("td-value evaluation requires a value checkpoint.")
        command.extend(["--td-value-checkpoint", str(checkpoint.value_path)])
    return command


def _checkpoints_from_train_summary(payload: Dict[str, Any]) -> List[LoopCheckpoint]:
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


def _select_latest_checkpoint(
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


def _read_gate_outcome(*, gate_payload: Dict[str, Any], artifact: Path) -> Dict[str, Any]:
    status = str(gate_payload.get("status", "unknown"))
    decision = gate_payload.get("decision")
    if not isinstance(decision, dict):
        raise SystemExit(f"Gate artifact is missing decision payload: {artifact}")
    decision_state = str(decision.get("state", status))
    decision_reason = str(decision.get("reason", "unknown"))

    results = gate_payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Gate artifact is missing results payload: {artifact}")

    row = _row_from_results(path=artifact, opponent_policy="search", results=results)
    return {
        "artifact": str(artifact),
        "status": status,
        "decision": decision_state,
        "reason": decision_reason,
        "candidateWinRate": row.candidate_win_rate,
        "candidateWinRateCi95": {"low": row.ci_low, "high": row.ci_high},
        "sideGap": row.side_gap,
        "candidateWins": row.candidate_wins,
        "opponentWins": row.opponent_wins,
        "draws": row.draws,
        "totalGames": row.total_games,
    }


def _read_eval_row(*, path: Path, opponent_policy: str) -> EvalRow:
    payload = _read_json(path, label=f"eval artifact {path.name}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise SystemExit(f"Eval artifact is missing results payload: {path}")
    return _row_from_results(path=path, opponent_policy=opponent_policy, results=results)


def _row_from_results(*, path: Path, opponent_policy: str, results: Dict[str, Any]) -> EvalRow:
    return EvalRow(
        artifact=path,
        opponent_policy=opponent_policy,
        candidate_win_rate=float(results["candidateWinRate"]),
        ci_low=float(results["candidateWinRateCi95"]["low"]),
        ci_high=float(results["candidateWinRateCi95"]["high"]),
        side_gap=float(results["sideGap"]),
        candidate_wins=int(results["candidateWins"]),
        opponent_wins=int(results["opponentWins"]),
        draws=int(results["draws"]),
        total_games=int(results["totalGames"]),
    )


def _evaluate_certify_result(*, row: EvalRow, args: argparse.Namespace) -> CertifyComparison:
    checks = {
        "minWinRate": row.candidate_win_rate >= args.certify_min_win_rate,
        "maxSideGap": row.side_gap <= args.certify_max_side_gap,
        "minCiLow": row.ci_low >= args.certify_min_ci_low,
    }
    return CertifyComparison(
        row=row,
        checks=checks,
        passed=all(checks.values()),
    )


def _promotion_decision(
    *,
    gate_decision: str,
    gate_reason: str,
    certify: Dict[str, Any],
) -> Dict[str, Any]:
    if gate_decision != "accepted":
        return {
            "promoted": False,
            "reason": f"gate_{gate_decision}:{gate_reason}",
        }
    if not bool(certify.get("ran")):
        return {
            "promoted": False,
            "reason": "gate_accepted_but_certify_not_run",
        }
    if not bool(certify.get("overallPassed")):
        failed = [
            row["opponentPolicy"]
            for row in certify.get("comparisons", [])
            if not bool(row.get("passed"))
        ]
        return {
            "promoted": False,
            "reason": "certify_failed",
            "failedOpponents": failed,
        }
    return {
        "promoted": True,
        "reason": "gate_and_certify_passed",
    }


def _config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pythonBin": str(args.python_bin),
        "cloud": bool(args.cloud),
        "cloudVcpus": args.cloud_vcpus,
        "chunksPerGate": args.chunks_per_gate,
        "collect": {
            "gamesPerChunk": args.collect_games,
            "workers": args.collect_workers,
            "seedPrefix": args.collect_seed_prefix,
            "playerAPolicy": args.collect_player_a_policy,
            "playerBPolicy": args.collect_player_b_policy,
            "search": {
                "worlds": args.collect_search_worlds,
                "rollouts": args.collect_search_rollouts,
                "depth": args.collect_search_depth,
                "maxRootActions": args.collect_search_max_root_actions,
                "rolloutEpsilon": args.collect_search_rollout_epsilon,
            },
            "progressEveryGames": args.collect_progress_every_games,
        },
        "train": {
            "stepsPerChunk": args.train_steps,
            "valueBatchSize": args.train_value_batch_size,
            "opponentBatchSize": args.train_opponent_batch_size,
            "seed": args.train_seed,
            "hiddenDim": args.train_hidden_dim,
            "gamma": args.train_gamma,
            "valueLearningRate": args.train_value_learning_rate,
            "valueWeightDecay": args.train_value_weight_decay,
            "opponentLearningRate": args.train_opponent_learning_rate,
            "opponentWeightDecay": args.train_opponent_weight_decay,
            "maxGradNorm": args.train_max_grad_norm,
            "targetSyncInterval": args.train_target_sync_interval,
            "valueTargetMode": args.train_value_target_mode,
            "tdLambda": args.train_td_lambda,
            "saveEverySteps": args.train_save_every_steps,
            "progressEverySteps": args.train_progress_every_steps,
            "numThreads": args.train_num_threads,
            "numInteropThreads": args.train_num_interop_threads,
            "useMseLoss": bool(args.train_use_mse_loss),
            "disableValue": bool(args.train_disable_value),
            "disableOpponent": bool(args.train_disable_opponent),
        },
        "eval": {
            "candidatePolicy": args.eval_candidate_policy,
            "search": {
                "worlds": args.eval_search_worlds,
                "rollouts": args.eval_search_rollouts,
                "depth": args.eval_search_depth,
                "maxRootActions": args.eval_search_max_root_actions,
                "rolloutEpsilon": args.eval_search_rollout_epsilon,
            },
            "tdWorlds": args.eval_td_worlds,
            "tdSearchOpponentTemperature": args.eval_td_search_opponent_temperature,
            "tdSearchSampleOpponentActions": bool(args.eval_td_search_sample_opponent_actions),
        },
        "gate": {
            "workers": args.gate_workers,
            "seedPrefix": args.gate_seed_prefix,
            "seedStartIndex": args.gate_seed_start_index,
            "h0WinRate": args.gate_h0_win_rate,
            "h1WinRate": args.gate_h1_win_rate,
            "alpha": args.gate_alpha,
            "beta": args.gate_beta,
            "batchGamesPerSide": args.gate_batch_games_per_side,
            "maxGamesPerSide": args.gate_max_games_per_side,
            "maxSideGap": args.gate_max_side_gap,
        },
        "certify": {
            "workers": args.certify_workers,
            "gamesPerSide": args.certify_games_per_side,
            "opponents": list(args.certify_opponents),
            "seedPrefix": args.certify_seed_prefix,
            "seedStartIndex": args.certify_seed_start_index,
            "minWinRate": args.certify_min_win_rate,
            "maxSideGap": args.certify_max_side_gap,
            "minCiLow": args.certify_min_ci_low,
        },
    }


def _require_supported_runtime(python_bin: Path) -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.11+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")
    if not python_bin.exists():
        raise SystemExit(f"--python-bin does not exist: {python_bin}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.chunks_per_gate <= 0:
        raise SystemExit("--chunks-per-gate must be > 0.")
    if args.collect_games <= 0:
        raise SystemExit("--collect-games must be > 0.")
    if args.collect_workers <= 0:
        raise SystemExit("--collect-workers must be > 0.")
    if args.train_steps <= 0:
        raise SystemExit("--train-steps must be > 0.")
    if args.train_num_threads is not None and args.train_num_threads <= 0:
        raise SystemExit("--train-num-threads must be > 0 when provided.")
    if args.train_num_interop_threads is not None and args.train_num_interop_threads <= 0:
        raise SystemExit("--train-num-interop-threads must be > 0 when provided.")
    if args.train_td_lambda < 0.0 or args.train_td_lambda > 1.0:
        raise SystemExit("--train-td-lambda must be in [0, 1].")
    if args.gate_workers <= 0:
        raise SystemExit("--gate-workers must be > 0.")
    if args.gate_seed_start_index < 0:
        raise SystemExit("--gate-seed-start-index must be >= 0.")
    if args.gate_h0_win_rate < 0.0 or args.gate_h0_win_rate > 1.0:
        raise SystemExit("--gate-h0-win-rate must be in [0, 1].")
    if args.gate_h1_win_rate < 0.0 or args.gate_h1_win_rate > 1.0:
        raise SystemExit("--gate-h1-win-rate must be in [0, 1].")
    if args.gate_h0_win_rate >= args.gate_h1_win_rate:
        raise SystemExit("--gate-h0-win-rate must be strictly less than --gate-h1-win-rate.")
    if args.gate_alpha <= 0.0 or args.gate_alpha >= 1.0:
        raise SystemExit("--gate-alpha must be in (0, 1).")
    if args.gate_beta <= 0.0 or args.gate_beta >= 1.0:
        raise SystemExit("--gate-beta must be in (0, 1).")
    if args.gate_batch_games_per_side <= 0:
        raise SystemExit("--gate-batch-games-per-side must be > 0.")
    if args.gate_max_games_per_side <= 0:
        raise SystemExit("--gate-max-games-per-side must be > 0.")
    if args.gate_batch_games_per_side > args.gate_max_games_per_side:
        raise SystemExit("--gate-batch-games-per-side must be <= --gate-max-games-per-side.")
    if args.gate_max_side_gap < 0.0 or args.gate_max_side_gap > 1.0:
        raise SystemExit("--gate-max-side-gap must be in [0, 1].")

    if args.certify_workers <= 0:
        raise SystemExit("--certify-workers must be > 0.")
    if args.certify_games_per_side <= 0:
        raise SystemExit("--certify-games-per-side must be > 0.")
    if args.certify_seed_start_index < 0:
        raise SystemExit("--certify-seed-start-index must be >= 0.")
    if args.certify_min_win_rate < 0.0 or args.certify_min_win_rate > 1.0:
        raise SystemExit("--certify-min-win-rate must be in [0, 1].")
    if args.certify_max_side_gap < 0.0 or args.certify_max_side_gap > 1.0:
        raise SystemExit("--certify-max-side-gap must be in [0, 1].")
    if args.certify_min_ci_low < 0.0 or args.certify_min_ci_low > 1.0:
        raise SystemExit("--certify-min-ci-low must be in [0, 1].")
    if len(set(args.certify_opponents)) != len(args.certify_opponents):
        raise SystemExit("--certify-opponents must not contain duplicates.")

    if args.train_disable_value and args.train_disable_opponent:
        raise SystemExit("At least one of value/opponent training must be enabled.")


def _run_step(*, name: str, command: Sequence[str]) -> None:
    print(f"[td-loop] step {name}: {_join_command(command)}")
    started = time.perf_counter()
    completed = subprocess.run(command)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(
            f"[td-loop] failed step={name} returnCode={completed.returncode} elapsedMin={elapsed / 60.0:.1f}"
        )
    print(f"[td-loop] completed step={name} elapsedMin={elapsed / 60.0:.1f}")


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {label}: {path}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return compact.strip("-") or "run"


def _join_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _recommended_cloud_worker_count(vcpus: int) -> int:
    # Keep one quarter of cores free to reduce scheduler contention and bridge overhead.
    return max(1, (vcpus * 3) // 4)


def _apply_cloud_profile(args: argparse.Namespace) -> None:
    workers = _recommended_cloud_worker_count(args.cloud_vcpus)
    args.collect_workers = workers
    args.gate_workers = workers
    args.certify_workers = workers
    if args.train_num_threads is None:
        args.train_num_threads = args.cloud_vcpus
    if args.train_num_interop_threads is None:
        args.train_num_interop_threads = 1


if __name__ == "__main__":
    raise SystemExit(main())
