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
from typing import Dict, List


@dataclass(frozen=True)
class Step:
    name: str
    command: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full guidance A/B pipeline unattended: "
            "teacher-data generation -> guidance training -> baseline eval -> guided eval."
        )
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(".venv/Scripts/python.exe"),
        help="Python executable used for all subprocess steps.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Games for teacher-data generation and each A/B eval.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="guidance-pilot",
        help="Run label used in artifact names and seed prefixes.",
    )
    parser.add_argument(
        "--teacher-policy",
        type=str,
        default="search",
        help="Teacher policy for data generation (random|heuristic|search|mcts|bc|ppo).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint when teacher policy is bc or ppo.",
    )
    parser.add_argument(
        "--teacher-players",
        type=str,
        choices=("both", "player-a", "player-b"),
        default="both",
        help="Which players use teacher policy during data generation.",
    )
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="heuristic",
        help="Opponent policy when teacher-players is not both.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint when opponent policy is bc or ppo.",
    )

    parser.add_argument("--search-worlds", type=int, default=6)
    parser.add_argument("--search-rollouts", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=14)
    parser.add_argument("--search-max-root-actions", type=int, default=6)
    parser.add_argument("--search-rollout-epsilon", type=float, default=0.08)

    parser.add_argument("--mcts-worlds", type=int, default=6)
    parser.add_argument("--mcts-simulations", type=int, default=192)
    parser.add_argument("--mcts-depth", type=int, default=20)
    parser.add_argument("--mcts-max-root-actions", type=int, default=10)
    parser.add_argument("--mcts-c-puct", type=float, default=1.15)

    parser.add_argument("--guidance-epochs", type=int, default=12)
    parser.add_argument("--guidance-batch-size", type=int, default=128)
    parser.add_argument("--guidance-learning-rate", type=float, default=3e-4)
    parser.add_argument("--guidance-weight-decay", type=float, default=1e-5)
    parser.add_argument("--guidance-value-loss-coef", type=float, default=0.5)
    parser.add_argument("--guidance-entropy-coef", type=float, default=0.0)
    parser.add_argument("--guidance-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--guidance-hidden-dim", type=int, default=256)
    parser.add_argument("--guidance-seed", type=int, default=0)
    parser.add_argument(
        "--guidance-temperature",
        type=float,
        default=1.0,
        help="Temperature used when injecting guidance checkpoint into search/MCTS.",
    )

    parser.add_argument(
        "--eval-policy-a",
        type=str,
        default="search",
        help="Policy A used for baseline and guided evals.",
    )
    parser.add_argument(
        "--eval-policy-b",
        type=str,
        default="heuristic",
        help="Policy B used for baseline and guided evals.",
    )
    parser.add_argument(
        "--eval-a-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for eval policy A when bc/ppo.",
    )
    parser.add_argument(
        "--eval-b-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for eval policy B when bc/ppo.",
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root artifact directory.",
    )
    parser.add_argument(
        "--run-manifest-out",
        type=Path,
        default=None,
        help="Optional explicit output path for pipeline manifest JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    safe_label = _slug(args.run_label)
    run_id = f"{stamp}-{safe_label}"

    artifact_root = args.artifact_dir
    teacher_dir = artifact_root / "teacher_data"
    eval_dir = artifact_root / "evals"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    teacher_samples = teacher_dir / f"{run_id}-teacher.jsonl"
    teacher_summary = teacher_dir / f"{run_id}-teacher.summary.json"
    guidance_checkpoint = artifact_root / f"search_guidance_{run_id}.pt"
    baseline_eval_out = eval_dir / f"{run_id}-baseline.json"
    guided_eval_out = eval_dir / f"{run_id}-guided.json"
    manifest_path = args.run_manifest_out or (artifact_root / f"{run_id}-pipeline-manifest.json")

    python_bin = str(args.python_bin)
    teacher_seed = f"{safe_label}-teacher"
    base_seed = f"{safe_label}-base"
    guided_seed = f"{safe_label}-guided"

    steps = _build_steps(
        python_bin=python_bin,
        args=args,
        teacher_seed=teacher_seed,
        teacher_samples=teacher_samples,
        teacher_summary=teacher_summary,
        guidance_checkpoint=guidance_checkpoint,
        base_seed=base_seed,
        baseline_eval_out=baseline_eval_out,
        guided_seed=guided_seed,
        guided_eval_out=guided_eval_out,
    )

    step_results: List[Dict[str, object]] = []
    for index, step in enumerate(steps, start=1):
        cmd_str = " ".join(shlex.quote(part) for part in step.command)
        print(f"[pipeline] step {index}/{len(steps)} {step.name}: {cmd_str}", file=sys.stderr, flush=True)
        step_start = time.perf_counter()
        completed = subprocess.run(step.command)
        elapsed_seconds = time.perf_counter() - step_start
        result = {
            "name": step.name,
            "command": step.command,
            "exitCode": completed.returncode,
            "elapsedSeconds": round(elapsed_seconds, 3),
        }
        step_results.append(result)
        if completed.returncode != 0:
            payload = _manifest_payload(
                run_id=run_id,
                args=args,
                started_at_seconds=started_at,
                steps=step_results,
                artifacts={
                    "teacherSamples": str(teacher_samples),
                    "teacherSummary": str(teacher_summary),
                    "guidanceCheckpoint": str(guidance_checkpoint),
                    "baselineEval": str(baseline_eval_out),
                    "guidedEval": str(guided_eval_out),
                },
                status="failed",
            )
            _write_json(manifest_path, payload)
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "failedStep": step.name,
                        "manifest": str(manifest_path),
                    },
                    indent=2,
                )
            )
            return completed.returncode

    payload = _manifest_payload(
        run_id=run_id,
        args=args,
        started_at_seconds=started_at,
        steps=step_results,
        artifacts={
            "teacherSamples": str(teacher_samples),
            "teacherSummary": str(teacher_summary),
            "guidanceCheckpoint": str(guidance_checkpoint),
            "baselineEval": str(baseline_eval_out),
            "guidedEval": str(guided_eval_out),
        },
        status="ok",
    )
    _write_json(manifest_path, payload)

    print(
        json.dumps(
            {
                "status": "ok",
                "runId": run_id,
                "manifest": str(manifest_path),
                "artifacts": payload["artifacts"],
            },
            indent=2,
        )
    )
    return 0


def _build_steps(
    *,
    python_bin: str,
    args: argparse.Namespace,
    teacher_seed: str,
    teacher_samples: Path,
    teacher_summary: Path,
    guidance_checkpoint: Path,
    base_seed: str,
    baseline_eval_out: Path,
    guided_seed: str,
    guided_eval_out: Path,
) -> List[Step]:
    teacher_cmd = [
        python_bin,
        "-m",
        "scripts.generate_teacher_data",
        "--games",
        str(args.games),
        "--seed-prefix",
        teacher_seed,
        "--teacher-policy",
        args.teacher_policy,
        "--teacher-players",
        args.teacher_players,
        "--opponent-policy",
        args.opponent_policy,
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
        "--mcts-worlds",
        str(args.mcts_worlds),
        "--mcts-simulations",
        str(args.mcts_simulations),
        "--mcts-depth",
        str(args.mcts_depth),
        "--mcts-max-root-actions",
        str(args.mcts_max_root_actions),
        "--mcts-c-puct",
        str(args.mcts_c_puct),
        "--out",
        str(teacher_samples),
        "--summary-out",
        str(teacher_summary),
    ]
    if args.teacher_checkpoint is not None:
        teacher_cmd.extend(["--teacher-checkpoint", str(args.teacher_checkpoint)])
    if args.opponent_checkpoint is not None:
        teacher_cmd.extend(["--opponent-checkpoint", str(args.opponent_checkpoint)])

    guidance_cmd = [
        python_bin,
        "-m",
        "scripts.train_search_guidance",
        "--samples-in",
        str(teacher_samples),
        "--checkpoint-out",
        str(guidance_checkpoint),
        "--epochs",
        str(args.guidance_epochs),
        "--batch-size",
        str(args.guidance_batch_size),
        "--learning-rate",
        str(args.guidance_learning_rate),
        "--weight-decay",
        str(args.guidance_weight_decay),
        "--value-loss-coef",
        str(args.guidance_value_loss_coef),
        "--entropy-coef",
        str(args.guidance_entropy_coef),
        "--max-grad-norm",
        str(args.guidance_max_grad_norm),
        "--hidden-dim",
        str(args.guidance_hidden_dim),
        "--seed",
        str(args.guidance_seed),
    ]

    baseline_cmd = [
        python_bin,
        "-m",
        "scripts.eval",
        "--games",
        str(args.games),
        "--seed-prefix",
        base_seed,
        "--player-a-policy",
        args.eval_policy_a,
        "--player-b-policy",
        args.eval_policy_b,
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
        "--mcts-worlds",
        str(args.mcts_worlds),
        "--mcts-simulations",
        str(args.mcts_simulations),
        "--mcts-depth",
        str(args.mcts_depth),
        "--mcts-max-root-actions",
        str(args.mcts_max_root_actions),
        "--mcts-c-puct",
        str(args.mcts_c_puct),
        "--out",
        str(baseline_eval_out),
    ]
    if args.eval_a_checkpoint is not None:
        baseline_cmd.extend(["--player-a-checkpoint", str(args.eval_a_checkpoint)])
    if args.eval_b_checkpoint is not None:
        baseline_cmd.extend(["--player-b-checkpoint", str(args.eval_b_checkpoint)])

    guided_cmd = [
        python_bin,
        "-m",
        "scripts.eval",
        "--games",
        str(args.games),
        "--seed-prefix",
        guided_seed,
        "--player-a-policy",
        args.eval_policy_a,
        "--player-b-policy",
        args.eval_policy_b,
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
        "--mcts-worlds",
        str(args.mcts_worlds),
        "--mcts-simulations",
        str(args.mcts_simulations),
        "--mcts-depth",
        str(args.mcts_depth),
        "--mcts-max-root-actions",
        str(args.mcts_max_root_actions),
        "--mcts-c-puct",
        str(args.mcts_c_puct),
        "--search-guidance-checkpoint",
        str(guidance_checkpoint),
        "--mcts-guidance-checkpoint",
        str(guidance_checkpoint),
        "--guidance-temperature",
        str(args.guidance_temperature),
        "--out",
        str(guided_eval_out),
    ]
    if args.eval_a_checkpoint is not None:
        guided_cmd.extend(["--player-a-checkpoint", str(args.eval_a_checkpoint)])
    if args.eval_b_checkpoint is not None:
        guided_cmd.extend(["--player-b-checkpoint", str(args.eval_b_checkpoint)])

    return [
        Step(name="generate_teacher_data", command=teacher_cmd),
        Step(name="train_search_guidance", command=guidance_cmd),
        Step(name="baseline_eval", command=baseline_cmd),
        Step(name="guided_eval", command=guided_cmd),
    ]


def _manifest_payload(
    *,
    run_id: str,
    args: argparse.Namespace,
    started_at_seconds: float,
    steps: List[Dict[str, object]],
    artifacts: Dict[str, str],
    status: str,
) -> Dict[str, object]:
    elapsed_seconds = time.perf_counter() - started_at_seconds
    return {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runId": run_id,
        "status": status,
        "elapsedSeconds": round(elapsed_seconds, 3),
        "config": {
            "pythonBin": str(args.python_bin),
            "games": args.games,
            "runLabel": args.run_label,
            "teacherPolicy": args.teacher_policy,
            "teacherCheckpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            "teacherPlayers": args.teacher_players,
            "opponentPolicy": args.opponent_policy,
            "opponentCheckpoint": str(args.opponent_checkpoint) if args.opponent_checkpoint else None,
            "search": {
                "worlds": args.search_worlds,
                "rollouts": args.search_rollouts,
                "depth": args.search_depth,
                "maxRootActions": args.search_max_root_actions,
                "rolloutEpsilon": args.search_rollout_epsilon,
            },
            "mcts": {
                "worlds": args.mcts_worlds,
                "simulations": args.mcts_simulations,
                "depth": args.mcts_depth,
                "maxRootActions": args.mcts_max_root_actions,
                "cPuct": args.mcts_c_puct,
            },
            "guidanceTrain": {
                "epochs": args.guidance_epochs,
                "batchSize": args.guidance_batch_size,
                "learningRate": args.guidance_learning_rate,
                "weightDecay": args.guidance_weight_decay,
                "valueLossCoef": args.guidance_value_loss_coef,
                "entropyCoef": args.guidance_entropy_coef,
                "maxGradNorm": args.guidance_max_grad_norm,
                "hiddenDim": args.guidance_hidden_dim,
                "seed": args.guidance_seed,
            },
            "guidanceEval": {
                "temperature": args.guidance_temperature,
            },
            "evalPolicies": {
                "playerA": args.eval_policy_a,
                "playerB": args.eval_policy_b,
                "playerACheckpoint": str(args.eval_a_checkpoint) if args.eval_a_checkpoint else None,
                "playerBCheckpoint": str(args.eval_b_checkpoint) if args.eval_b_checkpoint else None,
            },
        },
        "steps": steps,
        "artifacts": artifacts,
    }


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    safe = safe.strip("-")
    return safe or "run"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
