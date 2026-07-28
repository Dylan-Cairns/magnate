from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from trainer.td import REPLAY_KEY_MODE_RUN_QUALIFIED, sha256_file

from scripts import run_td_hard_extra_data_continuation as training_runner
from scripts.td_loop_common import join_command, read_json, run_step, write_progress

EXPERIMENT_ID = "td-hard-extra-data-continuation-v1"
DEFAULT_TRAINING_PLAN = Path(
    "artifacts/training_inputs/td-hard-extra-data-continuation-v1/resolved-plan.json"
)
DEFAULT_TRAINING_ROOT = Path("artifacts/td_checkpoints/td-hard-extra-data-continuation-v1")
DEFAULT_EVALUATION_ROOT = Path("artifacts/td_extra_data_evals/td-hard-extra-data-continuation-v1")
DEFAULT_DEVELOPMENT_PLAN = Path(
    "artifacts/training_inputs/td-hard-extra-data-continuation-v1/development-evaluation.plan.json"
)
CHECKPOINT_STEPS = tuple(range(1_000, 10_001, 1_000))
EVALUATION_THREADS = 4
VALUE_BATCH_SIZE = 4_096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume development-only checkpoint selection for the frozen "
            "Hard-teacher extra-data continuation experiment."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--training-plan", type=Path, default=DEFAULT_TRAINING_PLAN)
    parser.add_argument(
        "--development-plan",
        type=Path,
        default=DEFAULT_DEVELOPMENT_PLAN,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-minutes", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    python_bin = args.python_bin.resolve()
    training_plan_path = _resolve_path(repo_root, args.training_plan)
    development_plan_path = _resolve_path(repo_root, args.development_plan)
    training_root = _resolve_path(repo_root, DEFAULT_TRAINING_ROOT)
    evaluation_root = _resolve_path(repo_root, DEFAULT_EVALUATION_ROOT)
    development_root = evaluation_root / "development"
    launch_root = development_root / "launch"
    progress_root = launch_root / "progress"
    completion_path = launch_root / "development.complete.json"
    result_path = development_root / "development.result.json"
    _validate_runtime_args(
        repo_root=repo_root,
        python_bin=python_bin,
        heartbeat_minutes=args.heartbeat_minutes,
    )

    run_step(
        name="training-preflight",
        command=[
            str(python_bin),
            "-m",
            "scripts.run_td_hard_extra_data_continuation",
            "--repo-root",
            str(repo_root),
            "--python-bin",
            str(python_bin),
            "--heartbeat-minutes",
            "0",
            "--dry-run",
        ],
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_root / "training-preflight.json",
        log_prefix="[extra-data-dev]",
    )
    training_plan = read_json(training_plan_path, label="resolved training plan")
    training_commands = training_runner._validate_resolved_plan(
        plan=training_plan,
        repo_root=repo_root,
        python_bin=python_bin,
        checkpoint_root=training_root,
    )
    training_results = [
        training_runner._validate_completed_summary(
            row=row,
            plan=training_plan,
            checkpoint_root=training_root,
        )
        for row in training_commands
    ]
    if len(training_results) != 8:
        raise SystemExit("Development evaluation requires all eight completed training jobs.")
    _validate_selection_contract(training_plan)

    checkpoint_inventory = _checkpoint_inventory(training_commands)
    development_inputs = _development_inputs(training_plan)
    incumbent = _incumbent_pair(training_plan)
    selection_commands = [
        _evaluation_row(
            row_id=f"primary-treatment-step-{step:05d}",
            role="primary-treatment-selection",
            step=step,
            value_checkpoint=checkpoint_inventory[
                ("primary", "extra-data-treatment", "value", step)
            ],
            opponent_checkpoint=checkpoint_inventory[
                ("primary", "extra-data-treatment", "opponent", step)
            ],
            development_inputs=development_inputs,
            python_bin=python_bin,
            output=development_root / "selection" / f"primary-treatment-step-{step:05d}.json",
        )
        for step in CHECKPOINT_STEPS
    ]
    development_plan = {
        "schemaVersion": 1,
        "experimentId": EXPERIMENT_ID,
        "status": "development-only",
        "finalTestAccessAuthorized": False,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "trainingPlan": str(training_plan_path),
        "trainingPlanSha256": sha256_file(training_plan_path),
        "sourceManifestSha256": _require_str(training_plan, "sourceManifestSha256"),
        "implementationSha256": _require_str(
            _require_object(training_plan, "verification"),
            "implementationSha256",
        ),
        "evaluationImplementation": {
            "orchestrator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "evaluator": {
                "path": str((repo_root / "scripts" / "evaluate_td_replay_holdout.py").resolve()),
                "sha256": sha256_file(repo_root / "scripts" / "evaluate_td_replay_holdout.py"),
            },
        },
        "developmentInputs": development_inputs,
        "selectionRule": _require_object(training_plan, "postTraining").get("selection"),
        "selectionCommands": selection_commands,
        "selectedComparisonRoles": [
            "primary-treatment",
            "primary-control",
            "replication-treatment",
            "replication-control",
            "unchanged-incumbent",
        ],
        "sealedFinalTest": {
            "accessed": False,
            "reason": "candidate checkpoint selection is not yet frozen",
        },
    }
    write_progress(development_plan_path, development_plan)
    _print_preflight(
        training_plan_path=training_plan_path,
        development_plan_path=development_plan_path,
        evaluation_root=evaluation_root,
        dry_run=bool(args.dry_run),
    )

    selection_results: list[dict[str, Any]] = []
    for index, row in enumerate(selection_commands, start=1):
        selection_results.append(
            _run_or_validate_evaluation(
                row=row,
                index=index,
                count=len(selection_commands),
                stage="selection",
                dry_run=bool(args.dry_run),
                heartbeat_minutes=args.heartbeat_minutes,
                progress_root=progress_root,
            )
        )
    if args.dry_run:
        print(
            "[extra-data-dev] selected-step comparisons are deferred until all "
            "ten selection results exist.",
            flush=True,
        )
        print(
            "[extra-data-dev] DRY RUN COMPLETE; no evaluation was started.",
            flush=True,
        )
        return 0

    selection = _select_checkpoint_step(selection_results)
    selected_step = _require_int(selection, "selectedStep")
    selected_primary_treatment = next(
        row
        for row in selection_results
        if _require_int(_require_object(row, "provenance"), "checkpointStep") == selected_step
    )
    comparison_commands = [
        _evaluation_row(
            row_id=f"primary-control-step-{selected_step:05d}",
            role="primary-control",
            step=selected_step,
            value_checkpoint=checkpoint_inventory[
                ("primary", "continued-control", "value", selected_step)
            ],
            opponent_checkpoint=checkpoint_inventory[
                ("primary", "continued-control", "opponent", selected_step)
            ],
            development_inputs=development_inputs,
            python_bin=python_bin,
            output=development_root / "selected" / f"primary-control-step-{selected_step:05d}.json",
        ),
        _evaluation_row(
            row_id=f"replication-treatment-step-{selected_step:05d}",
            role="replication-treatment",
            step=selected_step,
            value_checkpoint=checkpoint_inventory[
                ("replication", "extra-data-treatment", "value", selected_step)
            ],
            opponent_checkpoint=checkpoint_inventory[
                ("replication", "extra-data-treatment", "opponent", selected_step)
            ],
            development_inputs=development_inputs,
            python_bin=python_bin,
            output=development_root
            / "selected"
            / f"replication-treatment-step-{selected_step:05d}.json",
        ),
        _evaluation_row(
            row_id=f"replication-control-step-{selected_step:05d}",
            role="replication-control",
            step=selected_step,
            value_checkpoint=checkpoint_inventory[
                ("replication", "continued-control", "value", selected_step)
            ],
            opponent_checkpoint=checkpoint_inventory[
                ("replication", "continued-control", "opponent", selected_step)
            ],
            development_inputs=development_inputs,
            python_bin=python_bin,
            output=development_root
            / "selected"
            / f"replication-control-step-{selected_step:05d}.json",
        ),
        _evaluation_row(
            row_id="unchanged-incumbent",
            role="unchanged-incumbent",
            step=_require_int(incumbent, "step"),
            value_checkpoint=_require_checkpoint(incumbent, "value"),
            opponent_checkpoint=_require_checkpoint(incumbent, "opponent"),
            development_inputs=development_inputs,
            python_bin=python_bin,
            output=development_root / "selected" / "unchanged-incumbent.json",
        ),
    ]
    selected_results: dict[str, dict[str, Any]] = {
        "primary-treatment": selected_primary_treatment,
    }
    for index, row in enumerate(comparison_commands, start=1):
        result = _run_or_validate_evaluation(
            row=row,
            index=index,
            count=len(comparison_commands),
            stage="selected",
            dry_run=False,
            heartbeat_minutes=args.heartbeat_minutes,
            progress_root=progress_root,
        )
        selected_results[_require_str(row, "role")] = result

    aggregate = _aggregate_result(
        training_plan=training_plan,
        development_plan_path=development_plan_path,
        selection=selection,
        selection_results=selection_results,
        selected_results=selected_results,
    )
    write_progress(result_path, aggregate)
    write_progress(
        completion_path,
        {
            "schemaVersion": 1,
            "experimentId": EXPERIMENT_ID,
            "status": "completed",
            "completedAtUtc": datetime.now(timezone.utc).isoformat(),
            "selectedStep": selected_step,
            "selectionEvaluations": len(selection_results),
            "selectedComparisons": len(selected_results),
            "result": str(result_path),
            "resultSha256": sha256_file(result_path),
            "finalTestAccessed": False,
        },
    )
    print(
        f"[extra-data-dev] DEVELOPMENT COMPLETE selectedStep={selected_step}",
        flush=True,
    )
    print(f"[extra-data-dev] result={result_path}", flush=True)
    print(f"[extra-data-dev] completionMarker={completion_path}", flush=True)
    return 0


def _checkpoint_inventory(
    training_commands: list[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    inventory: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in training_commands:
        seed_id = _require_str(row, "seedId")
        arm_id = _require_str(row, "armId")
        model = _require_str(row, "model")
        summary = read_json(training_runner._summary_path(row), label=f"{row['id']} summary")
        checkpoints = _require_object_list(_require_object(summary, "results"), "checkpoints")
        for entry in checkpoints:
            step = _require_int(entry, "step")
            path = Path(_require_str(entry, model)).resolve()
            if not path.exists() or not path.is_file():
                raise SystemExit(f"Checkpoint not found: {path}")
            inventory[(seed_id, arm_id, model, step)] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
    expected = 2 * 2 * 2 * len(CHECKPOINT_STEPS)
    if len(inventory) != expected:
        raise SystemExit(f"Expected {expected} frozen checkpoints, found {len(inventory)}.")
    return inventory


def _development_inputs(training_plan: dict[str, Any]) -> dict[str, Any]:
    verification = _require_object(training_plan, "verification")
    development = _require_object(verification, "development")
    path_lists = _require_object(training_plan, "pathLists")
    value_list = _require_object(path_lists, "developmentValue")
    opponent_list = _require_object(path_lists, "developmentOpponent")
    inputs = {
        "games": _require_int(development, "games"),
        "rows": _require_int(development, "rows"),
        "replayKeyMode": REPLAY_KEY_MODE_RUN_QUALIFIED,
        "valueReplayList": _require_str(value_list, "path"),
        "valueReplayListSha256": _require_str(value_list, "sha256"),
        "valueReplayContentSha256": _require_str(
            _require_object(development, "contentSha256"),
            "value",
        ),
        "opponentReplayList": _require_str(opponent_list, "path"),
        "opponentReplayListSha256": _require_str(opponent_list, "sha256"),
        "opponentReplayContentSha256": _require_str(
            _require_object(development, "contentSha256"),
            "opponent",
        ),
    }
    for key in ("valueReplayList", "opponentReplayList"):
        path = Path(_require_str(inputs, key)).resolve()
        if not path.exists() or not path.is_file():
            raise SystemExit(f"Development replay path list not found: {path}")
        _expect_equal(
            f"{key} SHA-256",
            sha256_file(path),
            _require_str(inputs, f"{key}Sha256"),
        )
    _expect_equal("development games", _require_int(inputs, "games"), 100)
    _expect_equal("development rows", _require_int(inputs, "rows"), 18_083)
    return inputs


def _incumbent_pair(training_plan: dict[str, Any]) -> dict[str, Any]:
    verification = _require_object(training_plan, "verification")
    value = _require_object(verification, "valueWarmStart")
    opponent = _require_object(verification, "opponentWarmStart")
    return {
        "step": 30_000,
        "value": {
            "path": _require_str(value, "path"),
            "sha256": _require_str(value, "sha256"),
        },
        "opponent": {
            "path": _require_str(opponent, "path"),
            "sha256": _require_str(opponent, "sha256"),
        },
    }


def _evaluation_row(
    *,
    row_id: str,
    role: str,
    step: int,
    value_checkpoint: dict[str, Any],
    opponent_checkpoint: dict[str, Any],
    development_inputs: dict[str, Any],
    python_bin: Path,
    output: Path,
) -> dict[str, Any]:
    command = [
        str(python_bin),
        "-m",
        "scripts.evaluate_td_replay_holdout",
        "--value-checkpoint",
        _require_str(value_checkpoint, "path"),
        "--opponent-checkpoint",
        _require_str(opponent_checkpoint, "path"),
        "--value-replay-list",
        _require_str(development_inputs, "valueReplayList"),
        "--opponent-replay-list",
        _require_str(development_inputs, "opponentReplayList"),
        "--expected-value-replay-content-sha256",
        _require_str(development_inputs, "valueReplayContentSha256"),
        "--expected-opponent-replay-content-sha256",
        _require_str(development_inputs, "opponentReplayContentSha256"),
        "--replay-key-mode",
        REPLAY_KEY_MODE_RUN_QUALIFIED,
        "--expected-value-checkpoint-sha256",
        _require_str(value_checkpoint, "sha256"),
        "--expected-opponent-checkpoint-sha256",
        _require_str(opponent_checkpoint, "sha256"),
        "--expected-checkpoint-step",
        str(step),
        "--gamma",
        "0.995",
        "--value-batch-size",
        str(VALUE_BATCH_SIZE),
        "--num-threads",
        str(EVALUATION_THREADS),
        "--output",
        str(output.resolve()),
    ]
    return {
        "id": row_id,
        "role": role,
        "checkpointStep": step,
        "output": str(output.resolve()),
        "command": command,
    }


def _run_or_validate_evaluation(
    *,
    row: dict[str, Any],
    index: int,
    count: int,
    stage: str,
    dry_run: bool,
    heartbeat_minutes: float,
    progress_root: Path,
) -> dict[str, Any]:
    row_id = _require_str(row, "id")
    output = Path(_require_str(row, "output")).resolve()
    if output.exists():
        result = _validate_evaluation_result(row=row)
        print(
            f"[extra-data-dev] SKIP {stage} {index}/{count}: {row_id} (validated result exists)",
            flush=True,
        )
        return result
    if dry_run:
        print(
            f"[extra-data-dev] WOULD RUN {stage} {index}/{count}: {row_id} | "
            f"{join_command(_command(row))}",
            flush=True,
        )
        return {}
    print(
        f"[extra-data-dev] START {stage} {index}/{count}: {row_id}",
        flush=True,
    )
    run_step(
        name=f"{stage}-{row_id}",
        command=_command(row),
        heartbeat_minutes=heartbeat_minutes,
        progress_path=progress_root / f"{stage}-{row_id}.json",
        log_prefix="[extra-data-dev]",
    )
    result = _validate_evaluation_result(row=row)
    print(
        f"[extra-data-dev] COMPLETE {stage} {index}/{count}: {row_id}",
        flush=True,
    )
    return result


def _validate_evaluation_result(*, row: dict[str, Any]) -> dict[str, Any]:
    row_id = _require_str(row, "id")
    output = Path(_require_str(row, "output")).resolve()
    result = read_json(output, label=f"{row_id} development result")
    _expect_equal(f"{row_id} schema version", _require_int(result, "schemaVersion"), 1)
    provenance = _require_object(result, "provenance")
    value = _require_object(result, "value")
    opponent = _require_object(result, "opponent")
    command = _command(row)
    _expect_equal(
        f"{row_id} checkpoint step",
        _require_int(provenance, "checkpointStep"),
        _require_int(row, "checkpointStep"),
    )
    for model in ("value", "opponent"):
        _expect_equal(
            f"{row_id} {model} checkpoint SHA-256",
            _require_str(provenance, f"{model}CheckpointSha256"),
            _flag_value(command, f"--expected-{model}-checkpoint-sha256"),
        )
        _expect_equal(
            f"{row_id} {model} replay SHA-256",
            _require_str(provenance, f"{model}ReplayContentSha256"),
            _flag_value(command, f"--expected-{model}-replay-content-sha256"),
        )
    _expect_equal(
        f"{row_id} replay key mode",
        _require_str(provenance, "replayKeyMode"),
        REPLAY_KEY_MODE_RUN_QUALIFIED,
    )
    _expect_equal(f"{row_id} value rows", _require_int(value, "rows"), 18_083)
    _expect_equal(f"{row_id} opponent rows", _require_int(opponent, "rows"), 18_083)
    for payload, metric in (
        (value, "monteCarloMse"),
        (value, "monteCarloMae"),
        (opponent, "softTargetCrossEntropy"),
        (opponent, "softTargetKl"),
        (opponent, "teacherTopActionAgreement"),
    ):
        metric_value = _require_number(payload, metric)
        if not math.isfinite(float(metric_value)):
            raise SystemExit(f"{row_id} metric {metric} is not finite.")
    return result


def _select_checkpoint_step(
    selection_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(selection_results) != len(CHECKPOINT_STEPS):
        raise SystemExit(
            f"Expected {len(CHECKPOINT_STEPS)} selection results, found {len(selection_results)}."
        )
    by_step: dict[int, dict[str, Any]] = {}
    for result in selection_results:
        provenance = _require_object(result, "provenance")
        step = _require_int(provenance, "checkpointStep")
        if step in by_step:
            raise SystemExit(f"Duplicate development result for step {step}.")
        by_step[step] = result
    _expect_equal("selection checkpoint steps", tuple(sorted(by_step)), CHECKPOINT_STEPS)

    value_order = sorted(
        CHECKPOINT_STEPS,
        key=lambda step: (
            float(_require_number(_require_object(by_step[step], "value"), "monteCarloMse")),
            step,
        ),
    )
    opponent_order = sorted(
        CHECKPOINT_STEPS,
        key=lambda step: (
            float(
                _require_number(
                    _require_object(by_step[step], "opponent"),
                    "softTargetCrossEntropy",
                )
            ),
            step,
        ),
    )
    value_rank = {step: index + 1 for index, step in enumerate(value_order)}
    opponent_rank = {step: index + 1 for index, step in enumerate(opponent_order)}
    ranking = []
    for step in CHECKPOINT_STEPS:
        value = _require_object(by_step[step], "value")
        opponent = _require_object(by_step[step], "opponent")
        ranking.append(
            {
                "step": step,
                "valueMonteCarloMse": _require_number(value, "monteCarloMse"),
                "valueMseRank": value_rank[step],
                "opponentSoftTargetCrossEntropy": _require_number(
                    opponent,
                    "softTargetCrossEntropy",
                ),
                "opponentCrossEntropyRank": opponent_rank[step],
                "rankSum": value_rank[step] + opponent_rank[step],
                "valueMonteCarloMae": _require_number(value, "monteCarloMae"),
                "opponentSoftTargetKl": _require_number(opponent, "softTargetKl"),
            }
        )
    ordered = sorted(
        ranking,
        key=lambda row: (
            _require_int(row, "rankSum"),
            float(_require_number(row, "valueMonteCarloMae")),
            float(_require_number(row, "opponentSoftTargetKl")),
            _require_int(row, "step"),
        ),
    )
    return {
        "rule": (
            "minimum ordinal rank sum of value.monteCarloMse and opponent.softTargetCrossEntropy"
        ),
        "tieBreakers": [
            "lower value.monteCarloMae",
            "lower opponent.softTargetKl",
            "earlier checkpoint step",
        ],
        "selectedStep": _require_int(ordered[0], "step"),
        "ranking": ordered,
    }


def _aggregate_result(
    *,
    training_plan: dict[str, Any],
    development_plan_path: Path,
    selection: dict[str, Any],
    selection_results: list[dict[str, Any]],
    selected_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics = {role: _metrics(result) for role, result in selected_results.items()}
    comparisons = {
        "primaryTreatmentVsControl": _compare_metrics(
            metrics["primary-treatment"],
            metrics["primary-control"],
        ),
        "replicationTreatmentVsControl": _compare_metrics(
            metrics["replication-treatment"],
            metrics["replication-control"],
        ),
        "primaryTreatmentVsIncumbent": _compare_metrics(
            metrics["primary-treatment"],
            metrics["unchanged-incumbent"],
        ),
        "replicationTreatmentVsIncumbent": _compare_metrics(
            metrics["replication-treatment"],
            metrics["unchanged-incumbent"],
        ),
    }
    return {
        "schemaVersion": 1,
        "experimentId": EXPERIMENT_ID,
        "status": "development-completed",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceManifestSha256": _require_str(training_plan, "sourceManifestSha256"),
        "developmentPlanSha256": sha256_file(development_plan_path),
        "finalTestAccessed": False,
        "selection": selection,
        "selectionMetrics": [
            {
                "step": _require_int(_require_object(result, "provenance"), "checkpointStep"),
                **_metrics(result),
            }
            for result in selection_results
        ],
        "selectedMetrics": metrics,
        "comparisons": comparisons,
    }


def _metrics(result: dict[str, Any]) -> dict[str, float]:
    value = _require_object(result, "value")
    opponent = _require_object(result, "opponent")
    return {
        "valueMonteCarloMse": float(_require_number(value, "monteCarloMse")),
        "valueMonteCarloMae": float(_require_number(value, "monteCarloMae")),
        "valueMeanPredictionBias": float(_require_number(value, "meanPredictionBias")),
        "opponentSoftTargetCrossEntropy": float(
            _require_number(opponent, "softTargetCrossEntropy")
        ),
        "opponentSoftTargetKl": float(_require_number(opponent, "softTargetKl")),
        "opponentTeacherTopActionAgreement": float(
            _require_number(opponent, "teacherTopActionAgreement")
        ),
        "opponentSelectedActionAccuracy": float(
            _require_number(opponent, "selectedActionAccuracy")
        ),
    }


def _compare_metrics(
    treatment: dict[str, float],
    reference: dict[str, float],
) -> dict[str, float]:
    treatment_mse = float(treatment["valueMonteCarloMse"])
    reference_mse = float(reference["valueMonteCarloMse"])
    return {
        "valueMseDelta": treatment_mse - reference_mse,
        "valueMseRatio": treatment_mse / reference_mse,
        "valueMaeDelta": (
            float(treatment["valueMonteCarloMae"]) - float(reference["valueMonteCarloMae"])
        ),
        "opponentCrossEntropyDelta": (
            float(treatment["opponentSoftTargetCrossEntropy"])
            - float(reference["opponentSoftTargetCrossEntropy"])
        ),
        "opponentKlDelta": (
            float(treatment["opponentSoftTargetKl"]) - float(reference["opponentSoftTargetKl"])
        ),
        "opponentTeacherTopAgreementDelta": (
            float(treatment["opponentTeacherTopActionAgreement"])
            - float(reference["opponentTeacherTopActionAgreement"])
        ),
    }


def _validate_selection_contract(training_plan: dict[str, Any]) -> None:
    post_training = _require_object(training_plan, "postTraining")
    selection = post_training.get("selection")
    if not isinstance(selection, dict):
        raise SystemExit("Resolved training plan is missing frozen selection rules.")
    _expect_equal(
        "selection checkpoint steps",
        tuple(_require_int_list(selection, "checkpointSteps")),
        CHECKPOINT_STEPS,
    )
    _expect_equal(
        "selection data",
        _require_str(selection, "selectionData"),
        "imported development split only",
    )
    _expect_equal(
        "selection models",
        _require_str(selection, "selectionModels"),
        "primary extra-data-treatment value and opponent pair",
    )
    if selection.get("matchedComparisonsUseSameStep") is not True:
        raise SystemExit("Matched comparisons must use the selected treatment step.")
    if selection.get("candidateMustBeFrozenBeforeFinalTest") is not True:
        raise SystemExit("Candidate must remain frozen before final-test access.")


def _print_preflight(
    *,
    training_plan_path: Path,
    development_plan_path: Path,
    evaluation_root: Path,
    dry_run: bool,
) -> None:
    print(f"[extra-data-dev] trainingPlan={training_plan_path}", flush=True)
    print(f"[extra-data-dev] developmentPlan={development_plan_path}", flush=True)
    print(f"[extra-data-dev] evaluationRoot={evaluation_root}", flush=True)
    print("[extra-data-dev] selectionCheckpoints=10 threads=4", flush=True)
    print("[extra-data-dev] finalTestAccess=false", flush=True)
    if dry_run:
        print(
            "[extra-data-dev] dry run only; no evaluation will be started.",
            flush=True,
        )


def _validate_runtime_args(
    *,
    repo_root: Path,
    python_bin: Path,
    heartbeat_minutes: float,
) -> None:
    if not repo_root.exists() or not repo_root.is_dir():
        raise SystemExit(f"Repository root not found: {repo_root}")
    if not python_bin.exists() or not python_bin.is_file():
        raise SystemExit(f"Python runtime not found: {python_bin}")
    if heartbeat_minutes < 0.0:
        raise SystemExit("--heartbeat-minutes must be >= 0.")


def _require_checkpoint(payload: dict[str, Any], key: str) -> dict[str, Any]:
    checkpoint = _require_object(payload, key)
    path = Path(_require_str(checkpoint, "path")).resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Checkpoint not found: {path}")
    _expect_equal(
        f"{key} checkpoint SHA-256",
        sha256_file(path),
        _require_str(checkpoint, "sha256"),
    )
    return checkpoint


def _command(row: dict[str, Any]) -> list[str]:
    value = row.get("command")
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Expected non-empty command list for {row.get('id')!r}.")
    if not all(isinstance(item, str) and item for item in value):
        raise SystemExit("Prepared command entries must be non-empty strings.")
    return [str(item) for item in value]


def _flag_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise SystemExit(f"Prepared command must contain one value for {flag}.")
    return str(command[indices[0] + 1])


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected object at {key}.")
    return value


def _require_object_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Expected non-empty object list at {key}.")
    if not all(isinstance(item, dict) for item in value):
        raise SystemExit(f"Expected every {key} entry to be an object.")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Expected non-empty string at {key}.")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"Expected integer at {key}.")
    return value


def _require_number(payload: dict[str, Any], key: str) -> int | float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"Expected number at {key}.")
    return value


def _require_int_list(payload: dict[str, Any], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Expected non-empty integer list at {key}.")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise SystemExit(f"Expected every {key} entry to be an integer.")
    return [int(item) for item in value]


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected={expected!r} actual={actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
