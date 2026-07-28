from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from trainer.td import REPLAY_KEY_MODE_RUN_QUALIFIED, sha256_file

from scripts.td_loop_common import join_command, read_json, run_step, write_progress

EXPERIMENT_ID = "td-hard-extra-data-continuation-v1"
DEFAULT_RESOLVED_PLAN = Path(
    "artifacts/training_inputs/td-hard-extra-data-continuation-v1/resolved-plan.json"
)
DEFAULT_CHECKPOINT_ROOT = Path("artifacts/td_checkpoints/td-hard-extra-data-continuation-v1")
EXPECTED_STEPS = 10_000
EXPECTED_CHECKPOINT_STEPS = tuple(range(1_000, EXPECTED_STEPS + 1, 1_000))
EXPECTED_THREADS = 4
EXPECTED_INTEROP_THREADS = 1
EXPECTED_COMMAND_IDS = tuple(
    f"{seed_id}-{arm_id}-{model}"
    for seed_id in ("primary", "replication")
    for arm_id in ("continued-control", "extra-data-treatment")
    for model in ("value", "opponent")
)
EXPECTED_SEEDS = {
    "primary": 2026072801,
    "replication": 2026072901,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the frozen Hard-teacher extra-data continuation experiment sequentially."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--resolved-plan", type=Path, default=DEFAULT_RESOLVED_PLAN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--heartbeat-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    python_bin = args.python_bin.resolve()
    resolved_plan_path = _resolve_path(repo_root, args.resolved_plan)
    checkpoint_root = _resolve_path(repo_root, DEFAULT_CHECKPOINT_ROOT)
    launch_root = checkpoint_root / "launch"
    progress_root = launch_root / "progress"
    overall_progress_path = launch_root / "training.progress.json"
    completion_path = launch_root / "training.complete.json"
    _validate_runtime_args(
        repo_root=repo_root,
        python_bin=python_bin,
        heartbeat_minutes=args.heartbeat_minutes,
    )

    run_step(
        name="preflight",
        command=[
            str(python_bin),
            "-m",
            "scripts.prepare_td_hard_extra_data_continuation",
            "--repo-root",
            str(repo_root),
        ],
        heartbeat_minutes=args.heartbeat_minutes,
        progress_path=progress_root / "preflight.json",
        log_prefix="[extra-data]",
    )
    plan = read_json(resolved_plan_path, label="resolved extra-data continuation plan")
    commands = _validate_resolved_plan(
        plan=plan,
        repo_root=repo_root,
        python_bin=python_bin,
        checkpoint_root=checkpoint_root,
    )

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in commands:
        summary_path = _summary_path(row)
        if summary_path.exists():
            completed.append(
                _validate_completed_summary(
                    row=row,
                    plan=plan,
                    checkpoint_root=checkpoint_root,
                )
            )
        else:
            pending.append(row)

    _validate_existing_completion_marker(
        completion_path=completion_path,
        plan=plan,
        completed=completed,
        pending=pending,
    )
    _print_preflight(
        resolved_plan_path=resolved_plan_path,
        launch_root=launch_root,
        completed=completed,
        pending=pending,
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        for index, row in enumerate(commands, start=1):
            if any(result["id"] == row["id"] for result in completed):
                print(
                    f"[extra-data] SKIP {index}/8: {row['id']} "
                    "(validated 10,000-update summary exists)",
                    flush=True,
                )
            else:
                print(
                    f"[extra-data] WOULD RUN {index}/8: {row['id']} | "
                    f"{join_command(_command(row))}",
                    flush=True,
                )
        print("[extra-data] DRY RUN COMPLETE; no training was started.", flush=True)
        return 0

    if not pending:
        if not completion_path.exists():
            write_progress(
                completion_path,
                _completion_payload(plan=plan, completed=completed),
            )
        print("[extra-data] ALL EIGHT TRAINING JOBS ARE ALREADY COMPLETE", flush=True)
        print(f"[extra-data] completionMarker={completion_path}", flush=True)
        return 0

    write_progress(
        overall_progress_path,
        _overall_progress_payload(
            status="running",
            completed=completed,
            pending=pending,
            active=None,
        ),
    )
    for index, row in enumerate(commands, start=1):
        row_id = _require_str(row, "id")
        if any(result["id"] == row_id for result in completed):
            print(
                f"[extra-data] SKIP {index}/8: {row_id} (validated 10,000-update summary exists)",
                flush=True,
            )
            continue

        print(
            f"[extra-data] START {index}/8: {row_id} at {datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )
        write_progress(
            overall_progress_path,
            _overall_progress_payload(
                status="running",
                completed=completed,
                pending=[
                    candidate
                    for candidate in commands
                    if _require_str(candidate, "id") not in {result["id"] for result in completed}
                ],
                active=row_id,
            ),
        )
        run_step(
            name=row_id,
            command=_command(row),
            heartbeat_minutes=args.heartbeat_minutes,
            progress_path=progress_root / f"{row_id}.json",
            log_prefix="[extra-data]",
        )
        result = _validate_completed_summary(
            row=row,
            plan=plan,
            checkpoint_root=checkpoint_root,
        )
        completed.append(result)
        print(
            f"[extra-data] COMPLETE {index}/8: {row_id} at "
            f"{datetime.now(timezone.utc).isoformat()}",
            flush=True,
        )

    if len(completed) != len(commands):
        raise SystemExit(
            f"Training ended with {len(completed)}/{len(commands)} validated summaries."
        )
    write_progress(
        overall_progress_path,
        _overall_progress_payload(
            status="completed",
            completed=completed,
            pending=[],
            active=None,
        ),
    )
    write_progress(
        completion_path,
        _completion_payload(plan=plan, completed=completed),
    )
    print("", flush=True)
    print("[extra-data] ALL EIGHT TRAINING JOBS ARE COMPLETE", flush=True)
    print(f"[extra-data] completionMarker={completion_path}", flush=True)
    return 0


def _validate_resolved_plan(
    *,
    plan: dict[str, Any],
    repo_root: Path,
    python_bin: Path,
    checkpoint_root: Path,
) -> list[dict[str, Any]]:
    _expect_equal("plan schema version", _require_int(plan, "schemaVersion"), 1)
    _expect_equal("plan experiment ID", _require_str(plan, "experimentId"), EXPERIMENT_ID)
    _expect_equal("plan status", _require_str(plan, "status"), "review-required")
    if plan.get("launchAuthorized") is not False:
        raise SystemExit("Resolved plan must retain launchAuthorized=false.")

    source_manifest = Path(_require_str(plan, "sourceManifest")).resolve()
    if not source_manifest.exists() or not source_manifest.is_file():
        raise SystemExit(f"Resolved source manifest not found: {source_manifest}")
    _expect_equal(
        "source manifest SHA-256",
        sha256_file(source_manifest),
        _require_str(plan, "sourceManifestSha256"),
    )
    verification = _require_object(plan, "verification")
    implementation_sha256 = _require_str(verification, "implementationSha256")
    value_warm = _require_object(verification, "valueWarmStart")
    opponent_warm = _require_object(verification, "opponentWarmStart")
    expected_warm_paths = {
        "value": str(Path(_require_str(value_warm, "path")).resolve()),
        "opponent": str(Path(_require_str(opponent_warm, "path")).resolve()),
    }
    expected_warm_hashes = {
        "value": _require_str(value_warm, "sha256"),
        "opponent": _require_str(opponent_warm, "sha256"),
    }
    path_lists = _require_object(plan, "pathLists")
    expected_lists = {
        ("continued-control", "value"): _path_list_path(path_lists, "controlValue"),
        ("continued-control", "opponent"): _path_list_path(path_lists, "controlOpponent"),
        ("extra-data-treatment", "value"): _path_list_path(path_lists, "treatmentValue"),
        ("extra-data-treatment", "opponent"): _path_list_path(path_lists, "treatmentOpponent"),
    }
    expected_replay_hashes = {
        ("continued-control", "value"): _content_sha(verification, "originalControl", "value"),
        ("continued-control", "opponent"): _content_sha(
            verification,
            "originalControl",
            "opponent",
        ),
        ("extra-data-treatment", "value"): _content_sha(
            verification,
            "treatmentCombined",
            "value",
        ),
        ("extra-data-treatment", "opponent"): _content_sha(
            verification,
            "treatmentCombined",
            "opponent",
        ),
    }

    commands = _require_object_list(plan, "trainingCommands")
    command_ids = tuple(_require_str(row, "id") for row in commands)
    _expect_equal("training command order", command_ids, EXPECTED_COMMAND_IDS)
    if len(set(command_ids)) != len(command_ids):
        raise SystemExit("Resolved training command IDs are not unique.")

    for row in commands:
        row_id = _require_str(row, "id")
        seed_id = _require_str(row, "seedId")
        arm_id = _require_str(row, "armId")
        model = _require_str(row, "model")
        command = _command(row)
        _expect_equal(f"{row_id} Python runtime", str(Path(command[0]).resolve()), str(python_bin))
        _expect_equal(f"{row_id} module switch", command[1:3], ["-m", "scripts.train_td"])
        _expect_equal(f"{row_id} steps", _flag_value(command, "--steps"), str(EXPECTED_STEPS))
        _expect_equal(
            f"{row_id} seed",
            _flag_value(command, "--seed"),
            str(EXPECTED_SEEDS[seed_id]),
        )
        _expect_equal(
            f"{row_id} replay key mode",
            _flag_value(command, "--replay-key-mode"),
            REPLAY_KEY_MODE_RUN_QUALIFIED,
        )
        _expect_equal(
            f"{row_id} augmentation",
            _flag_value(command, "--district-augmentation"),
            "none",
        )
        _expect_equal(f"{row_id} threads", _flag_value(command, "--num-threads"), "4")
        _expect_equal(
            f"{row_id} interop threads",
            _flag_value(command, "--num-interop-threads"),
            "1",
        )
        _expect_equal(
            f"{row_id} save interval",
            _flag_value(command, "--save-every-steps"),
            "1000",
        )
        _expect_equal(
            f"{row_id} progress interval",
            _flag_value(command, "--progress-every-steps"),
            "50",
        )
        _expect_equal(
            f"{row_id} source manifest",
            str(Path(_flag_value(command, "--experiment-manifest")).resolve()),
            str(source_manifest),
        )
        _expect_equal(
            f"{row_id} source manifest SHA-256",
            _flag_value(command, "--expected-experiment-manifest-sha256"),
            _require_str(plan, "sourceManifestSha256"),
        )
        _expect_equal(
            f"{row_id} implementation SHA-256",
            _flag_value(command, "--expected-implementation-sha256"),
            implementation_sha256,
        )
        _expect_equal(
            f"{row_id} provenance repo root",
            str(Path(_flag_value(command, "--provenance-repo-root")).resolve()),
            str(repo_root),
        )
        expected_summary = checkpoint_root / seed_id / arm_id / f"{model}.summary.json"
        _expect_equal(
            f"{row_id} summary path",
            str(Path(_flag_value(command, "--summary-out")).resolve()),
            str(expected_summary.resolve()),
        )
        _expect_equal(
            f"{row_id} output path",
            str(Path(_flag_value(command, "--out-dir")).resolve()),
            str((checkpoint_root / seed_id / arm_id / model).resolve()),
        )

        replay_list_flag = f"--{model}-replay-list"
        replay_sha_flag = f"--expected-{model}-replay-content-sha256"
        _expect_equal(
            f"{row_id} replay path list",
            str(Path(_flag_value(command, replay_list_flag)).resolve()),
            str(expected_lists[(arm_id, model)]),
        )
        _expect_equal(
            f"{row_id} replay SHA-256",
            _flag_value(command, replay_sha_flag),
            expected_replay_hashes[(arm_id, model)],
        )
        warm_flag = f"--warm-start-{model}-checkpoint"
        warm_sha_flag = f"--expected-warm-start-{model}-sha256"
        _expect_equal(
            f"{row_id} warm-start path",
            str(Path(_flag_value(command, warm_flag)).resolve()),
            expected_warm_paths[model],
        )
        _expect_equal(
            f"{row_id} warm-start SHA-256",
            _flag_value(command, warm_sha_flag),
            expected_warm_hashes[model],
        )
        required_disable = "--disable-opponent" if model == "value" else "--disable-value"
        forbidden_disable = "--disable-value" if model == "value" else "--disable-opponent"
        if required_disable not in command or forbidden_disable in command:
            raise SystemExit(f"{row_id} does not isolate {model} training.")

    return commands


def _validate_completed_summary(
    *,
    row: dict[str, Any],
    plan: dict[str, Any],
    checkpoint_root: Path,
) -> dict[str, Any]:
    row_id = _require_str(row, "id")
    seed_id = _require_str(row, "seedId")
    arm_id = _require_str(row, "armId")
    model = _require_str(row, "model")
    command = _command(row)
    summary_path = _summary_path(row)
    summary = read_json(summary_path, label=f"{row_id} training summary")
    config = _require_object(summary, "config")
    provenance = _require_object(summary, "provenance")
    results = _require_object(summary, "results")
    verification = _require_object(plan, "verification")

    _expect_equal(f"{row_id} summary steps", _require_int(config, "steps"), EXPECTED_STEPS)
    _expect_equal(
        f"{row_id} summary seed",
        _require_int(config, "seed"),
        EXPECTED_SEEDS[seed_id],
    )
    _expect_equal(
        f"{row_id} summary augmentation",
        _require_str(config, "districtAugmentation"),
        "none",
    )
    _expect_equal(
        f"{row_id} summary replay key mode",
        _require_str(config, "replayKeyMode"),
        REPLAY_KEY_MODE_RUN_QUALIFIED,
    )
    _expect_equal(
        f"{row_id} summary threads",
        _require_int(config, "numThreads"),
        EXPECTED_THREADS,
    )
    _expect_equal(
        f"{row_id} summary interop threads",
        _require_int(config, "numInteropThreads"),
        EXPECTED_INTEROP_THREADS,
    )
    _expect_equal(
        f"{row_id} summary save interval",
        _require_int(config, "saveEverySteps"),
        1_000,
    )
    _expect_equal(
        f"{row_id} summary progress interval",
        _require_int(config, "progressEverySteps"),
        50,
    )
    _expect_equal(
        f"{row_id} provenance replay key mode",
        _require_str(provenance, "replayKeyMode"),
        REPLAY_KEY_MODE_RUN_QUALIFIED,
    )
    _expect_equal(
        f"{row_id} provenance source manifest",
        _require_str(provenance, "experimentManifestSha256"),
        _require_str(plan, "sourceManifestSha256"),
    )
    _expect_equal(
        f"{row_id} provenance implementation",
        _require_str(provenance, "implementationSha256"),
        _require_str(verification, "implementationSha256"),
    )

    expected_rows = _require_int(
        _require_object(
            verification,
            "originalControl" if arm_id == "continued-control" else "treatmentCombined",
        ),
        "rows",
    )
    replay_sha_key = f"{model}ReplayContentSha256"
    _expect_equal(
        f"{row_id} replay content SHA-256",
        _require_str(provenance, replay_sha_key),
        _flag_value(command, f"--expected-{model}-replay-content-sha256"),
    )
    warm_sha_key = f"warmStart{model.capitalize()}Sha256"
    _expect_equal(
        f"{row_id} warm-start SHA-256",
        _require_str(provenance, warm_sha_key),
        _flag_value(command, f"--expected-warm-start-{model}-sha256"),
    )
    _expect_equal(
        f"{row_id} value training flag",
        bool(config.get("trainValue")),
        model == "value",
    )
    _expect_equal(
        f"{row_id} opponent training flag",
        bool(config.get("trainOpponent")),
        model == "opponent",
    )
    _expect_equal(
        f"{row_id} value replay rows",
        _require_int(results, "valueReplaySize"),
        expected_rows if model == "value" else 0,
    )
    _expect_equal(
        f"{row_id} opponent replay rows",
        _require_int(results, "opponentReplaySize"),
        expected_rows if model == "opponent" else 0,
    )
    replay_files_key = f"{model}ReplayFiles"
    replay_files = _require_str_list(config, replay_files_key)
    expected_games = _require_int(
        _require_object(
            verification,
            "originalControl" if arm_id == "continued-control" else "treatmentCombined",
        ),
        "games",
    )
    _expect_equal(f"{row_id} replay file count", len(replay_files), expected_games)
    if any("development." in path.lower() or "test." in path.lower() for path in replay_files):
        raise SystemExit(f"{row_id} summary provenance contains development or test replay.")

    trace_key = f"{model}SamplingTraceSha256"
    trace = _require_str(results, trace_key)
    if re.fullmatch(r"[0-9a-f]{64}", trace) is None:
        raise SystemExit(f"{row_id} summary sampling trace is missing or malformed.")
    latest = _require_object(results, f"latest{model.capitalize()}")
    _expect_equal(f"{row_id} latest step", _require_int(latest, "step"), EXPECTED_STEPS)

    checkpoints = _require_object_list(results, "checkpoints")
    checkpoint_steps = tuple(_require_int(entry, "step") for entry in checkpoints)
    _expect_equal(f"{row_id} checkpoint steps", checkpoint_steps, EXPECTED_CHECKPOINT_STEPS)
    output_root = Path(_flag_value(command, "--out-dir")).resolve()
    checkpoint_paths: list[str] = []
    for entry in checkpoints:
        checkpoint_path = Path(_require_str(entry, model)).resolve()
        try:
            checkpoint_path.relative_to(output_root)
        except ValueError as exc:
            raise SystemExit(
                f"{row_id} checkpoint escapes its frozen output root: {checkpoint_path}"
            ) from exc
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise SystemExit(f"{row_id} checkpoint not found: {checkpoint_path}")
        checkpoint_paths.append(str(checkpoint_path))

    expected_summary_root = checkpoint_root / seed_id / arm_id
    try:
        summary_path.resolve().relative_to(expected_summary_root.resolve())
    except ValueError as exc:
        raise SystemExit(f"{row_id} summary escapes its experiment output root.") from exc
    return {
        "id": row_id,
        "seedId": seed_id,
        "armId": arm_id,
        "model": model,
        "summary": str(summary_path),
        "samplingTraceSha256": trace,
        "finalCheckpoint": checkpoint_paths[-1],
        "completedSteps": EXPECTED_STEPS,
    }


def _validate_existing_completion_marker(
    *,
    completion_path: Path,
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> None:
    if not completion_path.exists():
        return
    marker = read_json(completion_path, label="extra-data training completion marker")
    _expect_equal(
        "completion marker experiment ID",
        _require_str(marker, "experimentId"),
        EXPERIMENT_ID,
    )
    _expect_equal(
        "completion marker source manifest SHA-256",
        _require_str(marker, "sourceManifestSha256"),
        _require_str(plan, "sourceManifestSha256"),
    )
    if pending:
        raise SystemExit(
            "Completion marker exists but one or more validated training summaries "
            f"are missing: {[row['id'] for row in pending]}"
        )
    _expect_equal("completion marker command count", _require_int(marker, "commands"), 8)
    marker_ids = tuple(_require_str_list(marker, "completedCommandIds"))
    _expect_equal(
        "completion marker command IDs",
        marker_ids,
        tuple(result["id"] for result in completed),
    )


def _completion_payload(
    *,
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "experimentId": EXPERIMENT_ID,
        "status": "completed",
        "completedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceManifestSha256": _require_str(plan, "sourceManifestSha256"),
        "implementationSha256": _require_str(
            _require_object(plan, "verification"),
            "implementationSha256",
        ),
        "commands": len(completed),
        "stepsPerCommand": EXPECTED_STEPS,
        "trainerThreads": EXPECTED_THREADS,
        "interopThreads": EXPECTED_INTEROP_THREADS,
        "completedCommandIds": [result["id"] for result in completed],
        "summaries": [result["summary"] for result in completed],
        "finalCheckpoints": [result["finalCheckpoint"] for result in completed],
        "samplingTraceSha256": {
            result["id"]: result["samplingTraceSha256"] for result in completed
        },
    }


def _overall_progress_payload(
    *,
    status: str,
    completed: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    active: str | None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "experimentId": EXPERIMENT_ID,
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "activeCommandId": active,
        "completedCommandIds": [result["id"] for result in completed],
        "pendingCommandIds": [_require_str(row, "id") for row in pending],
        "completed": len(completed),
        "total": 8,
    }


def _print_preflight(
    *,
    resolved_plan_path: Path,
    launch_root: Path,
    completed: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    print(f"[extra-data] resolvedPlan={resolved_plan_path}", flush=True)
    print("[extra-data] execution=sequential commands=8", flush=True)
    print("[extra-data] trainerThreads=4 interopThreads=1", flush=True)
    print(f"[extra-data] launchArtifacts={launch_root}", flush=True)
    print(
        f"[extra-data] resumeState=completed:{len(completed)} pending:{len(pending)}",
        flush=True,
    )
    if dry_run:
        print("[extra-data] dry run only; no training will be started.", flush=True)


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


def _summary_path(row: dict[str, Any]) -> Path:
    return Path(_flag_value(_command(row), "--summary-out")).resolve()


def _path_list_path(path_lists: dict[str, Any], key: str) -> Path:
    entry = _require_object(path_lists, key)
    path = Path(_require_str(entry, "path")).resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"Frozen replay path list not found: {path}")
    return path


def _content_sha(verification: dict[str, Any], dataset: str, model: str) -> str:
    return _require_str(
        _require_object(_require_object(verification, dataset), "contentSha256"),
        model,
    )


def _command(row: dict[str, Any]) -> list[str]:
    raw = row.get("command")
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"Expected non-empty command list for {row.get('id')!r}.")
    if not all(isinstance(item, str) and item for item in raw):
        raise SystemExit(f"Command entries must be non-empty strings for {row.get('id')!r}.")
    return [str(item) for item in raw]


def _flag_value(command: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1:
        raise SystemExit(f"Frozen command must contain {flag} exactly once; found {len(indices)}.")
    index = indices[0]
    if index + 1 >= len(command):
        raise SystemExit(f"Frozen command flag {flag} has no value.")
    return str(command[index + 1])


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _require_object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected object at {key}.")
    return value


def _require_object_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"Expected object list at {key}.")
    if not all(isinstance(item, dict) for item in value):
        raise SystemExit(f"Expected every {key} entry to be an object.")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Expected non-empty string at {key}.")
    return value


def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"Expected string list at {key}.")
    if not all(isinstance(item, str) and item for item in value):
        raise SystemExit(f"Expected every {key} entry to be a non-empty string.")
    return [str(item) for item in value]


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"Expected integer at {key}.")
    return value


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected={expected!r} actual={actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
