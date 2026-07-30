from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.td import (
    REPLAY_KEY_MODE_RUN_QUALIFIED,
    count_jsonl_rows,
    named_files_content_sha256,
    replay_content_sha256,
    sha256_file,
)

DEFAULT_MANIFEST = Path("configs/td-training/td-hard-extra-data-continuation-v1.json")
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT_ID = "td-hard-extra-data-continuation-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and resolve the matched Hard-teacher extra-data continuation "
            "experiment. This command never launches training or reads final-test replay."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve_path(repo_root, args.manifest)
    manifest = _read_json_object(manifest_path, label="experiment manifest")
    _validate_manifest(manifest)

    replay_key_mode = _require_str(manifest, "replayKeyMode")
    _expect_equal(
        "replay key mode",
        replay_key_mode,
        REPLAY_KEY_MODE_RUN_QUALIFIED,
    )
    replay = _require_object(manifest, "replay")
    original = _require_object(replay, "originalControl")
    imported = _require_object(replay, "importedSplit")
    treatment = _require_object(replay, "treatmentCombined")

    split_manifest_path = _resolve_path(
        repo_root,
        Path(_require_str(imported, "manifest")),
    )
    _expect_equal(
        "imported split manifest SHA-256",
        sha256_file(split_manifest_path),
        _require_str(imported, "manifestSha256"),
    )
    split_manifest = _read_json_object(split_manifest_path, label="imported split manifest")
    _validate_imported_split(split_manifest, imported)

    split_verification = _require_object(split_manifest, "verification")
    split_path_lists = _require_object(split_verification, "pathLists")
    training_config = _require_object(imported, "training")
    development_config = _require_object(imported, "development")
    sealed_test_config = _require_object(imported, "sealedFinalTest")

    imported_training_value = _validated_split_paths(
        repo_root=repo_root,
        path_lists=split_path_lists,
        entry_name="trainingValue",
        expected_sha256=_require_str(
            _require_object(training_config, "pathListSha256"),
            "value",
        ),
    )
    imported_training_opponent = _validated_split_paths(
        repo_root=repo_root,
        path_lists=split_path_lists,
        entry_name="trainingOpponent",
        expected_sha256=_require_str(
            _require_object(training_config, "pathListSha256"),
            "opponent",
        ),
    )
    development_value = _validated_split_paths(
        repo_root=repo_root,
        path_lists=split_path_lists,
        entry_name="developmentValue",
        expected_sha256=_require_str(
            _require_object(development_config, "pathListSha256"),
            "value",
        ),
    )
    development_opponent = _validated_split_paths(
        repo_root=repo_root,
        path_lists=split_path_lists,
        entry_name="developmentOpponent",
        expected_sha256=_require_str(
            _require_object(development_config, "pathListSha256"),
            "opponent",
        ),
    )
    sealed_test_lists = _validate_sealed_test_lists(
        repo_root=repo_root,
        path_lists=split_path_lists,
        config=sealed_test_config,
    )

    original_shards = _resolve_path(
        repo_root,
        Path(_require_str(original, "shardsDirectory")),
    )
    original_value = sorted(original_shards.glob("*.value.jsonl"))
    original_opponent = sorted(original_shards.glob("*.opponent.jsonl"))
    _expect_equal("original value games", len(original_value), _require_int(original, "games"))
    _expect_equal(
        "original opponent games",
        len(original_opponent),
        _require_int(original, "games"),
    )

    original_stats = _verify_dataset(
        label="original control",
        value_paths=original_value,
        opponent_paths=original_opponent,
        expected=original,
        replay_key_mode=replay_key_mode,
    )
    imported_training_stats = _verify_dataset(
        label="imported training",
        value_paths=imported_training_value,
        opponent_paths=imported_training_opponent,
        expected=training_config,
        replay_key_mode=replay_key_mode,
    )
    development_stats = _verify_dataset(
        label="imported development",
        value_paths=development_value,
        opponent_paths=development_opponent,
        expected=development_config,
        replay_key_mode=replay_key_mode,
    )

    treatment_value = [*original_value, *imported_training_value]
    treatment_opponent = [*original_opponent, *imported_training_opponent]
    treatment_stats = _verify_dataset(
        label="combined treatment",
        value_paths=treatment_value,
        opponent_paths=treatment_opponent,
        expected=treatment,
        replay_key_mode=replay_key_mode,
    )
    _expect_equal(
        "combined treatment rows",
        treatment_stats["rows"],
        original_stats["rows"] + imported_training_stats["rows"],
    )

    warm_start = _require_object(manifest, "warmStart")
    value_warm = _validate_checkpoint(
        repo_root=repo_root,
        payload=_require_object(warm_start, "value"),
        label="value",
    )
    opponent_warm = _validate_checkpoint(
        repo_root=repo_root,
        payload=_require_object(warm_start, "opponent"),
        label="opponent",
    )

    provenance = _require_object(manifest, "provenance")
    implementation_files = _require_str_list(provenance, "implementationFiles")
    try:
        implementation_sha256 = named_files_content_sha256(
            repo_root=repo_root,
            relative_paths=implementation_files,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _expect_equal(
        "implementation SHA-256",
        implementation_sha256,
        _require_str(provenance, "expectedImplementationSha256"),
    )
    source_manifest_sha256 = sha256_file(manifest_path)

    outputs = _require_object(manifest, "outputs")
    training_input_dir = _resolve_path(
        repo_root,
        Path(_require_str(outputs, "trainingInputDirectory")),
    )
    checkpoint_dir = _resolve_path(
        repo_root,
        Path(_require_str(outputs, "checkpointDirectory")),
    )
    resolved_plan_path = _resolve_path(
        repo_root,
        Path(_require_str(outputs, "resolvedPlan")),
    )
    path_lists = _write_experiment_path_lists(
        output_dir=training_input_dir,
        original_value=original_value,
        original_opponent=original_opponent,
        treatment_value=treatment_value,
        treatment_opponent=treatment_opponent,
        development_value=development_value,
        development_opponent=development_opponent,
    )
    commands = _training_commands(
        manifest=manifest,
        python_bin=Path(sys.executable).resolve(),
        manifest_path=manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        implementation_sha256=implementation_sha256,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        path_lists=path_lists,
        replay_key_mode=replay_key_mode,
        value_warm=value_warm,
        opponent_warm=opponent_warm,
        original_stats=original_stats,
        treatment_stats=treatment_stats,
    )
    smoke_commands = _guardrail_smoke_commands(
        commands=commands,
        checkpoint_dir=checkpoint_dir,
    )

    resolved_plan = {
        "schemaVersion": EXPECTED_SCHEMA_VERSION,
        "experimentId": EXPECTED_EXPERIMENT_ID,
        "status": "review-required",
        "launchAuthorized": False,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(manifest_path),
        "sourceManifestSha256": source_manifest_sha256,
        "verification": {
            "replayKeyMode": replay_key_mode,
            "originalControl": original_stats,
            "importedTraining": imported_training_stats,
            "treatmentCombined": treatment_stats,
            "development": development_stats,
            "implementationSha256": implementation_sha256,
            "valueWarmStart": {
                "path": str(value_warm),
                "sha256": sha256_file(value_warm),
            },
            "opponentWarmStart": {
                "path": str(opponent_warm),
                "sha256": sha256_file(opponent_warm),
            },
        },
        "pathLists": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "entries": len(_read_path_list(path)),
            }
            for name, path in path_lists.items()
        },
        "sealedFinalTest": {
            "rawReplayReadDuringPreparation": False,
            "pathLists": sealed_test_lists,
            "unlockCondition": (
                "Primary treatment checkpoint step is frozen from development metrics "
                "and replication direction has been reviewed."
            ),
        },
        "trainingCommands": commands,
        "guardrailSmokeCommands": smoke_commands,
        "postTraining": {
            "developmentEvaluationCommands": "deferred until checkpoint SHA-256 values exist",
            "selection": manifest.get("selection"),
            "finalTest": "sealed until candidate checkpoint step is frozen",
            "evaluation": manifest.get("evaluation"),
        },
        "guardrails": manifest.get("guardrails"),
    }
    resolved_plan_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_plan_path.write_text(json.dumps(resolved_plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "experimentId": EXPECTED_EXPERIMENT_ID,
                "status": "review-required",
                "launchAuthorized": False,
                "resolvedPlan": str(resolved_plan_path),
                "trainingCommandsPrepared": len(commands),
                "guardrailSmokeCommandsPrepared": len(smoke_commands),
                "originalControlGames": original_stats["games"],
                "treatmentGames": treatment_stats["games"],
                "developmentGames": development_stats["games"],
                "sealedFinalTestGames": _require_int(sealed_test_config, "games"),
                "rawFinalTestRead": False,
            },
            indent=2,
        )
    )
    return 0


def _training_commands(
    *,
    manifest: dict[str, Any],
    python_bin: Path,
    manifest_path: Path,
    source_manifest_sha256: str,
    implementation_sha256: str,
    repo_root: Path,
    checkpoint_dir: Path,
    path_lists: dict[str, Path],
    replay_key_mode: str,
    value_warm: Path,
    opponent_warm: Path,
    original_stats: dict[str, Any],
    treatment_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    training = _require_object(manifest, "training")
    arms = _require_object_list(manifest, "arms")
    seeds = _require_object_list(manifest, "seeds")
    experiment_id = _require_str(manifest, "experimentId")
    commands: list[dict[str, Any]] = []
    arm_inputs = {
        "continued-control": {
            "valueList": path_lists["controlValue"],
            "opponentList": path_lists["controlOpponent"],
            "valueSha256": _require_str(
                _require_object(original_stats, "contentSha256"),
                "value",
            ),
            "opponentSha256": _require_str(
                _require_object(original_stats, "contentSha256"),
                "opponent",
            ),
        },
        "extra-data-treatment": {
            "valueList": path_lists["treatmentValue"],
            "opponentList": path_lists["treatmentOpponent"],
            "valueSha256": _require_str(
                _require_object(treatment_stats, "contentSha256"),
                "value",
            ),
            "opponentSha256": _require_str(
                _require_object(treatment_stats, "contentSha256"),
                "opponent",
            ),
        },
    }
    for seed in seeds:
        seed_id = _require_str(seed, "id")
        training_seed = _require_int(seed, "trainingSeed")
        for arm in arms:
            arm_id = _require_str(arm, "id")
            arm_input = arm_inputs[arm_id]
            run_root = checkpoint_dir / seed_id / arm_id
            common = [
                "--seed",
                str(training_seed),
                "--replay-key-mode",
                replay_key_mode,
                "--district-augmentation",
                _require_str(training, "districtAugmentation"),
                "--max-grad-norm",
                str(_require_number(training, "maxGradNorm")),
                "--save-every-steps",
                str(_require_int(training, "saveEverySteps")),
                "--progress-every-steps",
                str(_require_int(training, "progressEverySteps")),
                "--num-threads",
                str(_require_int(training, "numThreads")),
                "--num-interop-threads",
                str(_require_int(training, "numInteropThreads")),
                "--experiment-manifest",
                str(manifest_path),
                "--expected-experiment-manifest-sha256",
                source_manifest_sha256,
                "--provenance-repo-root",
                str(repo_root),
                "--expected-implementation-sha256",
                implementation_sha256,
            ]
            value_command = [
                str(python_bin),
                "-m",
                "scripts.train_td",
                "--run-label",
                f"{experiment_id}-{seed_id}-{arm_id}-value",
                "--steps",
                str(_require_int(training, "valueUpdates")),
                "--value-replay-list",
                str(arm_input["valueList"]),
                "--expected-value-replay-content-sha256",
                str(arm_input["valueSha256"]),
                "--value-batch-size",
                str(_require_int(training, "valueBatchSize")),
                "--gamma",
                str(_require_number(training, "gamma")),
                "--value-learning-rate",
                str(_require_number(training, "valueLearningRate")),
                "--value-weight-decay",
                str(_require_number(training, "valueWeightDecay")),
                "--target-sync-interval",
                str(_require_int(training, "targetSyncInterval")),
                "--value-target-mode",
                _require_str(training, "valueTargetMode"),
                "--td-lambda",
                str(_require_number(training, "tdLambda")),
                "--warm-start-value-checkpoint",
                str(value_warm),
                "--expected-warm-start-value-sha256",
                sha256_file(value_warm),
                "--disable-opponent",
                "--out-dir",
                str(run_root / "value"),
                "--summary-out",
                str(run_root / "value.summary.json"),
                *common,
            ]
            opponent_command = [
                str(python_bin),
                "-m",
                "scripts.train_td",
                "--run-label",
                f"{experiment_id}-{seed_id}-{arm_id}-opponent",
                "--steps",
                str(_require_int(training, "opponentUpdates")),
                "--opponent-replay-list",
                str(arm_input["opponentList"]),
                "--expected-opponent-replay-content-sha256",
                str(arm_input["opponentSha256"]),
                "--opponent-batch-size",
                str(_require_int(training, "opponentBatchSize")),
                "--opponent-learning-rate",
                str(_require_number(training, "opponentLearningRate")),
                "--opponent-weight-decay",
                str(_require_number(training, "opponentWeightDecay")),
                "--warm-start-opponent-checkpoint",
                str(opponent_warm),
                "--expected-warm-start-opponent-sha256",
                sha256_file(opponent_warm),
                "--disable-value",
                "--out-dir",
                str(run_root / "opponent"),
                "--summary-out",
                str(run_root / "opponent.summary.json"),
                *common,
            ]
            for model, command in (("value", value_command), ("opponent", opponent_command)):
                commands.append(
                    {
                        "id": f"{seed_id}-{arm_id}-{model}",
                        "seedId": seed_id,
                        "armId": arm_id,
                        "model": model,
                        "trainingSeed": training_seed,
                        "promotionEligibility": bool(seed.get("promotionEligibility", False)),
                        "command": command,
                    }
                )
    return commands


def _guardrail_smoke_commands(
    *,
    commands: list[dict[str, Any]],
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    smoke_commands: list[dict[str, Any]] = []
    for row in commands:
        if row.get("seedId") != "primary":
            continue
        arm_id = str(row["armId"])
        model = str(row["model"])
        command = [str(value) for value in row["command"]]
        smoke_root = checkpoint_dir / "guardrail-smoke" / arm_id / model
        _replace_flag(command, "--run-label", f"guardrail-smoke-{arm_id}-{model}")
        _replace_flag(command, "--steps", "1")
        _replace_flag(command, "--save-every-steps", "0")
        _replace_flag(command, "--progress-every-steps", "1")
        _replace_flag(command, "--out-dir", str(smoke_root))
        _replace_flag(command, "--summary-out", str(smoke_root / "summary.json"))
        smoke_commands.append(
            {
                "id": f"guardrail-smoke-{arm_id}-{model}",
                "seedId": "primary",
                "armId": arm_id,
                "model": model,
                "steps": 1,
                "command": command,
            }
        )
    if len(smoke_commands) != 4:
        raise SystemExit(
            "Guardrail smoke preparation requires value and opponent commands "
            "for both primary-seed arms."
        )
    return smoke_commands


def _replace_flag(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as exc:
        raise SystemExit(f"Prepared command is missing required flag {flag}.") from exc
    if index + 1 >= len(command):
        raise SystemExit(f"Prepared command flag {flag} has no value.")
    command[index + 1] = value


def _verify_dataset(
    *,
    label: str,
    value_paths: list[Path],
    opponent_paths: list[Path],
    expected: dict[str, Any],
    replay_key_mode: str,
) -> dict[str, Any]:
    expected_games = _require_int(expected, "games")
    _expect_equal(f"{label} value files", len(value_paths), expected_games)
    _expect_equal(f"{label} opponent files", len(opponent_paths), expected_games)
    value_rows = count_jsonl_rows(value_paths)
    opponent_rows = count_jsonl_rows(opponent_paths)
    _expect_equal(f"{label} paired rows", value_rows, opponent_rows)
    _expect_equal(f"{label} rows", value_rows, _require_int(expected, "rows"))
    expected_bytes = _require_object(expected, "bytes")
    value_bytes = sum(path.stat().st_size for path in value_paths)
    opponent_bytes = sum(path.stat().st_size for path in opponent_paths)
    _expect_equal(f"{label} value bytes", value_bytes, _require_int(expected_bytes, "value"))
    _expect_equal(
        f"{label} opponent bytes",
        opponent_bytes,
        _require_int(expected_bytes, "opponent"),
    )
    value_sha256 = replay_content_sha256(value_paths, key_mode=replay_key_mode)
    opponent_sha256 = replay_content_sha256(opponent_paths, key_mode=replay_key_mode)
    expected_content = _require_object(expected, "contentSha256")
    _expect_equal(
        f"{label} value content SHA-256",
        value_sha256,
        _require_str(expected_content, "value"),
    )
    _expect_equal(
        f"{label} opponent content SHA-256",
        opponent_sha256,
        _require_str(expected_content, "opponent"),
    )
    return {
        "games": expected_games,
        "rows": value_rows,
        "bytes": {
            "value": value_bytes,
            "opponent": opponent_bytes,
        },
        "contentSha256": {
            "value": value_sha256,
            "opponent": opponent_sha256,
        },
    }


def _validated_split_paths(
    *,
    repo_root: Path,
    path_lists: dict[str, Any],
    entry_name: str,
    expected_sha256: str,
) -> list[Path]:
    entry = _require_object(path_lists, entry_name)
    path = _resolve_path(repo_root, Path(_require_str(entry, "path")))
    _expect_equal(
        f"{entry_name} path-list SHA-256",
        sha256_file(path),
        expected_sha256,
    )
    _expect_equal(
        f"{entry_name} split-manifest path-list SHA-256",
        _require_str(entry, "sha256"),
        expected_sha256,
    )
    paths = _read_path_list(path)
    _expect_equal(f"{entry_name} entries", len(paths), _require_int(entry, "entries"))
    return paths


def _validate_sealed_test_lists(
    *,
    repo_root: Path,
    path_lists: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected = _require_object(config, "pathListSha256")
    result: dict[str, Any] = {}
    for model, entry_name in (
        ("value", "testValue"),
        ("opponent", "testOpponent"),
    ):
        entry = _require_object(path_lists, entry_name)
        path = _resolve_path(repo_root, Path(_require_str(entry, "path")))
        actual = sha256_file(path)
        expected_sha = _require_str(expected, model)
        _expect_equal(f"{entry_name} path-list SHA-256", actual, expected_sha)
        _expect_equal(
            f"{entry_name} split-manifest path-list SHA-256",
            _require_str(entry, "sha256"),
            expected_sha,
        )
        _expect_equal(
            f"{entry_name} entries",
            _require_int(entry, "entries"),
            _require_int(config, "games"),
        )
        result[model] = {
            "path": str(path),
            "sha256": actual,
            "entriesDeclared": _require_int(entry, "entries"),
        }
    return result


def _write_experiment_path_lists(
    *,
    output_dir: Path,
    original_value: list[Path],
    original_opponent: list[Path],
    treatment_value: list[Path],
    treatment_opponent: list[Path],
    development_value: list[Path],
    development_opponent: list[Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path_sets = {
        "controlValue": original_value,
        "controlOpponent": original_opponent,
        "treatmentValue": treatment_value,
        "treatmentOpponent": treatment_opponent,
        "developmentValue": development_value,
        "developmentOpponent": development_opponent,
    }
    outputs: dict[str, Path] = {}
    for name, paths in path_sets.items():
        output = output_dir / f"{name}.paths.txt"
        output.write_text(
            "".join(f"{path.resolve()}\n" for path in paths),
            encoding="utf-8",
        )
        outputs[name] = output.resolve()
    return outputs


def _validate_manifest(payload: dict[str, Any]) -> None:
    _expect_equal("schema version", _require_int(payload, "schemaVersion"), 1)
    _expect_equal(
        "experiment ID",
        _require_str(payload, "experimentId"),
        EXPECTED_EXPERIMENT_ID,
    )
    _expect_equal("manifest status", _require_str(payload, "status"), "review-required")
    if payload.get("launchAuthorized") is not False:
        raise SystemExit("Experiment manifest must explicitly set launchAuthorized=false.")
    warm_start = _require_object(payload, "warmStart")
    _expect_equal(
        "optimizer initialization",
        _require_str(warm_start, "optimizerState"),
        "fresh-in-both-arms",
    )
    arms = _require_object_list(payload, "arms")
    arm_ids = [_require_str(arm, "id") for arm in arms]
    _expect_equal(
        "matched arm IDs",
        arm_ids,
        ["continued-control", "extra-data-treatment"],
    )
    seeds = _require_object_list(payload, "seeds")
    _expect_equal(
        "matched seed IDs",
        [_require_str(seed, "id") for seed in seeds],
        ["primary", "replication"],
    )
    if sum(bool(seed.get("promotionEligibility", False)) for seed in seeds) != 1:
        raise SystemExit("Exactly one seed must be promotion-eligible.")
    training = _require_object(payload, "training")
    value_updates = _require_int(training, "valueUpdates")
    opponent_updates = _require_int(training, "opponentUpdates")
    _expect_equal("matched model updates", value_updates, opponent_updates)
    save_every = _require_int(training, "saveEverySteps")
    if save_every <= 0 or value_updates % save_every != 0:
        raise SystemExit("saveEverySteps must evenly divide the update count.")
    _expect_equal("district augmentation", _require_str(training, "districtAugmentation"), "none")
    _expect_equal("value loss", _require_str(training, "valueLoss"), "huber")
    selection = _require_object(payload, "selection")
    expected_steps = list(range(save_every, value_updates + 1, save_every))
    _expect_equal(
        "selection checkpoint steps",
        selection.get("checkpointSteps"),
        expected_steps,
    )
    if selection.get("candidateMustBeFrozenBeforeFinalTest") is not True:
        raise SystemExit("Candidate must be frozen before final-test access.")
    evaluation = _require_object(payload, "evaluation")
    final_test = _require_object(evaluation, "sealedFinalTest")
    if final_test.get("mustRemainUnreadUntilCandidateFrozen") is not True:
        raise SystemExit("Final-test replay must remain sealed during preparation.")


def _validate_imported_split(
    split_manifest: dict[str, Any],
    imported_config: dict[str, Any],
) -> None:
    _expect_equal("split schema version", _require_int(split_manifest, "schemaVersion"), 1)
    _expect_equal("split status", _require_str(split_manifest, "status"), "frozen-no-training")
    if split_manifest.get("launchAuthorized") is not False:
        raise SystemExit("Imported split manifest must set launchAuthorized=false.")
    split = _require_object(split_manifest, "split")
    _expect_equal(
        "split membership SHA-256",
        _require_str(split, "membershipSha256"),
        _require_str(imported_config, "membershipSha256"),
    )
    verification = _require_object(split_manifest, "verification")
    if verification.get("disjoint") is not True or verification.get("complete") is not True:
        raise SystemExit("Imported train/development/test split is not complete and disjoint.")


def _validate_checkpoint(*, repo_root: Path, payload: dict[str, Any], label: str) -> Path:
    path = _resolve_path(repo_root, Path(_require_str(payload, "path")))
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{label} warm-start checkpoint not found: {path}")
    _expect_equal(
        f"{label} warm-start SHA-256",
        sha256_file(path),
        _require_str(payload, "sha256"),
    )
    return path


def _read_path_list(path: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        replay = Path(entry)
        if not replay.is_absolute():
            replay = path.parent / replay
        replay = replay.resolve()
        if not replay.exists() or not replay.is_file():
            raise SystemExit(f"Replay path from {path} not found: {replay}")
        paths.append(replay)
    if not paths:
        raise SystemExit(f"Replay path list is empty: {path}")
    return paths


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must contain a JSON object: {path}")
    return payload


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


def _require_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Expected non-empty string list at {key}.")
    if not all(isinstance(item, str) and item for item in value):
        raise SystemExit(f"Expected every {key} entry to be a non-empty string.")
    return [str(item) for item in value]


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


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected={expected!r} actual={actual!r}")


if __name__ == "__main__":
    raise SystemExit(main())
