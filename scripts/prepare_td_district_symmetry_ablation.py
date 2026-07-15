from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trainer.td import (
    count_jsonl_rows,
    named_files_content_sha256,
    resolve_frozen_replay_split,
    sha256_file,
    write_replay_path_lists,
)

DEFAULT_MANIFEST = Path("configs/td-training/district-s4-ablation-pilot-v1.json")
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_PERMUTATION_SCHEME = "pawn-district-s4-d3-fixed-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and resolve the frozen district-symmetry pilot manifest. "
            "This command never launches training."
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
    payload = _read_json_object(manifest_path, label="pilot manifest")
    _validate_top_level(payload)
    _validate_experiment_shape(payload)

    experiment_id = _require_str(payload, "experimentId")
    replay = _require_object(payload, "replay")
    split_config = _require_object(replay, "split")
    if _require_str(split_config, "scheme") != "sha256-ranked-holdout-v1":
        raise SystemExit("Unsupported replay split scheme.")
    shards_dir = _resolve_path(repo_root, Path(_require_str(replay, "shardsDirectory")))
    split = resolve_frozen_replay_split(
        shards_dir=shards_dir,
        salt=_require_str(split_config, "salt"),
        validation_shards=_require_int(split_config, "validationShards"),
    )
    _expect_equal("replay shards", len(split.value_paths), _require_int(replay, "expectedShards"))
    _expect_equal(
        "training shards",
        len(split.training_keys),
        _require_int(split_config, "trainingShards"),
    )
    _expect_equal(
        "value replay bytes", split.value_bytes, _require_int(replay, "expectedValueBytes")
    )
    _expect_equal(
        "opponent replay bytes",
        split.opponent_bytes,
        _require_int(replay, "expectedOpponentBytes"),
    )
    _expect_equal(
        "replay inventory SHA-256",
        split.inventory_sha256,
        _require_str(replay, "expectedInventorySha256"),
    )
    _expect_equal(
        "replay membership SHA-256",
        split.membership_sha256,
        _require_str(split_config, "expectedMembershipSha256"),
    )
    replay_content = _require_object(replay, "expectedContentSha256")
    training_content = _require_object(split_config, "expectedTrainingContentSha256")
    validation_content = _require_object(split_config, "expectedValidationContentSha256")
    _expect_equal(
        "value replay content SHA-256",
        split.value_content_sha256,
        _require_str(replay_content, "value"),
    )
    _expect_equal(
        "opponent replay content SHA-256",
        split.opponent_content_sha256,
        _require_str(replay_content, "opponent"),
    )
    _expect_equal(
        "training value replay content SHA-256",
        split.training_value_content_sha256,
        _require_str(training_content, "value"),
    )
    _expect_equal(
        "training opponent replay content SHA-256",
        split.training_opponent_content_sha256,
        _require_str(training_content, "opponent"),
    )
    _expect_equal(
        "validation value replay content SHA-256",
        split.validation_value_content_sha256,
        _require_str(validation_content, "value"),
    )
    _expect_equal(
        "validation opponent replay content SHA-256",
        split.validation_opponent_content_sha256,
        _require_str(validation_content, "opponent"),
    )

    all_value_paths = [split.value_paths[key] for key in sorted(split.value_paths)]
    all_opponent_paths = [split.opponent_paths[key] for key in sorted(split.opponent_paths)]
    value_rows = count_jsonl_rows(all_value_paths)
    opponent_rows = count_jsonl_rows(all_opponent_paths)
    _expect_equal("value replay rows", value_rows, _require_int(replay, "expectedValueRows"))
    _expect_equal(
        "opponent replay rows",
        opponent_rows,
        _require_int(replay, "expectedOpponentRows"),
    )

    warm_start = _require_object(payload, "warmStart")
    value_warm = _validate_warm_start(
        repo_root=repo_root,
        payload=_require_object(warm_start, "value"),
        label="value",
    )
    opponent_warm = _validate_warm_start(
        repo_root=repo_root,
        payload=_require_object(warm_start, "opponent"),
        label="opponent",
    )
    provenance = _require_object(payload, "provenance")
    implementation_files = _require_str_list(provenance, "implementationFiles")
    try:
        implementation_sha256 = named_files_content_sha256(
            repo_root=repo_root,
            relative_paths=implementation_files,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _expect_equal(
        "training implementation SHA-256",
        implementation_sha256,
        _require_str(provenance, "expectedImplementationSha256"),
    )
    source_manifest_sha256 = sha256_file(manifest_path)

    outputs = _require_object(payload, "outputs")
    training_input_dir = _resolve_path(
        repo_root, Path(_require_str(outputs, "trainingInputDirectory"))
    )
    checkpoint_dir = _resolve_path(repo_root, Path(_require_str(outputs, "checkpointDirectory")))
    resolved_manifest_path = _resolve_path(
        repo_root, Path(_require_str(outputs, "resolvedManifest"))
    )
    path_lists = write_replay_path_lists(split=split, output_dir=training_input_dir)
    commands = _training_commands(
        payload=payload,
        experiment_id=experiment_id,
        python_bin=Path(sys.executable).resolve(),
        checkpoint_dir=checkpoint_dir,
        path_lists=path_lists,
        value_warm=value_warm,
        opponent_warm=opponent_warm,
        repo_root=repo_root,
        manifest_path=manifest_path,
        source_manifest_sha256=source_manifest_sha256,
        implementation_sha256=implementation_sha256,
        training_value_content_sha256=split.training_value_content_sha256,
        training_opponent_content_sha256=split.training_opponent_content_sha256,
    )
    smoke_commands = _guardrail_smoke_commands(
        commands=commands,
        checkpoint_dir=checkpoint_dir,
        seed_id=_require_str(_require_object(payload, "selection"), "primarySeedId"),
    )

    resolved = {
        "schemaVersion": EXPECTED_SCHEMA_VERSION,
        "experimentId": experiment_id,
        "status": "review-required",
        "launchAuthorized": False,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(manifest_path),
        "sourceManifestSha256": source_manifest_sha256,
        "verification": {
            "permutationScheme": EXPECTED_PERMUTATION_SCHEME,
            "replayShards": len(split.value_paths),
            "trainingShards": len(split.training_keys),
            "validationShards": len(split.validation_keys),
            "valueRows": value_rows,
            "opponentRows": opponent_rows,
            "valueBytes": split.value_bytes,
            "opponentBytes": split.opponent_bytes,
            "inventorySha256": split.inventory_sha256,
            "membershipSha256": split.membership_sha256,
            "contentSha256": {
                "all": {
                    "value": split.value_content_sha256,
                    "opponent": split.opponent_content_sha256,
                },
                "training": {
                    "value": split.training_value_content_sha256,
                    "opponent": split.training_opponent_content_sha256,
                },
                "validation": {
                    "value": split.validation_value_content_sha256,
                    "opponent": split.validation_opponent_content_sha256,
                },
            },
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
        "split": {
            "trainingKeys": split.training_keys,
            "validationKeys": split.validation_keys,
            "pathLists": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in path_lists.items()
            },
        },
        "commands": commands,
        "guardrailSmokeCommands": smoke_commands,
        "selection": payload.get("selection"),
        "evaluation": payload.get("evaluation"),
        "experimentalPacks": payload.get("experimentalPacks"),
        "guardrails": payload.get("guardrails"),
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "experimentId": experiment_id,
                "status": "review-required",
                "launchAuthorized": False,
                "resolvedManifest": str(resolved_manifest_path),
                "commandsPrepared": len(commands),
                "trainingShards": len(split.training_keys),
                "validationShards": len(split.validation_keys),
            },
            indent=2,
        )
    )
    return 0


def _training_commands(
    *,
    payload: dict[str, Any],
    experiment_id: str,
    python_bin: Path,
    checkpoint_dir: Path,
    path_lists: dict[str, Path],
    value_warm: Path,
    opponent_warm: Path,
    repo_root: Path,
    manifest_path: Path,
    source_manifest_sha256: str,
    implementation_sha256: str,
    training_value_content_sha256: str,
    training_opponent_content_sha256: str,
) -> list[dict[str, Any]]:
    training = _require_object(payload, "training")
    arms = _require_object_list(payload, "arms")
    seeds = _require_object_list(payload, "seeds")
    commands: list[dict[str, Any]] = []
    for seed in seeds:
        seed_id = _require_str(seed, "id")
        training_seed = _require_int(seed, "trainingSeed")
        augmentation_seed = _require_int(seed, "augmentationSeed")
        for arm in arms:
            arm_id = _require_str(arm, "id")
            augmentation = _require_str(arm, "districtAugmentation")
            if augmentation not in ("none", "s4"):
                raise SystemExit(f"Unsupported district augmentation mode: {augmentation!r}")
            run_root = checkpoint_dir / seed_id / arm_id
            common = [
                "--seed",
                str(training_seed),
                "--district-augmentation",
                augmentation,
                "--district-augmentation-seed",
                str(augmentation_seed),
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
                "--out-dir",
                str(run_root),
            ]
            value_summary = run_root / "value.summary.json"
            value_command = [
                str(python_bin),
                "-m",
                "scripts.train_td",
                "--run-label",
                f"{experiment_id}-{seed_id}-{arm_id}-value",
                "--steps",
                str(_require_int(training, "valueUpdates")),
                "--value-replay-list",
                str(path_lists["trainingValue"]),
                "--expected-value-replay-content-sha256",
                training_value_content_sha256,
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
                "--summary-out",
                str(value_summary),
                *common,
            ]
            opponent_summary = run_root / "opponent.summary.json"
            opponent_command = [
                str(python_bin),
                "-m",
                "scripts.train_td",
                "--run-label",
                f"{experiment_id}-{seed_id}-{arm_id}-opponent",
                "--steps",
                str(_require_int(training, "opponentUpdates")),
                "--opponent-replay-list",
                str(path_lists["trainingOpponent"]),
                "--expected-opponent-replay-content-sha256",
                training_opponent_content_sha256,
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
                "--summary-out",
                str(opponent_summary),
                *common,
            ]
            commands.extend(
                (
                    {
                        "id": f"{seed_id}-{arm_id}-value",
                        "seedId": seed_id,
                        "armId": arm_id,
                        "model": "value",
                        "promotionEligibility": bool(seed.get("promotionEligibility", False)),
                        "command": value_command,
                    },
                    {
                        "id": f"{seed_id}-{arm_id}-opponent",
                        "seedId": seed_id,
                        "armId": arm_id,
                        "model": "opponent",
                        "promotionEligibility": bool(seed.get("promotionEligibility", False)),
                        "command": opponent_command,
                    },
                )
            )
    return commands


def _guardrail_smoke_commands(
    *,
    commands: list[dict[str, Any]],
    checkpoint_dir: Path,
    seed_id: str,
) -> list[dict[str, Any]]:
    smoke_commands: list[dict[str, Any]] = []
    for row in commands:
        if row.get("seedId") != seed_id:
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
                "seedId": seed_id,
                "armId": arm_id,
                "model": model,
                "steps": 1,
                "command": command,
            }
        )
    if len(smoke_commands) != 4:
        raise SystemExit(
            "Guardrail smoke preparation requires value and opponent commands for "
            "both primary-seed arms."
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


def _validate_top_level(payload: dict[str, Any]) -> None:
    _expect_equal("schema version", _require_int(payload, "schemaVersion"), EXPECTED_SCHEMA_VERSION)
    _expect_equal(
        "permutation scheme",
        _require_str(payload, "permutationScheme"),
        EXPECTED_PERMUTATION_SCHEME,
    )
    _expect_equal("manifest status", _require_str(payload, "status"), "review-required")
    if payload.get("launchAuthorized") is not False:
        raise SystemExit("Pilot manifest must explicitly set launchAuthorized=false.")


def _validate_experiment_shape(payload: dict[str, Any]) -> None:
    warm_start = _require_object(payload, "warmStart")
    _expect_equal(
        "optimizer initialization",
        _require_str(warm_start, "optimizerState"),
        "fresh-in-both-arms",
    )
    arms = _require_object_list(payload, "arms")
    if len(arms) != 2:
        raise SystemExit("Pilot manifest must contain exactly two matched arms.")
    arm_ids = [_require_str(arm, "id") for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise SystemExit("Pilot arm IDs must be unique.")
    modes = {_require_str(arm, "districtAugmentation") for arm in arms}
    if modes != {"none", "s4"}:
        raise SystemExit("Pilot arms must contain exactly one control and one S4 arm.")

    seeds = _require_object_list(payload, "seeds")
    if len(seeds) != 2:
        raise SystemExit("Pilot manifest must contain exactly two matched seeds.")
    seed_ids = [_require_str(seed, "id") for seed in seeds]
    if len(set(seed_ids)) != len(seed_ids):
        raise SystemExit("Pilot seed IDs must be unique.")
    if sum(bool(seed.get("promotionEligibility", False)) for seed in seeds) != 1:
        raise SystemExit("Exactly one pilot seed must be promotion-eligible.")

    training = _require_object(payload, "training")
    _expect_equal("value loss", _require_str(training, "valueLoss"), "huber")

    selection = _require_object(payload, "selection")
    primary_seed_id = _require_str(selection, "primarySeedId")
    replication_seed_id = _require_str(selection, "replicationSeedId")
    _expect_equal("primary seed", primary_seed_id, seed_ids[0])
    _expect_equal("replication seed", replication_seed_id, seed_ids[1])
    _expect_equal(
        "selected checkpoint step",
        _require_int(selection, "checkpointStep"),
        _require_int(training, "valueUpdates"),
    )
    _expect_equal(
        "value/opponent final step",
        _require_int(training, "valueUpdates"),
        _require_int(training, "opponentUpdates"),
    )
    if selection.get("intermediateCheckpointSelectionAllowed") is not False:
        raise SystemExit("Intermediate checkpoint selection must be disabled.")

    evaluation = _require_object(payload, "evaluation")
    heldout = _require_object(evaluation, "heldoutReplay")
    if heldout.get("usesAllValidationShards") is not True:
        raise SystemExit("Heldout replay evaluation must use all validation shards.")
    full_games = _require_object(evaluation, "fullGames")
    _require_int(full_games, "gamesPerSide")
    _require_str(full_games, "seedPrefix")


def _validate_warm_start(*, repo_root: Path, payload: dict[str, Any], label: str) -> Path:
    path = _resolve_path(repo_root, Path(_require_str(payload, "path")))
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{label} warm-start checkpoint not found: {path}")
    _expect_equal(
        f"{label} warm-start SHA-256",
        sha256_file(path),
        _require_str(payload, "sha256").lower(),
    )
    return path


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
