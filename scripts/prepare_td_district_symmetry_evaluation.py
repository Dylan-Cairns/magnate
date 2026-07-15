from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from trainer.td import (
    load_opponent_checkpoint,
    load_value_checkpoint,
    sha256_file,
)

DEFAULT_MANIFEST = Path("configs/td-training/district-s4-ablation-pilot-v1.json")


@dataclass(frozen=True)
class FinalRunArtifacts:
    seed_id: str
    arm_id: str
    value_checkpoint: Path
    value_sha256: str
    value_sampling_trace_sha256: str
    opponent_checkpoint: Path
    opponent_sha256: str
    opponent_sampling_trace_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate completed district-S4 pilot summaries, freeze the final-step "
            "candidate, and prepare exact export/evaluation commands. This command "
            "does not export packs or run evaluations."
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
    _require_supported_runtime()
    repo_root = args.repo_root.resolve()
    manifest_path = _resolve_path(repo_root, args.manifest)
    manifest = _read_json_object(manifest_path, label="pilot manifest")
    outputs = _require_object(manifest, "outputs")
    resolved_path = _resolve_path(repo_root, Path(_require_str(outputs, "resolvedManifest")))
    resolved = _read_json_object(resolved_path, label="resolved pilot manifest")
    source_sha256 = sha256_file(manifest_path)
    _expect_equal(
        "resolved source manifest SHA-256",
        resolved.get("sourceManifestSha256"),
        source_sha256,
    )

    training = _require_object(manifest, "training")
    selection = _require_object(manifest, "selection")
    checkpoint_step = _require_int(selection, "checkpointStep")
    _expect_equal(
        "final value checkpoint step",
        checkpoint_step,
        _require_int(training, "valueUpdates"),
    )
    _expect_equal(
        "final opponent checkpoint step",
        checkpoint_step,
        _require_int(training, "opponentUpdates"),
    )
    checkpoint_root = _resolve_path(repo_root, Path(_require_str(outputs, "checkpointDirectory")))
    expected_implementation_sha256 = _require_str(
        _require_object(manifest, "provenance"),
        "expectedImplementationSha256",
    )
    expected_training_content = _require_object(
        _require_object(_require_object(manifest, "replay"), "split"),
        "expectedTrainingContentSha256",
    )
    warm_start = _require_object(manifest, "warmStart")

    runs: dict[tuple[str, str], FinalRunArtifacts] = {}
    arm_modes = {
        _require_str(arm, "id"): _require_str(arm, "districtAugmentation")
        for arm in _require_object_list(manifest, "arms")
    }
    for seed in _require_object_list(manifest, "seeds"):
        seed_id = _require_str(seed, "id")
        training_seed = _require_int(seed, "trainingSeed")
        for arm_id, augmentation_mode in arm_modes.items():
            runs[(seed_id, arm_id)] = _load_final_run(
                run_root=checkpoint_root / seed_id / arm_id,
                seed_id=seed_id,
                arm_id=arm_id,
                training_seed=training_seed,
                augmentation_mode=augmentation_mode,
                checkpoint_step=checkpoint_step,
                source_manifest_sha256=source_sha256,
                implementation_sha256=expected_implementation_sha256,
                value_replay_sha256=_require_str(expected_training_content, "value"),
                opponent_replay_sha256=_require_str(expected_training_content, "opponent"),
                value_warm_sha256=_require_str(_require_object(warm_start, "value"), "sha256"),
                opponent_warm_sha256=_require_str(
                    _require_object(warm_start, "opponent"), "sha256"
                ),
            )

    _verify_matched_sampling_traces(runs)
    primary_seed_id = _require_str(selection, "primarySeedId")
    replication_seed_id = _require_str(selection, "replicationSeedId")
    primary_augmented = runs[(primary_seed_id, "s4-augmented")]
    candidate_freeze = {
        "seedId": primary_seed_id,
        "armId": "s4-augmented",
        "checkpointStep": checkpoint_step,
        "valueCheckpoint": str(primary_augmented.value_checkpoint),
        "valueSha256": primary_augmented.value_sha256,
        "opponentCheckpoint": str(primary_augmented.opponent_checkpoint),
        "opponentSha256": primary_augmented.opponent_sha256,
        "frozenBeforeReservedStrategicEvaluation": True,
    }
    candidate_freeze_sha256 = _canonical_sha256(candidate_freeze)

    evaluation_input_dir = _resolve_path(
        repo_root, Path(_require_str(outputs, "evaluationInputDirectory"))
    )
    evaluation_artifact_dir = _resolve_path(
        repo_root, Path(_require_str(outputs, "evaluationArtifactDirectory"))
    )
    evaluation_input_dir.mkdir(parents=True, exist_ok=True)
    packs = _prepare_export_commands(
        manifest=manifest,
        repo_root=repo_root,
        runs=runs,
        primary_seed_id=primary_seed_id,
        replication_seed_id=replication_seed_id,
    )
    heldout_commands = _prepare_holdout_commands(
        manifest=manifest,
        resolved=resolved,
        runs=runs,
        python_bin=Path(sys.executable).resolve(),
        output_dir=evaluation_artifact_dir / "heldout",
        checkpoint_step=checkpoint_step,
    )
    symmetry_commands = _prepare_symmetry_commands(
        manifest=manifest,
        resolved=resolved,
        packs=packs,
        output_dir=evaluation_artifact_dir / "symmetry",
    )
    strategic_commands = _prepare_strategic_commands(
        manifest=manifest,
        packs=packs,
        output_dir=evaluation_artifact_dir / "strategic",
    )
    head_to_head = _prepare_head_to_head(
        manifest=manifest,
        packs=packs,
        config_dir=evaluation_input_dir / "head-to-head",
        output_dir=evaluation_artifact_dir / "head-to-head",
    )

    plan_path = evaluation_input_dir / "evaluation-plan.json"
    plan = {
        "schemaVersion": 1,
        "experimentId": _require_str(manifest, "experimentId"),
        "status": "prepared-not-launched",
        "launchAuthorized": False,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceManifest": str(manifest_path),
        "sourceManifestSha256": source_sha256,
        "resolvedManifest": str(resolved_path),
        "resolvedManifestSha256": sha256_file(resolved_path),
        "candidateFreeze": candidate_freeze,
        "candidateFreezeSha256": candidate_freeze_sha256,
        "selectionRules": selection,
        "evaluationRules": manifest.get("evaluation"),
        "samplingTraceVerification": _sampling_trace_payload(runs),
        "exportCommands": packs["commands"],
        "heldoutCommands": heldout_commands,
        "symmetryCommands": symmetry_commands,
        "strategicCommands": strategic_commands,
        "headToHead": head_to_head,
        "guardrails": manifest.get("guardrails"),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "prepared-not-launched",
                "evaluationPlan": str(plan_path),
                "candidateFreezeSha256": candidate_freeze_sha256,
                "exportCommands": len(packs["commands"]),
                "heldoutCommands": len(heldout_commands),
                "symmetryCommands": len(symmetry_commands),
                "strategicCommands": len(strategic_commands),
                "headToHeadCommands": len(head_to_head),
            },
            indent=2,
        )
    )
    return 0


def _load_final_run(
    *,
    run_root: Path,
    seed_id: str,
    arm_id: str,
    training_seed: int,
    augmentation_mode: str,
    checkpoint_step: int,
    source_manifest_sha256: str,
    implementation_sha256: str,
    value_replay_sha256: str,
    opponent_replay_sha256: str,
    value_warm_sha256: str,
    opponent_warm_sha256: str,
) -> FinalRunArtifacts:
    value_summary = _read_json_object(
        run_root / "value.summary.json", label=f"{seed_id}/{arm_id} value summary"
    )
    opponent_summary = _read_json_object(
        run_root / "opponent.summary.json",
        label=f"{seed_id}/{arm_id} opponent summary",
    )
    value_path, value_trace = _validate_training_summary(
        summary=value_summary,
        model="value",
        expected_step=checkpoint_step,
        expected_seed=training_seed,
        expected_augmentation=augmentation_mode,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_implementation_sha256=implementation_sha256,
        expected_replay_sha256=value_replay_sha256,
        expected_warm_sha256=value_warm_sha256,
    )
    opponent_path, opponent_trace = _validate_training_summary(
        summary=opponent_summary,
        model="opponent",
        expected_step=checkpoint_step,
        expected_seed=training_seed,
        expected_augmentation=augmentation_mode,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_implementation_sha256=implementation_sha256,
        expected_replay_sha256=opponent_replay_sha256,
        expected_warm_sha256=opponent_warm_sha256,
    )
    value_model, value_payload = load_value_checkpoint(path=value_path)
    opponent_model, opponent_payload = load_opponent_checkpoint(path=opponent_path)
    del value_model, opponent_model
    _validate_checkpoint_metadata(
        payload=value_payload,
        model="value",
        expected_step=checkpoint_step,
        expected_trace=value_trace,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_implementation_sha256=implementation_sha256,
        expected_replay_sha256=value_replay_sha256,
    )
    _validate_checkpoint_metadata(
        payload=opponent_payload,
        model="opponent",
        expected_step=checkpoint_step,
        expected_trace=opponent_trace,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_implementation_sha256=implementation_sha256,
        expected_replay_sha256=opponent_replay_sha256,
    )
    return FinalRunArtifacts(
        seed_id=seed_id,
        arm_id=arm_id,
        value_checkpoint=value_path,
        value_sha256=sha256_file(value_path),
        value_sampling_trace_sha256=value_trace,
        opponent_checkpoint=opponent_path,
        opponent_sha256=sha256_file(opponent_path),
        opponent_sampling_trace_sha256=opponent_trace,
    )


def _validate_training_summary(
    *,
    summary: dict[str, Any],
    model: Literal["value", "opponent"],
    expected_step: int,
    expected_seed: int,
    expected_augmentation: str,
    expected_source_manifest_sha256: str,
    expected_implementation_sha256: str,
    expected_replay_sha256: str,
    expected_warm_sha256: str,
) -> tuple[Path, str]:
    config = _require_object(summary, "config")
    _expect_equal(f"{model} summary steps", config.get("steps"), expected_step)
    _expect_equal(f"{model} summary seed", config.get("seed"), expected_seed)
    _expect_equal(
        f"{model} summary augmentation",
        config.get("districtAugmentation"),
        expected_augmentation,
    )
    provenance = _require_object(summary, "provenance")
    _expect_equal(
        f"{model} source manifest SHA-256",
        provenance.get("experimentManifestSha256"),
        expected_source_manifest_sha256,
    )
    _expect_equal(
        f"{model} implementation SHA-256",
        provenance.get("implementationSha256"),
        expected_implementation_sha256,
    )
    replay_key = "valueReplayContentSha256" if model == "value" else "opponentReplayContentSha256"
    warm_key = "warmStartValueSha256" if model == "value" else "warmStartOpponentSha256"
    _expect_equal(
        f"{model} replay content SHA-256",
        provenance.get(replay_key),
        expected_replay_sha256,
    )
    _expect_equal(
        f"{model} warm-start SHA-256",
        provenance.get(warm_key),
        expected_warm_sha256,
    )

    results = _require_object(summary, "results")
    trace_key = "valueSamplingTraceSha256" if model == "value" else "opponentSamplingTraceSha256"
    trace = _require_str(results, trace_key)
    checkpoints = results.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise SystemExit(f"{model} summary checkpoints must be a list.")
    matching = [
        row
        for row in checkpoints
        if isinstance(row, dict)
        and row.get("step") == expected_step
        and isinstance(row.get(model), str)
    ]
    if len(matching) != 1:
        raise SystemExit(
            f"{model} summary must contain exactly one checkpoint at step {expected_step}."
        )
    path = Path(str(matching[0][model])).resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{model} final checkpoint not found: {path}")
    return path, trace


def _validate_checkpoint_metadata(
    *,
    payload: dict[str, Any],
    model: Literal["value", "opponent"],
    expected_step: int,
    expected_trace: str,
    expected_source_manifest_sha256: str,
    expected_implementation_sha256: str,
    expected_replay_sha256: str,
) -> None:
    metadata = _require_object(payload, "metadata")
    _expect_equal(f"{model} checkpoint step", metadata.get("step"), expected_step)
    trace_key = "valueSamplingTraceSha256" if model == "value" else "opponentSamplingTraceSha256"
    _expect_equal(
        f"{model} checkpoint sampling trace",
        metadata.get(trace_key),
        expected_trace,
    )
    provenance = _require_object(metadata, "provenance")
    _expect_equal(
        f"{model} checkpoint source manifest",
        provenance.get("experimentManifestSha256"),
        expected_source_manifest_sha256,
    )
    _expect_equal(
        f"{model} checkpoint implementation",
        provenance.get("implementationSha256"),
        expected_implementation_sha256,
    )
    replay_key = "valueReplayContentSha256" if model == "value" else "opponentReplayContentSha256"
    _expect_equal(
        f"{model} checkpoint replay content",
        provenance.get(replay_key),
        expected_replay_sha256,
    )


def _verify_matched_sampling_traces(
    runs: dict[tuple[str, str], FinalRunArtifacts],
) -> None:
    for seed_id in sorted({seed_id for seed_id, _arm_id in runs}):
        control = runs[(seed_id, "continued-control")]
        augmented = runs[(seed_id, "s4-augmented")]
        _expect_equal(
            f"{seed_id} matched value sampling trace",
            augmented.value_sampling_trace_sha256,
            control.value_sampling_trace_sha256,
        )
        _expect_equal(
            f"{seed_id} matched opponent sampling trace",
            augmented.opponent_sampling_trace_sha256,
            control.opponent_sampling_trace_sha256,
        )


def _sampling_trace_payload(
    runs: dict[tuple[str, str], FinalRunArtifacts],
) -> list[dict[str, str]]:
    rows = []
    for seed_id in sorted({seed_id for seed_id, _arm_id in runs}):
        control = runs[(seed_id, "continued-control")]
        rows.append(
            {
                "seedId": seed_id,
                "valueSamplingTraceSha256": control.value_sampling_trace_sha256,
                "opponentSamplingTraceSha256": control.opponent_sampling_trace_sha256,
                "controlAndAugmentedMatch": "true",
            }
        )
    return rows


def _prepare_export_commands(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    runs: dict[tuple[str, str], FinalRunArtifacts],
    primary_seed_id: str,
    replication_seed_id: str,
) -> dict[str, Any]:
    pack_config = _require_object(manifest, "experimentalPacks")
    pack_ids = _require_object(pack_config, "packIds")
    output_root = _resolve_path(repo_root, Path(_require_str(pack_config, "outputRoot")))
    manifest_prefix = _require_str(pack_config, "manifestPathPrefix")
    warm = _require_object(manifest, "warmStart")
    incumbent_value = _resolve_path(
        repo_root, Path(_require_str(_require_object(warm, "value"), "path"))
    )
    incumbent_opponent = _resolve_path(
        repo_root, Path(_require_str(_require_object(warm, "opponent"), "path"))
    )
    primary_control = runs[(primary_seed_id, "continued-control")]
    primary_augmented = runs[(primary_seed_id, "s4-augmented")]
    replication_control = runs[(replication_seed_id, "continued-control")]
    replication_augmented = runs[(replication_seed_id, "s4-augmented")]
    roles = {
        "incumbent": (incumbent_value, incumbent_opponent),
        "pilotAControl": (
            primary_control.value_checkpoint,
            primary_control.opponent_checkpoint,
        ),
        "pilotAAugmented": (
            primary_augmented.value_checkpoint,
            primary_augmented.opponent_checkpoint,
        ),
        "pilotAControlValueAugmentedOpponent": (
            primary_control.value_checkpoint,
            primary_augmented.opponent_checkpoint,
        ),
        "pilotAAugmentedValueControlOpponent": (
            primary_augmented.value_checkpoint,
            primary_control.opponent_checkpoint,
        ),
        "pilotBControl": (
            replication_control.value_checkpoint,
            replication_control.opponent_checkpoint,
        ),
        "pilotBAugmented": (
            replication_augmented.value_checkpoint,
            replication_augmented.opponent_checkpoint,
        ),
    }
    commands = []
    resolved_ids: dict[str, str] = {}
    for role, (value_path, opponent_path) in roles.items():
        pack_id = _require_str(pack_ids, role)
        resolved_ids[role] = pack_id
        commands.append(
            {
                "id": f"export-{role}",
                "role": role,
                "packId": pack_id,
                "command": [
                    str(Path(sys.executable).resolve()),
                    "-m",
                    "scripts.export_browser_td_root_pack",
                    "--value-checkpoint",
                    str(value_path),
                    "--opponent-checkpoint",
                    str(opponent_path),
                    "--output-root",
                    str(output_root),
                    "--manifest-path-prefix",
                    manifest_prefix,
                    "--pack-id",
                    pack_id,
                    "--label",
                    f"District S4 Pilot {role}",
                    "--overwrite",
                    "--no-set-default",
                ],
            }
        )
    return {
        "commands": commands,
        "packIds": resolved_ids,
        "modelIndexPath": _require_str(pack_config, "modelIndexPath"),
    }


def _prepare_holdout_commands(
    *,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
    runs: dict[tuple[str, str], FinalRunArtifacts],
    python_bin: Path,
    output_dir: Path,
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    path_lists = _require_object(_require_object(resolved, "split"), "pathLists")
    value_list = _require_str(_require_object(path_lists, "validationValue"), "path")
    opponent_list = _require_str(_require_object(path_lists, "validationOpponent"), "path")
    expected = _require_object(
        _require_object(_require_object(manifest, "replay"), "split"),
        "expectedValidationContentSha256",
    )
    gamma = _require_number(
        _require_object(_require_object(manifest, "evaluation"), "heldoutReplay"),
        "gamma",
    )
    commands = []
    for (seed_id, arm_id), run in sorted(runs.items()):
        output = output_dir / seed_id / f"{arm_id}.json"
        commands.append(
            {
                "id": f"heldout-{seed_id}-{arm_id}",
                "seedId": seed_id,
                "armId": arm_id,
                "command": [
                    str(python_bin),
                    "-m",
                    "scripts.evaluate_td_replay_holdout",
                    "--value-checkpoint",
                    str(run.value_checkpoint),
                    "--opponent-checkpoint",
                    str(run.opponent_checkpoint),
                    "--value-replay-list",
                    value_list,
                    "--opponent-replay-list",
                    opponent_list,
                    "--expected-value-replay-content-sha256",
                    _require_str(expected, "value"),
                    "--expected-opponent-replay-content-sha256",
                    _require_str(expected, "opponent"),
                    "--expected-value-checkpoint-sha256",
                    run.value_sha256,
                    "--expected-opponent-checkpoint-sha256",
                    run.opponent_sha256,
                    "--expected-checkpoint-step",
                    str(checkpoint_step),
                    "--gamma",
                    str(gamma),
                    "--num-threads",
                    str(_require_int(_require_object(manifest, "training"), "numThreads")),
                    "--output",
                    str(output),
                ],
            }
        )
    return commands


def _prepare_symmetry_commands(
    *,
    manifest: dict[str, Any],
    resolved: dict[str, Any],
    packs: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    direct = _require_object(_require_object(manifest, "evaluation"), "directSymmetry")
    path_lists = _require_object(_require_object(resolved, "split"), "pathLists")
    replay_list = _require_str(_require_object(path_lists, "validationOpponent"), "path")
    roles = (
        "pilotAControl",
        "pilotAAugmented",
        "pilotBControl",
        "pilotBAugmented",
    )
    commands = []
    for role in roles:
        pack_id = _require_str(_require_object(packs, "packIds"), role)
        commands.append(
            {
                "id": f"symmetry-{role}",
                "role": role,
                "command": [
                    "yarn",
                    "bot:eval",
                    "td-symmetry",
                    "--replay-list",
                    replay_list,
                    "--sample-size",
                    str(_require_int(direct, "sampleSize")),
                    "--sampling-seed",
                    _require_str(direct, "samplingSeed"),
                    "--pack-id",
                    pack_id,
                    "--model-index-path",
                    _require_str(packs, "modelIndexPath"),
                    "--out-dir",
                    str(output_dir / role),
                    "--progress-interval-seconds",
                    "30",
                ],
            }
        )
    return commands


def _prepare_strategic_commands(
    *,
    manifest: dict[str, Any],
    packs: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    evaluation = _require_object(manifest, "evaluation")
    development = _inclusive_range(
        _require_int_list(
            _require_object(evaluation, "strategicDevelopment"),
            "repetitionsInclusive",
        )
    )
    reserved = _inclusive_range(
        _require_int_list(
            _require_object(evaluation, "strategicReserved"),
            "repetitionsInclusive",
        )
    )
    pack_ids = _require_object(packs, "packIds")
    commands = []
    for role in (
        "pilotAControl",
        "pilotAAugmented",
        "pilotBControl",
        "pilotBAugmented",
    ):
        commands.append(
            _strategic_command(
                command_id=f"strategic-development-{role}",
                pack_id=_require_str(pack_ids, role),
                model_index_path=_require_str(packs, "modelIndexPath"),
                repetition_start=development[0],
                repetitions=len(development),
                output_dir=output_dir / "development" / role,
                candidate_selection_allowed=True,
            )
        )
    commands.append(
        _strategic_command(
            command_id="strategic-reserved-primary-augmented",
            pack_id=_require_str(pack_ids, "pilotAAugmented"),
            model_index_path=_require_str(packs, "modelIndexPath"),
            repetition_start=reserved[0],
            repetitions=len(reserved),
            output_dir=output_dir / "reserved" / "pilotAAugmented",
            candidate_selection_allowed=False,
        )
    )
    return commands


def _strategic_command(
    *,
    command_id: str,
    pack_id: str,
    model_index_path: str,
    repetition_start: int,
    repetitions: int,
    output_dir: Path,
    candidate_selection_allowed: bool,
) -> dict[str, Any]:
    return {
        "id": command_id,
        "packId": pack_id,
        "candidateSelectionAllowed": candidate_selection_allowed,
        "command": [
            "yarn",
            "bot:eval",
            "strategic-positions",
            "--repetitions",
            str(repetitions),
            "--start-repetition",
            str(repetition_start),
            "--variants",
            "heuristic-v2-direct,rollout-search-v2-hard,td-root-search-v2-medium",
            "--model-index-path",
            model_index_path,
            "--pack-id",
            pack_id,
            "--out-dir",
            str(output_dir),
        ],
    }


def _prepare_head_to_head(
    *,
    manifest: dict[str, Any],
    packs: dict[str, Any],
    config_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    full_games = _require_object(_require_object(manifest, "evaluation"), "fullGames")
    pack_ids = _require_object(packs, "packIds")
    comparisons = (
        ("primary-augmented-vs-control", "pilotAAugmented", "pilotAControl"),
        ("primary-augmented-vs-incumbent", "pilotAAugmented", "incumbent"),
        ("primary-control-vs-incumbent-diagnostic", "pilotAControl", "incumbent"),
        ("replication-augmented-vs-control", "pilotBAugmented", "pilotBControl"),
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for comparison_id, candidate_role, opponent_role in comparisons:
        candidate_id = _require_str(pack_ids, candidate_role)
        opponent_id = _require_str(pack_ids, opponent_role)
        config = {
            "schemaVersion": 1,
            "runLabel": comparison_id,
            "seedPrefix": f"{_require_str(full_games, 'seedPrefix')}-{comparison_id}",
            "gamesPerSide": _require_int(full_games, "gamesPerSide"),
            "candidate": _td_root_bot_spec(
                bot_id=f"candidate-{candidate_id}",
                pack_id=candidate_id,
                model_index_path=_require_str(packs, "modelIndexPath"),
                full_games=full_games,
            ),
            "opponent": _td_root_bot_spec(
                bot_id=f"opponent-{opponent_id}",
                pack_id=opponent_id,
                model_index_path=_require_str(packs, "modelIndexPath"),
                full_games=full_games,
            ),
        }
        config_path = config_dir / f"{comparison_id}.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        prepared.append(
            {
                "id": comparison_id,
                "candidateRole": candidate_role,
                "opponentRole": opponent_role,
                "config": str(config_path),
                "configSha256": sha256_file(config_path),
                "command": [
                    "yarn",
                    "bot:eval",
                    "head-to-head",
                    "--config",
                    str(config_path),
                    "--out-dir",
                    str(output_dir / comparison_id),
                    "--workers",
                    str(_require_int(full_games, "workers")),
                    "--progress-interval-seconds",
                    "30",
                ],
            }
        )
    return prepared


def _td_root_bot_spec(
    *,
    bot_id: str,
    pack_id: str,
    model_index_path: str,
    full_games: dict[str, Any],
) -> dict[str, Any]:
    spec = _require_object(full_games, "botSpec")
    guidance = _require_object(spec, "guidance")
    return {
        "id": bot_id,
        "kind": "td-root-search",
        "modelIndexPath": f"{model_index_path}?tdPackId={pack_id}",
        "guidance": {
            "root": _require_str(guidance, "root"),
            "rollout": _require_str(guidance, "rollout"),
            "leaf": _require_str(guidance, "leaf"),
        },
        "config": {
            "worlds": _require_int(spec, "worlds"),
            "rollouts": _require_int(spec, "rollouts"),
            "depth": _require_int(spec, "depth"),
            "maxRootActions": _require_int(spec, "maxRootActions"),
            "rolloutEpsilon": _require_number(spec, "rolloutEpsilon"),
        },
    }


def _inclusive_range(bounds: list[int]) -> list[int]:
    if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
        raise SystemExit("Inclusive repetition range must contain [start, end].")
    return list(range(bounds[0], bounds[1] + 1))


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
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise SystemExit(f"Expected non-empty object list at {key}.")
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
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise SystemExit(f"Expected integer list at {key}.")
    return [int(item) for item in value]


def _expect_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{label} mismatch: expected={expected!r} actual={actual!r}")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_supported_runtime() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python 3.12+ is required.")
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Run this script from the project virtual environment (.venv).")


if __name__ == "__main__":
    raise SystemExit(main())
