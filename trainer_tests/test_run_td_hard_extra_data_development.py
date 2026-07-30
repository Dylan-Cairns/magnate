from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import run_td_hard_extra_data_development as development


def _selection_result(
    *,
    step: int,
    mse: float,
    mae: float,
    cross_entropy: float,
    kl: float,
) -> dict[str, object]:
    return {
        "provenance": {"checkpointStep": step},
        "value": {
            "monteCarloMse": mse,
            "monteCarloMae": mae,
        },
        "opponent": {
            "softTargetCrossEntropy": cross_entropy,
            "softTargetKl": kl,
        },
    }


def test_select_checkpoint_step_uses_frozen_rank_sum_and_tie_breakers() -> None:
    results = [
        _selection_result(
            step=step,
            mse=float(index),
            mae=0.1 if step == 6_000 else 0.2,
            cross_entropy=float(11 - index),
            kl=0.3,
        )
        for index, step in enumerate(development.CHECKPOINT_STEPS, start=1)
    ]

    selection = development._select_checkpoint_step(results)

    assert selection["selectedStep"] == 6_000
    assert all(row["rankSum"] == 11 for row in selection["ranking"])


def test_select_checkpoint_step_rejects_missing_checkpoint() -> None:
    results = [
        _selection_result(
            step=step,
            mse=0.1,
            mae=0.1,
            cross_entropy=0.2,
            kl=0.2,
        )
        for step in development.CHECKPOINT_STEPS[:-1]
    ]

    with pytest.raises(SystemExit, match="Expected 10 selection results"):
        development._select_checkpoint_step(results)


def test_validate_evaluation_result_binds_frozen_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    result = {
        "schemaVersion": 1,
        "provenance": {
            "checkpointStep": 4_000,
            "valueCheckpointSha256": "value-checkpoint",
            "opponentCheckpointSha256": "opponent-checkpoint",
            "valueReplayContentSha256": "value-development",
            "opponentReplayContentSha256": "opponent-development",
            "replayKeyMode": development.REPLAY_KEY_MODE_RUN_QUALIFIED,
        },
        "value": {
            "rows": 18_083,
            "monteCarloMse": 0.1,
            "monteCarloMae": 0.2,
        },
        "opponent": {
            "rows": 18_083,
            "softTargetCrossEntropy": 0.3,
            "softTargetKl": 0.4,
            "teacherTopActionAgreement": 0.5,
        },
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    row = {
        "id": "step-04000",
        "checkpointStep": 4_000,
        "output": str(output),
        "command": [
            "python",
            "--expected-value-checkpoint-sha256",
            "value-checkpoint",
            "--expected-opponent-checkpoint-sha256",
            "opponent-checkpoint",
            "--expected-value-replay-content-sha256",
            "value-development",
            "--expected-opponent-replay-content-sha256",
            "opponent-development",
        ],
    }

    assert development._validate_evaluation_result(row=row) == result
