from __future__ import annotations

from typing import Dict, Sequence

import torch

from .models import OpponentModel, ValueNet
from .train import build_value_sequence_index
from .types import OpponentSample, ValueTransition


def evaluate_value_holdout(
    *,
    model: ValueNet,
    transitions: Sequence[ValueTransition],
    gamma: float,
    batch_size: int = 4096,
) -> Dict[str, float | int]:
    if not transitions:
        raise ValueError("Value holdout transitions must not be empty.")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    sequence_index = build_value_sequence_index(transitions=transitions)
    targets_by_key: dict[tuple[str, str, int], float] = {}
    for (episode_id, player_id), sequence in sequence_index.items():
        return_value = 0.0
        for transition in reversed(sequence):
            if transition.timestep is None:
                raise ValueError("Value holdout transition is missing timestep.")
            return_value = float(transition.reward) + (
                0.0 if transition.done else gamma * return_value
            )
            targets_by_key[(episode_id, player_id, transition.timestep)] = return_value

    targets: list[float] = []
    for transition in transitions:
        if transition.episode_id is None or transition.timestep is None:
            raise ValueError(
                "Value holdout requires sequence-aware episode_id and timestep fields."
            )
        targets.append(
            targets_by_key[(transition.episode_id, transition.player_id, transition.timestep)]
        )

    predictions: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(transitions), batch_size):
            batch = transitions[start : start + batch_size]
            observations = torch.tensor(
                [list(transition.observation) for transition in batch],
                dtype=torch.float32,
            )
            predictions.extend(float(value) for value in model(observations).tolist())

    count = len(targets)
    errors = [prediction - target for prediction, target in zip(predictions, targets)]
    return {
        "rows": count,
        "sequences": len(sequence_index),
        "monteCarloMse": sum(error * error for error in errors) / count,
        "monteCarloMae": sum(abs(error) for error in errors) / count,
        "meanPredictionBias": sum(errors) / count,
        "predictionMean": sum(predictions) / count,
        "targetMean": sum(targets) / count,
    }


def evaluate_opponent_holdout(
    *,
    model: OpponentModel,
    samples: Sequence[OpponentSample],
) -> Dict[str, float | int]:
    if not samples:
        raise ValueError("Opponent holdout samples must not be empty.")

    soft_cross_entropy = 0.0
    soft_kl = 0.0
    hard_nll = 0.0
    teacher_top_matches = 0
    selected_action_matches = 0
    candidate_count = 0
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            logits = model.logits_tensor(
                torch.tensor(sample.observation, dtype=torch.float32),
                torch.tensor(sample.action_features, dtype=torch.float32),
            )
            log_probabilities = torch.log_softmax(logits, dim=0)
            target = torch.tensor(sample.action_probs, dtype=torch.float32)
            cross_entropy = float(-(target * log_probabilities).sum().item())
            target_entropy = float(
                -(target * torch.log(target.clamp_min(torch.finfo(target.dtype).tiny))).sum().item()
            )
            model_top = int(torch.argmax(logits).item())
            teacher_top = int(torch.argmax(target).item())

            soft_cross_entropy += cross_entropy
            soft_kl += max(0.0, cross_entropy - target_entropy)
            hard_nll += float(-log_probabilities[sample.action_index].item())
            teacher_top_matches += int(model_top == teacher_top)
            selected_action_matches += int(model_top == sample.action_index)
            candidate_count += len(sample.action_features)

    count = len(samples)
    return {
        "rows": count,
        "softTargetCrossEntropy": soft_cross_entropy / count,
        "softTargetKl": soft_kl / count,
        "hardSelectedActionNll": hard_nll / count,
        "teacherTopActionAgreement": teacher_top_matches / count,
        "selectedActionAccuracy": selected_action_matches / count,
        "meanCandidateCount": candidate_count / count,
    }
