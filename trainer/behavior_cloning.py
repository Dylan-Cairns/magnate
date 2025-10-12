from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .types import DecisionSample

CHECKPOINT_TYPE = "magnate_behavior_cloning_v1"


@dataclass(frozen=True)
class BehaviorCloningConfig:
    epochs: int = 8
    learning_rate: float = 0.05
    l2: float = 1e-4
    seed: int = 0


@dataclass(frozen=True)
class DatasetMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class BehaviorCloningSummary:
    initial: DatasetMetrics
    final: DatasetMetrics
    history: List[DatasetMetrics]


@dataclass
class BehaviorCloningModel:
    observation_dim: int
    action_feature_dim: int
    obs_action_weights: List[List[float]]
    action_weights: List[float]

    @classmethod
    def zeros(cls, observation_dim: int, action_feature_dim: int) -> "BehaviorCloningModel":
        return cls(
            observation_dim=observation_dim,
            action_feature_dim=action_feature_dim,
            obs_action_weights=[[0.0 for _ in range(action_feature_dim)] for _ in range(observation_dim)],
            action_weights=[0.0 for _ in range(action_feature_dim)],
        )

    def copy(self) -> "BehaviorCloningModel":
        return BehaviorCloningModel(
            observation_dim=self.observation_dim,
            action_feature_dim=self.action_feature_dim,
            obs_action_weights=[row[:] for row in self.obs_action_weights],
            action_weights=self.action_weights[:],
        )

    def score_candidates(
        self,
        observation: Sequence[float],
        action_features: Sequence[Sequence[float]],
    ) -> List[float]:
        if len(observation) != self.observation_dim:
            raise ValueError(
                f"Observation length mismatch. expected={self.observation_dim}, actual={len(observation)}"
            )
        if not action_features:
            raise ValueError("At least one candidate action is required.")

        contextual = self.action_weights[:]
        for obs_index, obs_value in enumerate(observation):
            if obs_value == 0.0:
                continue
            row = self.obs_action_weights[obs_index]
            for feature_index in range(self.action_feature_dim):
                contextual[feature_index] += obs_value * row[feature_index]

        scores: List[float] = []
        for features in action_features:
            if len(features) != self.action_feature_dim:
                raise ValueError(
                    "Action feature length mismatch. "
                    f"expected={self.action_feature_dim}, actual={len(features)}"
                )
            score = 0.0
            for feature_index, value in enumerate(features):
                score += contextual[feature_index] * value
            scores.append(score)
        return scores

    def choose_action_index(
        self,
        observation: Sequence[float],
        action_features: Sequence[Sequence[float]],
    ) -> int:
        scores = self.score_candidates(observation, action_features)
        best_index = 0
        best_score = scores[0]
        for index, score in enumerate(scores[1:], start=1):
            if score > best_score:
                best_index = index
                best_score = score
        return best_index


def train_behavior_cloning(
    samples: Sequence[DecisionSample],
    config: BehaviorCloningConfig | None = None,
    initial_model: BehaviorCloningModel | None = None,
) -> Tuple[BehaviorCloningModel, BehaviorCloningSummary]:
    if not samples:
        raise ValueError("Behavior cloning requires at least one decision sample.")

    cfg = config or BehaviorCloningConfig()
    _validate_config(cfg)

    observation_dim, action_feature_dim = _infer_dimensions(samples)
    model = (initial_model.copy() if initial_model is not None else BehaviorCloningModel.zeros(observation_dim, action_feature_dim))
    _validate_model_dims(model, observation_dim, action_feature_dim)

    initial_metrics = evaluate_behavior_cloning(model, samples)
    history: List[DatasetMetrics] = []

    rng = random.Random(cfg.seed)
    order = list(range(len(samples)))

    for _ in range(cfg.epochs):
        rng.shuffle(order)
        for sample_index in order:
            _sgd_update(model, samples[sample_index], cfg)
        history.append(evaluate_behavior_cloning(model, samples))

    final_metrics = history[-1] if history else initial_metrics
    summary = BehaviorCloningSummary(
        initial=initial_metrics,
        final=final_metrics,
        history=history,
    )
    return model, summary


def evaluate_behavior_cloning(
    model: BehaviorCloningModel,
    samples: Sequence[DecisionSample],
) -> DatasetMetrics:
    if not samples:
        return DatasetMetrics(loss=0.0, accuracy=0.0)

    total_loss = 0.0
    correct = 0

    for sample in samples:
        _validate_sample(sample, model.observation_dim, model.action_feature_dim)
        scores = model.score_candidates(sample.observation, sample.action_features)
        probs = _softmax(scores)
        action_index = sample.action_index
        total_loss += -math.log(max(probs[action_index], 1e-12))

        predicted_index = 0
        predicted_score = scores[0]
        for index, score in enumerate(scores[1:], start=1):
            if score > predicted_score:
                predicted_index = index
                predicted_score = score
        if predicted_index == action_index:
            correct += 1

    count = float(len(samples))
    return DatasetMetrics(
        loss=total_loss / count,
        accuracy=correct / count,
    )


def save_behavior_cloning_checkpoint(
    model: BehaviorCloningModel,
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpointType": CHECKPOINT_TYPE,
        "observationDim": model.observation_dim,
        "actionFeatureDim": model.action_feature_dim,
        "obsActionWeights": model.obs_action_weights,
        "actionWeights": model.action_weights,
        "metadata": dict(metadata) if metadata is not None else {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_behavior_cloning_checkpoint(path: Path) -> BehaviorCloningModel:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Behavior cloning checkpoint must be a JSON object.")
    return _model_from_payload(raw)


def _model_from_payload(payload: Mapping[str, Any]) -> BehaviorCloningModel:
    checkpoint_type = str(payload.get("checkpointType", ""))
    if checkpoint_type != CHECKPOINT_TYPE:
        raise ValueError(
            f"Unsupported checkpoint type {checkpoint_type!r}; expected {CHECKPOINT_TYPE!r}."
        )

    observation_dim = _as_positive_int(payload.get("observationDim"), "observationDim")
    action_feature_dim = _as_positive_int(payload.get("actionFeatureDim"), "actionFeatureDim")

    obs_action_weights = payload.get("obsActionWeights")
    if not isinstance(obs_action_weights, list):
        raise ValueError("obsActionWeights must be a list.")
    if len(obs_action_weights) != observation_dim:
        raise ValueError(
            "obsActionWeights row count mismatch. "
            f"expected={observation_dim}, actual={len(obs_action_weights)}"
        )

    parsed_obs_action_weights: List[List[float]] = []
    for row in obs_action_weights:
        if not isinstance(row, list):
            raise ValueError("obsActionWeights rows must be lists.")
        if len(row) != action_feature_dim:
            raise ValueError(
                "obsActionWeights column count mismatch. "
                f"expected={action_feature_dim}, actual={len(row)}"
            )
        parsed_obs_action_weights.append([_as_float(value, "obsActionWeights") for value in row])

    action_weights = payload.get("actionWeights")
    if not isinstance(action_weights, list):
        raise ValueError("actionWeights must be a list.")
    if len(action_weights) != action_feature_dim:
        raise ValueError(
            "actionWeights length mismatch. "
            f"expected={action_feature_dim}, actual={len(action_weights)}"
        )

    parsed_action_weights = [_as_float(value, "actionWeights") for value in action_weights]
    return BehaviorCloningModel(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        obs_action_weights=parsed_obs_action_weights,
        action_weights=parsed_action_weights,
    )


def _sgd_update(
    model: BehaviorCloningModel,
    sample: DecisionSample,
    config: BehaviorCloningConfig,
) -> None:
    _validate_sample(sample, model.observation_dim, model.action_feature_dim)
    scores = model.score_candidates(sample.observation, sample.action_features)
    probs = _softmax(scores)

    delta = [0.0 for _ in range(model.action_feature_dim)]
    for candidate_index, features in enumerate(sample.action_features):
        coeff = probs[candidate_index]
        if candidate_index == sample.action_index:
            coeff -= 1.0
        for feature_index, value in enumerate(features):
            delta[feature_index] += coeff * value

    learning_rate = config.learning_rate
    l2 = config.l2

    for feature_index in range(model.action_feature_dim):
        gradient = delta[feature_index] + (l2 * model.action_weights[feature_index])
        model.action_weights[feature_index] -= learning_rate * gradient

    for obs_index, obs_value in enumerate(sample.observation):
        row = model.obs_action_weights[obs_index]
        for feature_index in range(model.action_feature_dim):
            gradient = (obs_value * delta[feature_index]) + (l2 * row[feature_index])
            row[feature_index] -= learning_rate * gradient


def _softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        raise ValueError("Softmax requires at least one score.")
    maximum = max(scores)
    exps = [math.exp(score - maximum) for score in scores]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exps]


def _infer_dimensions(samples: Sequence[DecisionSample]) -> Tuple[int, int]:
    first = samples[0]
    observation_dim = len(first.observation)
    if observation_dim == 0:
        raise ValueError("Decision samples must include at least one observation feature.")

    if not first.action_features:
        raise ValueError("Decision samples must include at least one action candidate.")
    action_feature_dim = len(first.action_features[0])
    if action_feature_dim == 0:
        raise ValueError("Decision samples must include at least one action feature.")

    for sample in samples:
        _validate_sample(sample, observation_dim, action_feature_dim)
    return observation_dim, action_feature_dim


def _validate_sample(sample: DecisionSample, observation_dim: int, action_feature_dim: int) -> None:
    if len(sample.observation) != observation_dim:
        raise ValueError(
            "Observation length mismatch in sample. "
            f"expected={observation_dim}, actual={len(sample.observation)}"
        )
    if not sample.action_features:
        raise ValueError("Sample must include at least one action candidate.")
    for features in sample.action_features:
        if len(features) != action_feature_dim:
            raise ValueError(
                "Action feature length mismatch in sample. "
                f"expected={action_feature_dim}, actual={len(features)}"
            )
    if sample.action_index < 0 or sample.action_index >= len(sample.action_features):
        raise ValueError(
            "Action index out of bounds for sample. "
            f"index={sample.action_index}, candidateCount={len(sample.action_features)}"
        )


def _validate_model_dims(
    model: BehaviorCloningModel,
    observation_dim: int,
    action_feature_dim: int,
) -> None:
    if model.observation_dim != observation_dim:
        raise ValueError(
            "Initial model observation dimension mismatch. "
            f"expected={observation_dim}, actual={model.observation_dim}"
        )
    if model.action_feature_dim != action_feature_dim:
        raise ValueError(
            "Initial model action-feature dimension mismatch. "
            f"expected={action_feature_dim}, actual={model.action_feature_dim}"
        )


def _validate_config(config: BehaviorCloningConfig) -> None:
    if config.epochs < 0:
        raise ValueError("epochs must be >= 0.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.l2 < 0:
        raise ValueError("l2 must be >= 0.")


def _as_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} entries must be numeric.")
    if isinstance(value, (int, float)):
        as_float = float(value)
        if not math.isfinite(as_float):
            raise ValueError(f"{label} entries must be finite.")
        return as_float
    raise ValueError(f"{label} entries must be numeric.")
