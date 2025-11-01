from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ppo_model import CandidateActorCritic
from .types import DecisionSample


@dataclass(frozen=True)
class GuidanceConfig:
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    hidden_dim: int = 256
    seed: int = 0


@dataclass(frozen=True)
class GuidanceEpochSummary:
    epoch: int
    policy_loss: float
    value_loss: float
    entropy: float
    accuracy: float


@dataclass(frozen=True)
class GuidanceTrainingSummary:
    sample_count: int
    observation_dim: int
    action_feature_dim: int
    history: List[GuidanceEpochSummary]

    @property
    def final(self) -> GuidanceEpochSummary | None:
        if not self.history:
            return None
        return self.history[-1]


def train_guidance_model(
    samples: Sequence[DecisionSample],
    config: GuidanceConfig,
) -> Tuple[CandidateActorCritic, GuidanceTrainingSummary]:
    _validate_config(config)
    observation_dim, action_feature_dim = _infer_dims(samples)
    _validate_samples(samples, observation_dim=observation_dim, action_feature_dim=action_feature_dim)

    model = CandidateActorCritic(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_dim=config.hidden_dim,
    )
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rng = random.Random(config.seed)
    indices = list(range(len(samples)))
    history: List[GuidanceEpochSummary] = []

    for epoch in range(config.epochs):
        rng.shuffle(indices)
        policy_terms: List[float] = []
        value_terms: List[float] = []
        entropy_terms: List[float] = []
        correct = 0
        total = 0

        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            if not batch_indices:
                continue

            batch_policy: List[torch.Tensor] = []
            batch_value: List[torch.Tensor] = []
            batch_entropy: List[torch.Tensor] = []

            for index in batch_indices:
                sample = samples[index]
                observation = torch.tensor(sample.observation, dtype=torch.float32)
                action_features = torch.tensor(sample.action_features, dtype=torch.float32)
                logits = model.policy_logits_tensor(observation, action_features)
                target_index = torch.tensor(sample.action_index, dtype=torch.int64)
                policy_loss = F.cross_entropy(logits.unsqueeze(0), target_index.unsqueeze(0))
                batch_policy.append(policy_loss)

                value = model.value_tensor(observation)
                target_value = torch.tensor(sample.reward, dtype=torch.float32)
                value_loss = F.mse_loss(value, target_value)
                batch_value.append(value_loss)

                entropy = torch.distributions.Categorical(logits=logits).entropy()
                batch_entropy.append(entropy)

                predicted = int(torch.argmax(logits).item())
                if predicted == sample.action_index:
                    correct += 1
                total += 1

            policy_loss = torch.stack(batch_policy).mean()
            value_loss = torch.stack(batch_value).mean()
            entropy = torch.stack(batch_entropy).mean()
            loss = (
                policy_loss
                + (config.value_loss_coef * value_loss)
                - (config.entropy_coef * entropy)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            policy_terms.append(float(policy_loss.item()))
            value_terms.append(float(value_loss.item()))
            entropy_terms.append(float(entropy.item()))

        history.append(
            GuidanceEpochSummary(
                epoch=epoch + 1,
                policy_loss=(sum(policy_terms) / len(policy_terms)) if policy_terms else 0.0,
                value_loss=(sum(value_terms) / len(value_terms)) if value_terms else 0.0,
                entropy=(sum(entropy_terms) / len(entropy_terms)) if entropy_terms else 0.0,
                accuracy=(correct / total) if total > 0 else 0.0,
            )
        )

    return model, GuidanceTrainingSummary(
        sample_count=len(samples),
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        history=history,
    )


def _infer_dims(samples: Sequence[DecisionSample]) -> tuple[int, int]:
    if not samples:
        raise ValueError("Cannot train guidance model with zero samples.")
    first = samples[0]
    if not first.action_features:
        raise ValueError("Samples must include at least one legal action candidate.")
    return len(first.observation), len(first.action_features[0])


def _validate_samples(
    samples: Sequence[DecisionSample],
    observation_dim: int,
    action_feature_dim: int,
) -> None:
    for sample in samples:
        if len(sample.observation) != observation_dim:
            raise ValueError(
                "Sample observation length mismatch. "
                f"expected={observation_dim}, actual={len(sample.observation)}"
            )
        if not sample.action_features:
            raise ValueError("Sample action_features must not be empty.")
        if sample.action_index < 0 or sample.action_index >= len(sample.action_features):
            raise ValueError(
                "Sample action_index out of bounds. "
                f"index={sample.action_index}, candidates={len(sample.action_features)}"
            )
        for features in sample.action_features:
            if len(features) != action_feature_dim:
                raise ValueError(
                    "Sample action feature length mismatch. "
                    f"expected={action_feature_dim}, actual={len(features)}"
                )


def _validate_config(config: GuidanceConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if config.value_loss_coef < 0:
        raise ValueError("value_loss_coef must be >= 0.")
    if config.entropy_coef < 0:
        raise ValueError("entropy_coef must be >= 0.")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0.")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be > 0.")
