from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .models import ValueNet
from .types import ValueTransition


@dataclass(frozen=True)
class TDTrainConfig:
    gamma: float = 0.995
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    target_sync_interval: int = 200
    use_huber_loss: bool = True


@dataclass(frozen=True)
class TDTrainStepSummary:
    step: int
    loss: float
    prediction_mean: float
    target_mean: float
    target_synced: bool


def hard_sync(*, target_model: ValueNet, source_model: ValueNet) -> None:
    target_model.load_state_dict(source_model.state_dict())


def train_value_batch(
    *,
    model: ValueNet,
    target_model: ValueNet,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[ValueTransition],
    gamma: float,
    max_grad_norm: float,
    use_huber_loss: bool,
) -> tuple[float, float, float]:
    if len(transitions) == 0:
        raise ValueError("transitions must not be empty.")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0.")

    observation_dim = len(transitions[0].observation)
    if observation_dim == 0:
        raise ValueError("transition observations must be non-empty.")
    for transition in transitions:
        if len(transition.observation) != observation_dim:
            raise ValueError("All transition observations must have the same dimension.")
        if transition.next_observation is not None and len(transition.next_observation) != observation_dim:
            raise ValueError("All transition next_observation values must match observation dimension.")
    obs_tensor = torch.tensor([transition.observation for transition in transitions], dtype=torch.float32)
    next_obs_tensor = torch.tensor(
        [
            transition.next_observation
            if transition.next_observation is not None
            else [0.0] * observation_dim
            for transition in transitions
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor([float(transition.reward) for transition in transitions], dtype=torch.float32)
    done_mask = torch.tensor([1.0 if transition.done else 0.0 for transition in transitions], dtype=torch.float32)

    predictions = model(obs_tensor)
    with torch.no_grad():
        next_values = target_model(next_obs_tensor)
        targets = rewards + (gamma * (1.0 - done_mask) * next_values)

    if use_huber_loss:
        loss = F.smooth_l1_loss(predictions, targets)
    else:
        loss = F.mse_loss(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return (
        float(loss.item()),
        float(predictions.mean().item()),
        float(targets.mean().item()),
    )


class TDValueTrainer:
    def __init__(
        self,
        *,
        model: ValueNet,
        target_model: ValueNet,
        optimizer: torch.optim.Optimizer,
        config: TDTrainConfig,
    ) -> None:
        self._model = model
        self._target_model = target_model
        self._optimizer = optimizer
        self._config = config
        self._step = 0

        if config.target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be > 0.")
        hard_sync(target_model=self._target_model, source_model=self._model)

    @property
    def step(self) -> int:
        return self._step

    def train_batch(self, *, transitions: Sequence[ValueTransition]) -> TDTrainStepSummary:
        loss, prediction_mean, target_mean = train_value_batch(
            model=self._model,
            target_model=self._target_model,
            optimizer=self._optimizer,
            transitions=transitions,
            gamma=self._config.gamma,
            max_grad_norm=self._config.max_grad_norm,
            use_huber_loss=self._config.use_huber_loss,
        )
        self._step += 1
        synced = (self._step % self._config.target_sync_interval) == 0
        if synced:
            hard_sync(target_model=self._target_model, source_model=self._model)
        return TDTrainStepSummary(
            step=self._step,
            loss=loss,
            prediction_mean=prediction_mean,
            target_mean=target_mean,
            target_synced=synced,
        )
