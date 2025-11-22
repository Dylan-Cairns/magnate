from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from trainer.types import PlayerId

from .models import OpponentModel, ValueNet
from .targets import td_lambda_targets
from .types import OpponentSample, ValueTransition

TD_VALUE_TARGET_MODE = "td0"
TD_VALUE_TARGET_MODE_TD0 = "td0"
TD_VALUE_TARGET_MODE_TD_LAMBDA = "td-lambda"
TD_VALUE_TARGET_MODES = frozenset((TD_VALUE_TARGET_MODE_TD0, TD_VALUE_TARGET_MODE_TD_LAMBDA))

SequenceKey = Tuple[str, PlayerId]


@dataclass(frozen=True)
class TDTrainConfig:
    gamma: float = 0.995
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    target_sync_interval: int = 200
    use_huber_loss: bool = True
    value_target_mode: str = TD_VALUE_TARGET_MODE_TD0
    td_lambda: float = 0.7


@dataclass(frozen=True)
class TDTrainStepSummary:
    step: int
    loss: float
    prediction_mean: float
    target_mean: float
    target_synced: bool


@dataclass(frozen=True)
class OpponentTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class OpponentTrainStepSummary:
    step: int
    loss: float
    accuracy: float


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
    target_mode: str = TD_VALUE_TARGET_MODE_TD0,
    td_lambda: float = 0.7,
    sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]] | None = None,
) -> tuple[float, float, float]:
    if len(transitions) == 0:
        raise ValueError("transitions must not be empty.")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1].")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0.")

    if target_mode not in TD_VALUE_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {sorted(TD_VALUE_TARGET_MODES)}, got {target_mode!r}."
        )
    if td_lambda < 0.0 or td_lambda > 1.0:
        raise ValueError("td_lambda must be in [0, 1].")
    if target_mode == TD_VALUE_TARGET_MODE_TD_LAMBDA and sequence_index is None:
        raise ValueError("sequence_index is required for td-lambda target mode.")

    observation_dim = len(transitions[0].observation)
    if observation_dim == 0:
        raise ValueError("transition observations must be non-empty.")
    for transition in transitions:
        if len(transition.observation) != observation_dim:
            raise ValueError("All transition observations must have the same dimension.")
        if transition.done and transition.next_observation is not None:
            raise ValueError(
                "Terminal transition must have next_observation=None."
            )
        if (not transition.done) and transition.next_observation is None:
            raise ValueError(
                "Non-terminal transition must include next_observation."
            )
        if (
            transition.next_observation is not None
            and len(transition.next_observation) != observation_dim
        ):
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
        if target_mode == TD_VALUE_TARGET_MODE_TD0:
            next_values = target_model(next_obs_tensor)
            targets = rewards + (gamma * (1.0 - done_mask) * next_values)
        else:
            targets = _td_lambda_batch_targets(
                target_model=target_model,
                transitions=transitions,
                sequence_index=_require_sequence_index(sequence_index),
                gamma=gamma,
                td_lambda=td_lambda,
                observation_dim=observation_dim,
            )

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


def build_value_sequence_index(
    *,
    transitions: Sequence[ValueTransition],
) -> Dict[SequenceKey, tuple[ValueTransition, ...]]:
    if len(transitions) == 0:
        raise ValueError("transitions must not be empty.")

    grouped: Dict[SequenceKey, Dict[int, ValueTransition]] = {}
    for transition in transitions:
        if transition.episode_id is None or transition.timestep is None:
            raise ValueError(
                "td-lambda requires sequence-aware replay rows with episode_id and timestep."
            )
        if transition.timestep < 0:
            raise ValueError("transition timestep must be >= 0.")

        key: SequenceKey = (transition.episode_id, transition.player_id)
        by_step = grouped.setdefault(key, {})
        if transition.timestep in by_step:
            raise ValueError(
                "Duplicate timestep in sequence-aware replay rows. "
                f"episodeId={transition.episode_id!r} playerId={transition.player_id} "
                f"timestep={transition.timestep}"
            )
        by_step[transition.timestep] = transition

    out: Dict[SequenceKey, tuple[ValueTransition, ...]] = {}
    for key, by_step in grouped.items():
        expected = 0
        ordered: list[ValueTransition] = []
        while expected in by_step:
            ordered.append(by_step[expected])
            expected += 1
        if len(ordered) != len(by_step):
            raise ValueError(
                "Sequence timesteps must be contiguous from 0 for td-lambda. "
                f"episodeId={key[0]!r} playerId={key[1]}"
            )
        done_indices = [index for index, transition in enumerate(ordered) if transition.done]
        if len(done_indices) != 1 or done_indices[0] != (len(ordered) - 1):
            raise ValueError(
                "Each sequence must have exactly one terminal row at the final timestep "
                "for td-lambda training. "
                f"episodeId={key[0]!r} playerId={key[1]}"
            )
        out[key] = tuple(ordered)
    return out


def _td_lambda_batch_targets(
    *,
    target_model: ValueNet,
    transitions: Sequence[ValueTransition],
    sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]],
    gamma: float,
    td_lambda: float,
    observation_dim: int,
) -> torch.Tensor:
    sequence_targets: Dict[SequenceKey, list[float]] = {}

    for transition in transitions:
        key = _require_sequence_key(transition)
        if key in sequence_targets:
            continue
        sequence = sequence_index.get(key)
        if sequence is None:
            raise ValueError(
                "Missing sequence for td-lambda transition. "
                f"episodeId={key[0]!r} playerId={key[1]}"
            )
        next_obs_vectors = [
            list(item.next_observation) if item.next_observation is not None else [0.0] * observation_dim
            for item in sequence
        ]
        next_obs_tensor = torch.tensor(next_obs_vectors, dtype=torch.float32)
        next_values = target_model(next_obs_tensor).tolist()
        sequence_targets[key] = td_lambda_targets(
            rewards=[float(item.reward) for item in sequence],
            dones=[bool(item.done) for item in sequence],
            next_values=[float(value) for value in next_values],
            gamma=gamma,
            lambda_=td_lambda,
        )

    batch_targets: list[float] = []
    for transition in transitions:
        key = _require_sequence_key(transition)
        timestep = _require_timestep(transition)
        targets = sequence_targets[key]
        if timestep >= len(targets):
            raise ValueError(
                "Transition timestep is out of range for td-lambda sequence targets. "
                f"episodeId={key[0]!r} playerId={key[1]} timestep={timestep} "
                f"sequenceLen={len(targets)}"
            )
        batch_targets.append(float(targets[timestep]))
    return torch.tensor(batch_targets, dtype=torch.float32)


def _require_sequence_index(
    sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]] | None,
) -> Mapping[SequenceKey, Sequence[ValueTransition]]:
    if sequence_index is None:
        raise ValueError("sequence_index is required for td-lambda target mode.")
    return sequence_index


def _require_sequence_key(transition: ValueTransition) -> SequenceKey:
    if transition.episode_id is None:
        raise ValueError("transition.episode_id is required for td-lambda target mode.")
    return (transition.episode_id, transition.player_id)


def _require_timestep(transition: ValueTransition) -> int:
    if transition.timestep is None:
        raise ValueError("transition.timestep is required for td-lambda target mode.")
    return int(transition.timestep)


def train_opponent_batch(
    *,
    model: OpponentModel,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[OpponentSample],
    max_grad_norm: float,
) -> tuple[float, float]:
    if len(samples) == 0:
        raise ValueError("samples must not be empty.")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be > 0.")

    observation_dim = len(samples[0].observation)
    if observation_dim == 0:
        raise ValueError("sample observations must be non-empty.")

    total_loss = torch.tensor(0.0, dtype=torch.float32)
    correct = 0
    for sample in samples:
        if len(sample.observation) != observation_dim:
            raise ValueError("All sample observations must have the same dimension.")
        if len(sample.action_features) == 0:
            raise ValueError("Each sample must include at least one candidate action.")
        if sample.action_index < 0 or sample.action_index >= len(sample.action_features):
            raise ValueError(
                "sample action_index is out of bounds for action_features length."
            )
        feature_dim = len(sample.action_features[0])
        if feature_dim == 0:
            raise ValueError("Action feature vectors must be non-empty.")
        for action_features in sample.action_features:
            if len(action_features) != feature_dim:
                raise ValueError("All action feature vectors in a sample must share dimension.")

        observation = torch.tensor(sample.observation, dtype=torch.float32)
        action_features_tensor = torch.tensor(sample.action_features, dtype=torch.float32)
        logits = model.logits_tensor(observation, action_features_tensor).unsqueeze(0)
        target = torch.tensor([sample.action_index], dtype=torch.long)
        sample_loss = F.cross_entropy(logits, target)
        total_loss = total_loss + sample_loss
        predicted = int(torch.argmax(logits, dim=1).item())
        if predicted == sample.action_index:
            correct += 1

    loss = total_loss / float(len(samples))
    accuracy = correct / float(len(samples))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.item()), float(accuracy)


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
        if config.value_target_mode not in TD_VALUE_TARGET_MODES:
            raise ValueError(
                f"value_target_mode must be one of {sorted(TD_VALUE_TARGET_MODES)}."
            )
        if config.td_lambda < 0.0 or config.td_lambda > 1.0:
            raise ValueError("td_lambda must be in [0, 1].")
        hard_sync(target_model=self._target_model, source_model=self._model)

    @property
    def step(self) -> int:
        return self._step

    def train_batch(
        self,
        *,
        transitions: Sequence[ValueTransition],
        sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]] | None = None,
    ) -> TDTrainStepSummary:
        loss, prediction_mean, target_mean = train_value_batch(
            model=self._model,
            target_model=self._target_model,
            optimizer=self._optimizer,
            transitions=transitions,
            gamma=self._config.gamma,
            max_grad_norm=self._config.max_grad_norm,
            use_huber_loss=self._config.use_huber_loss,
            target_mode=self._config.value_target_mode,
            td_lambda=self._config.td_lambda,
            sequence_index=sequence_index,
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


class TDOpponentTrainer:
    def __init__(
        self,
        *,
        model: OpponentModel,
        optimizer: torch.optim.Optimizer,
        config: OpponentTrainConfig,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._config = config
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def train_batch(self, *, samples: Sequence[OpponentSample]) -> OpponentTrainStepSummary:
        loss, accuracy = train_opponent_batch(
            model=self._model,
            optimizer=self._optimizer,
            samples=samples,
            max_grad_norm=self._config.max_grad_norm,
        )
        self._step += 1
        return OpponentTrainStepSummary(
            step=self._step,
            loss=loss,
            accuracy=accuracy,
        )
