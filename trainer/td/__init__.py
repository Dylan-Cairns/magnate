"""TD/Keldon training primitives for Magnate."""

from .checkpoint import (
    TD_OPPONENT_CHECKPOINT_TYPE,
    TD_VALUE_CHECKPOINT_TYPE,
    load_opponent_checkpoint,
    load_value_checkpoint,
    save_opponent_checkpoint,
    save_value_checkpoint,
)
from .models import OpponentModel, ValueNet
from .replay import OpponentReplayBuffer, ValueReplayBuffer
from .self_play import (
    SelfPlayEpisode,
    collect_self_play_episode,
    collect_self_play_games,
    flatten_opponent_samples,
    flatten_value_transitions,
)
from .targets import n_step_bootstrap_targets, td_lambda_targets
from .train import (
    TDTrainConfig,
    TDTrainStepSummary,
    TDValueTrainer,
    hard_sync,
    train_value_batch,
)
from .types import OpponentSample, ValueTransition

__all__ = [
    "OpponentModel",
    "ValueNet",
    "OpponentReplayBuffer",
    "ValueReplayBuffer",
    "SelfPlayEpisode",
    "collect_self_play_episode",
    "collect_self_play_games",
    "flatten_opponent_samples",
    "flatten_value_transitions",
    "n_step_bootstrap_targets",
    "td_lambda_targets",
    "TDTrainConfig",
    "TDTrainStepSummary",
    "TDValueTrainer",
    "hard_sync",
    "train_value_batch",
    "OpponentSample",
    "ValueTransition",
    "TD_VALUE_CHECKPOINT_TYPE",
    "TD_OPPONENT_CHECKPOINT_TYPE",
    "save_value_checkpoint",
    "load_value_checkpoint",
    "save_opponent_checkpoint",
    "load_opponent_checkpoint",
]
