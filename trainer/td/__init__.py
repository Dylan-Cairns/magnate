"""TD/Keldon training primitives for Magnate."""

from .checkpoint import (
    TD_OPPONENT_CHECKPOINT_TYPE,
    TD_VALUE_CHECKPOINT_TYPE,
    load_opponent_checkpoint,
    load_value_checkpoint,
    save_opponent_checkpoint,
    save_value_checkpoint,
)
from .io import (
    read_opponent_samples_jsonl,
    read_value_transitions_jsonl,
    write_opponent_samples_jsonl,
    write_value_transitions_jsonl,
)
from .models import OpponentModel, ValueNet
from .replay import OpponentReplayBuffer, ValueReplayBuffer
from .self_play import (
    SelfPlayEpisode,
    SelfPlayProgressCallback,
    collect_self_play_episode,
    collect_self_play_games,
    flatten_opponent_samples,
    flatten_value_transitions,
)
from .targets import n_step_bootstrap_targets, td_lambda_targets
from .train import (
    OpponentTrainConfig,
    OpponentTrainStepSummary,
    TDTrainConfig,
    TDTrainStepSummary,
    TDOpponentTrainer,
    TDValueTrainer,
    hard_sync,
    train_opponent_batch,
    train_value_batch,
)
from .types import OpponentSample, ValueTransition

__all__ = [
    "OpponentModel",
    "ValueNet",
    "OpponentReplayBuffer",
    "ValueReplayBuffer",
    "read_value_transitions_jsonl",
    "write_value_transitions_jsonl",
    "read_opponent_samples_jsonl",
    "write_opponent_samples_jsonl",
    "SelfPlayEpisode",
    "SelfPlayProgressCallback",
    "collect_self_play_episode",
    "collect_self_play_games",
    "flatten_opponent_samples",
    "flatten_value_transitions",
    "n_step_bootstrap_targets",
    "td_lambda_targets",
    "TDTrainConfig",
    "TDTrainStepSummary",
    "OpponentTrainConfig",
    "OpponentTrainStepSummary",
    "TDValueTrainer",
    "TDOpponentTrainer",
    "hard_sync",
    "train_value_batch",
    "train_opponent_batch",
    "OpponentSample",
    "ValueTransition",
    "TD_VALUE_CHECKPOINT_TYPE",
    "TD_OPPONENT_CHECKPOINT_TYPE",
    "save_value_checkpoint",
    "load_value_checkpoint",
    "save_opponent_checkpoint",
    "load_opponent_checkpoint",
]
