from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from .encoding import ENCODING_VERSION

PPO_CHECKPOINT_TYPE = "magnate_ppo_policy_v1"


class CandidateActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_feature_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_feature_dim = action_feature_dim
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def policy_logits_tensor(
        self,
        observation: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if action_features.dim() == 1:
            action_features = action_features.unsqueeze(0)

        obs_embed = self.obs_encoder(observation).squeeze(0)
        action_embed = self.action_encoder(action_features)
        obs_expand = obs_embed.unsqueeze(0).expand(action_embed.shape[0], -1)
        pair_features = torch.cat(
            [obs_expand, action_embed, obs_expand * action_embed],
            dim=-1,
        )
        logits = self.policy_head(pair_features).squeeze(-1)
        return logits

    def value_tensor(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        obs_embed = self.obs_encoder(observation)
        value = self.value_head(obs_embed).squeeze(-1)
        return value.squeeze(0)

    def action_distribution(
        self,
        observation: torch.Tensor,
        action_features: torch.Tensor,
        temperature: float = 1.0,
    ) -> "torch.distributions.Categorical":
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        logits = self.policy_logits_tensor(observation, action_features) / temperature
        return torch.distributions.Categorical(logits=logits)


def save_ppo_checkpoint(
    model: CandidateActorCritic,
    output_path: Path,
    metadata: Mapping[str, Any] | None = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpointType": PPO_CHECKPOINT_TYPE,
        "encodingVersion": ENCODING_VERSION,
        "observationDim": int(model.observation_dim),
        "actionFeatureDim": int(model.action_feature_dim),
        "hiddenDim": int(model.hidden_dim),
        "stateDict": model.state_dict(),
        "metadata": dict(metadata) if metadata is not None else {},
    }
    if optimizer_state_dict is not None:
        payload["optimizerStateDict"] = optimizer_state_dict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_ppo_checkpoint(
    path: Path,
    map_location: str = "cpu",
) -> Tuple[CandidateActorCritic, Dict[str, Any]]:
    raw = torch.load(path, map_location=map_location)
    if not isinstance(raw, dict):
        raise ValueError("PPO checkpoint must be a mapping payload.")

    checkpoint_type = str(raw.get("checkpointType", ""))
    if checkpoint_type != PPO_CHECKPOINT_TYPE:
        raise ValueError(
            f"Unsupported checkpoint type {checkpoint_type!r}; expected {PPO_CHECKPOINT_TYPE!r}."
        )
    if "encodingVersion" not in raw:
        raise ValueError(
            "PPO checkpoint is missing encodingVersion metadata. "
            "Legacy checkpoints are not compatible with the current training encoding."
        )
    encoding_version = _as_positive_int(raw.get("encodingVersion"), "encodingVersion")
    if encoding_version != ENCODING_VERSION:
        raise ValueError(
            "Encoding version mismatch for PPO checkpoint. "
            f"checkpoint={encoding_version} expected={ENCODING_VERSION}."
        )

    observation_dim = _as_positive_int(raw.get("observationDim"), "observationDim")
    action_feature_dim = _as_positive_int(raw.get("actionFeatureDim"), "actionFeatureDim")
    hidden_dim = _as_positive_int(raw.get("hiddenDim"), "hiddenDim")
    state_dict = raw.get("stateDict")
    if not isinstance(state_dict, dict):
        raise ValueError("PPO checkpoint stateDict must be a mapping.")

    model = CandidateActorCritic(
        observation_dim=observation_dim,
        action_feature_dim=action_feature_dim,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, raw


def _as_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value <= 0:
        raise ValueError(f"{label} must be > 0.")
    return value
