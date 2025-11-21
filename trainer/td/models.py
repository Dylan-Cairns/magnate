from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ValueNet(nn.Module):
    def __init__(self, *, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        values = self.encoder(observations).squeeze(-1)
        return values


class OpponentModel(nn.Module):
    def __init__(
        self,
        *,
        observation_dim: int,
        action_feature_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be > 0.")
        if action_feature_dim <= 0:
            raise ValueError("action_feature_dim must be > 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0.")
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

    def logits_tensor(
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
        return self.policy_head(pair_features).squeeze(-1)

    def action_distribution(
        self,
        *,
        observation: Sequence[float] | torch.Tensor,
        action_features: Sequence[Sequence[float]] | torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.distributions.Categorical:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        observation_tensor = (
            observation
            if isinstance(observation, torch.Tensor)
            else torch.tensor(observation, dtype=torch.float32)
        )
        action_tensor = (
            action_features
            if isinstance(action_features, torch.Tensor)
            else torch.tensor(action_features, dtype=torch.float32)
        )
        logits = self.logits_tensor(observation_tensor, action_tensor) / temperature
        return torch.distributions.Categorical(logits=logits)
