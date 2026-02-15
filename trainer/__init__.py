"""Magnate training scaffold built on top of the TS bridge runtime."""

from .bridge_client import BridgeClient, BridgeError
from .encoding import (
    ACTION_FEATURE_DIM,
    OBSERVATION_DIM,
    encode_action_candidates,
    encode_observation,
)
from .env import MagnateBridgeEnv

__all__ = [
    "ACTION_FEATURE_DIM",
    "BridgeClient",
    "BridgeError",
    "MagnateBridgeEnv",
    "OBSERVATION_DIM",
    "encode_action_candidates",
    "encode_observation",
]

