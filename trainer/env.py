from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .bridge_client import BridgeClient
from .types import LegalActionsResult, ObservationResult, PlayerId, StateResult


@dataclass
class MagnateBridgeEnv:
    """Thin environment wrapper around bridge commands."""

    client: BridgeClient
    current_state: Optional[Dict] = None
    current_view: Optional[Dict] = None
    terminal: bool = False

    def reset(
        self,
        seed: str,
        first_player: PlayerId = "PlayerA",
    ) -> StateResult:
        result = self.client.reset(seed=seed, first_player=first_player)
        self.current_state = result.state
        self.current_view = result.view
        self.terminal = result.terminal
        return result

    def legal_actions(self) -> LegalActionsResult:
        return self.client.legal_actions()

    def observation(
        self,
        viewer_id: Optional[PlayerId] = None,
        include_legal_action_mask: bool = True,
    ) -> ObservationResult:
        return self.client.observation(
            viewer_id=viewer_id,
            include_legal_action_mask=include_legal_action_mask,
        )

    def step(
        self,
        action_key: Optional[str] = None,
        action: Optional[Dict] = None,
    ) -> StateResult:
        result = self.client.step(action_key=action_key, action=action)
        self.current_state = result.state
        self.current_view = result.view
        self.terminal = result.terminal
        return result

