from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bridge_payloads import GameActionPayload, PlayerViewPayload, SerializedStatePayload
from .bridge_client import BridgeClient
from .types import LegalActionsResult, ObservationResult, PlayerId, StateResult


@dataclass
class MagnateBridgeEnv:
    """Thin environment wrapper around bridge commands."""

    client: BridgeClient
    current_state: Optional[SerializedStatePayload] = None
    current_view: Optional[PlayerViewPayload] = None
    terminal: bool = False

    def reset(
        self,
        seed: Optional[str] = None,
        first_player: PlayerId = "PlayerA",
        serialized_state: Optional[SerializedStatePayload] = None,
        skip_advance_to_decision: bool = False,
    ) -> StateResult:
        result = self.client.reset(
            seed=seed,
            first_player=first_player,
            serialized_state=serialized_state,
            skip_advance_to_decision=skip_advance_to_decision,
        )
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
        action: Optional[GameActionPayload] = None,
    ) -> StateResult:
        result = self.client.step(action_key=action_key, action=action)
        self.current_state = result.state
        self.current_view = result.view
        self.terminal = result.terminal
        return result
