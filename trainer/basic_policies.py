from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .bridge_payloads import PlayerViewPayload, SerializedStatePayload
from .encoding import _card_rank
from .types import KeyedAction


class Policy:
    name: str

    def choose_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: SerializedStatePayload | None = None,
    ) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def root_action_probs(self) -> Mapping[str, float] | None:
        return None


@dataclass
class RandomLegalPolicy(Policy):
    name: str = "random"

    def choose_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: SerializedStatePayload | None = None,
    ) -> str:
        del view
        del state
        if not legal_actions:
            raise ValueError("Random policy requires at least one legal action.")
        return legal_actions[rng.randrange(len(legal_actions))].action_key


@dataclass
class HeuristicPolicy(Policy):
    name: str = "heuristic"

    def choose_action_key(
        self,
        view: PlayerViewPayload,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
        state: SerializedStatePayload | None = None,
    ) -> str:
        del view
        del rng
        del state
        if not legal_actions:
            raise ValueError("Heuristic policy requires at least one legal action.")

        ranked = sorted(
            legal_actions,
            key=lambda action: (
                -self.score_action(action),
                action.action_key,
            ),
        )
        return ranked[0].action_key

    def score_action(self, action: KeyedAction) -> float:
        payload = action.action
        action_id = action.action_id
        score = {
            "develop-outright": 8.0,
            "develop-deed": 6.0,
            "buy-deed": 5.0,
            "choose-income-suit": 4.0,
            "trade": 2.0,
            "sell-card": 1.0,
            "end-turn": 0.0,
        }.get(action_id, 0.0)

        card_id = payload["cardId"] if "cardId" in payload else ""
        card_rank = _card_rank(card_id)

        if action_id in ("develop-outright", "develop-deed"):
            score += card_rank * 0.4
        if action_id == "buy-deed":
            score += card_rank * 0.25
            if card_rank <= 2:
                score -= 1.5
        if action_id == "sell-card":
            score -= card_rank * 0.3
        if payload["type"] == "trade":
            give = payload["give"]
            receive = payload["receive"]
            if give == receive:
                score -= 10.0
            else:
                score += 0.2
        return score


__all__ = [
    "Policy",
    "RandomLegalPolicy",
    "HeuristicPolicy",
]
