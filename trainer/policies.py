from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from .behavior_cloning import BehaviorCloningModel, load_behavior_cloning_checkpoint
from .encoding import _card_rank, encode_action_candidates, encode_observation
from .types import KeyedAction


class Policy:
    name: str

    def choose_action_key(self, view: Dict, legal_actions: Sequence[KeyedAction], rng: random.Random) -> str:
        raise NotImplementedError


@dataclass
class RandomLegalPolicy(Policy):
    name: str = "random"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if not legal_actions:
            raise ValueError("Random policy requires at least one legal action.")
        return legal_actions[rng.randrange(len(legal_actions))].action_key


@dataclass
class HeuristicPolicy(Policy):
    name: str = "heuristic"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        if not legal_actions:
            raise ValueError("Heuristic policy requires at least one legal action.")

        ranked = sorted(
            legal_actions,
            key=lambda action: (
                -self._score(action),
                action.action_key,
            ),
        )
        return ranked[0].action_key

    def _score(self, action: KeyedAction) -> float:
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

        card_id = str(payload.get("cardId", ""))
        card_rank = _card_rank(card_id)

        if action_id in ("develop-outright", "develop-deed"):
            score += card_rank * 0.4
        if action_id == "buy-deed":
            score += card_rank * 0.25
            if card_rank <= 2:
                score -= 1.5
        if action_id == "sell-card":
            score -= card_rank * 0.3
        if action_id == "trade":
            give = str(payload.get("give", ""))
            receive = str(payload.get("receive", ""))
            if give == receive:
                score -= 10.0
            else:
                score += 0.2
        return score


@dataclass
class BehaviorCloningPolicy(Policy):
    model: BehaviorCloningModel
    checkpoint_path: str = ""
    name: str = "behavior-cloned"

    def choose_action_key(
        self,
        view: Dict,
        legal_actions: Sequence[KeyedAction],
        rng: random.Random,
    ) -> str:
        del rng  # deterministic action selection from the trained checkpoint
        if not legal_actions:
            raise ValueError("Behavior-cloned policy requires at least one legal action.")

        observation_vector = encode_observation(view)
        action_vectors = encode_action_candidates(legal_actions)
        action_index = self.model.choose_action_index(observation_vector, action_vectors)
        return legal_actions[action_index].action_key


def policy_from_name(name: str, checkpoint_path: str | Path | None = None) -> Policy:
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomLegalPolicy()
    if normalized == "heuristic":
        return HeuristicPolicy()
    if normalized in ("bc", "behavior-cloned", "behavior_cloned"):
        if checkpoint_path is None:
            raise ValueError("Policy 'bc' requires a checkpoint path.")
        path = Path(checkpoint_path)
        model = load_behavior_cloning_checkpoint(path)
        return BehaviorCloningPolicy(
            model=model,
            checkpoint_path=str(path),
            name=f"behavior-cloned:{path.name}",
        )
    raise ValueError(f"Unknown policy name: {name!r}. Expected one of: random, heuristic, bc.")
