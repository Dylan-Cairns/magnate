from __future__ import annotations

import math
from typing import Dict, Mapping, Protocol, Sequence

from trainer.types import KeyedAction


class SupportsScoreAction(Protocol):
    def score_action(self, action: KeyedAction) -> float: ...


def rank_root_actions(
    *,
    legal_actions: Sequence[KeyedAction],
    heuristic_policy: SupportsScoreAction,
) -> list[KeyedAction]:
    return sorted(
        legal_actions,
        key=lambda action: (
            -heuristic_policy.score_action(action),
            action.action_key,
        ),
    )


def root_priors_by_key(
    *,
    legal_actions: Sequence[KeyedAction],
    heuristic_policy: SupportsScoreAction,
) -> Dict[str, float]:
    if not legal_actions:
        return {}

    scores = [heuristic_policy.score_action(action) for action in legal_actions]
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    normalizer = sum(exp_scores)
    if normalizer <= 0.0:
        uniform = 1.0 / float(len(legal_actions))
        return {action.action_key: uniform for action in legal_actions}

    return {
        action.action_key: exp_scores[index] / normalizer
        for index, action in enumerate(legal_actions)
    }


def progressive_target_action_count(
    *,
    total_actions: int,
    initial_actions: int,
    visits: int,
) -> int:
    if total_actions <= 0:
        return 0
    base = max(1, min(initial_actions, total_actions))
    widened = base + int(math.sqrt(float(max(0, visits)) + 1.0))
    return min(total_actions, widened)


def select_root_ucb_action(
    *,
    action_keys: Sequence[str],
    visits_by_key: Mapping[str, int],
    value_sum_by_key: Mapping[str, float],
    priors_by_key: Mapping[str, float],
    total_visits: int,
    c_puct: float = 1.0,
) -> str:
    if not action_keys:
        raise ValueError("select_root_ucb_action requires at least one action key.")
    if c_puct <= 0:
        raise ValueError("c_puct must be > 0.")

    sqrt_parent = math.sqrt(float(max(0, total_visits)) + 1.0)
    best_key = action_keys[0]
    best_score = float("-inf")

    for action_key in action_keys:
        visits = visits_by_key.get(action_key, 0)
        value_sum = value_sum_by_key.get(action_key, 0.0)
        q = (value_sum / float(visits)) if visits > 0 else 0.0
        prior = priors_by_key.get(action_key, 0.0)
        score = q + (c_puct * prior * sqrt_parent / (1.0 + float(visits)))
        if score > best_score or (
            math.isclose(score, best_score, abs_tol=1e-9)
            and action_key < best_key
        ):
            best_key = action_key
            best_score = score
    return best_key
