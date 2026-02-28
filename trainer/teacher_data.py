from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Dict, Mapping, Sequence, Set

from .encoding import encode_action_candidates, encode_observation
from .env import MagnateBridgeEnv
from .policies import Policy
from .types import DecisionSample, PlayerId, Winner

ProgressCallback = Callable[
    [int, int, int, Mapping[Winner, int], Mapping[PlayerId, int]], None
]


@dataclass(frozen=True)
class TeacherCollectionSummary:
    games: int
    samples: int
    winners: Dict[Winner, int]
    decisions_by_player: Dict[PlayerId, int]
    average_turn: float
    teacher_players: tuple[PlayerId, ...]

    def as_json(self) -> Dict[str, object]:
        return {
            "games": self.games,
            "samples": self.samples,
            "winners": self.winners,
            "decisionsByPlayer": self.decisions_by_player,
            "averageTurn": self.average_turn,
            "teacherPlayers": list(self.teacher_players),
        }


def collect_teacher_samples(
    env: MagnateBridgeEnv,
    teacher_policy: Policy,
    opponent_policy: Policy | None,
    games: int,
    seed_prefix: str,
    teacher_player_ids: Set[PlayerId],
    rng_seed: int = 0,
    progress_every_games: int = 0,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[DecisionSample], TeacherCollectionSummary]:
    if games < 1:
        raise ValueError("games must be >= 1")
    if not teacher_player_ids:
        raise ValueError("teacher_player_ids must include at least one player.")
    if opponent_policy is None and teacher_player_ids != {"PlayerA", "PlayerB"}:
        raise ValueError(
            "opponent_policy is required unless teacher_player_ids includes both players."
        )

    winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    decisions_by_player: Dict[PlayerId, int] = {"PlayerA": 0, "PlayerB": 0}
    turn_total = 0
    all_samples: list[DecisionSample] = []
    rng = random.Random(rng_seed)

    for index in range(games):
        seed = f"{seed_prefix}-{index}"
        first_player: PlayerId = "PlayerA" if index % 2 == 0 else "PlayerB"
        step_result = env.reset(seed=seed, first_player=first_player)
        staged: list[DecisionSample] = []

        while not step_result.terminal:
            legal = env.legal_actions()
            active_player = legal.active_player_id
            active_policy = (
                teacher_policy
                if active_player in teacher_player_ids
                else _require_opponent_policy(opponent_policy)
            )

            action_key = active_policy.choose_action_key(
                step_result.view,
                legal.actions,
                rng,
                state=step_result.state,
            )

            if active_player in teacher_player_ids:
                observation_vector = encode_observation(step_result.view)
                action_vectors = encode_action_candidates(legal.actions)
                action_index = _find_action_index(legal.actions, action_key)
                chosen_action = legal.actions[action_index]
                action_probs = _action_probs_from_policy(
                    policy=active_policy,
                    legal_actions=legal.actions,
                    chosen_action_index=action_index,
                )
                staged.append(
                    DecisionSample(
                        seed=seed,
                        turn=int(step_result.state.get("turn", 0)),
                        phase=str(step_result.state.get("phase", "")),
                        active_player_id=active_player,
                        action_key=action_key,
                        action_id=chosen_action.action_id,
                        action_index=action_index,
                        observation=observation_vector,
                        action_features=action_vectors,
                        winner="Draw",
                        reward=0.0,
                        action_probs=action_probs,
                    )
                )
                decisions_by_player[active_player] += 1

            step_result = env.step(action_key=action_key)

        winner = _winner_from_state(step_result.state)
        winners[winner] += 1
        turn_total += int(step_result.state.get("turn", 0))
        all_samples.extend(_attach_reward(sample, winner) for sample in staged)

        completed_games = index + 1
        if (
            on_progress is not None
            and progress_every_games > 0
            and (
                completed_games % progress_every_games == 0
                or completed_games == games
            )
        ):
            on_progress(
                completed_games,
                games,
                len(all_samples),
                dict(winners),
                dict(decisions_by_player),
            )

    summary = TeacherCollectionSummary(
        games=games,
        samples=len(all_samples),
        winners=winners,
        decisions_by_player=decisions_by_player,
        average_turn=(turn_total / games),
        teacher_players=tuple(sorted(teacher_player_ids)),
    )
    return all_samples, summary


def _require_opponent_policy(opponent_policy: Policy | None) -> Policy:
    if opponent_policy is None:
        raise RuntimeError("opponent_policy is required for non-teacher turns.")
    return opponent_policy


def _winner_from_state(state: Mapping[str, object]) -> Winner:
    final_score = state.get("finalScore")
    if isinstance(final_score, dict):
        winner = final_score.get("winner")
        if winner in ("PlayerA", "PlayerB", "Draw"):
            return winner
    return "Draw"


def _attach_reward(sample: DecisionSample, winner: Winner) -> DecisionSample:
    if winner == "Draw":
        reward = 0.0
    elif winner == sample.active_player_id:
        reward = 1.0
    else:
        reward = -1.0
    return replace(sample, winner=winner, reward=reward)


def _find_action_index(actions, action_key: str) -> int:
    for index, action in enumerate(actions):
        if action.action_key == action_key:
            return index
    raise ValueError(f"Selected action key was not present in legal actions: {action_key}")


def _action_probs_from_policy(
    *,
    policy: Policy,
    legal_actions,
    chosen_action_index: int,
) -> list[float]:
    by_key = policy.root_action_probs()
    if not by_key:
        return _one_hot_distribution(len(legal_actions), chosen_action_index)

    raw: list[float] = []
    for action in legal_actions:
        raw.append(max(0.0, float(by_key.get(action.action_key, 0.0))))
    total = sum(raw)
    if total <= 0.0:
        return _one_hot_distribution(len(legal_actions), chosen_action_index)
    return [value / total for value in raw]


def _one_hot_distribution(size: int, index: int) -> list[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out
