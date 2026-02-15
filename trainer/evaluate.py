from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from .env import MagnateBridgeEnv
from .policies import Policy
from .types import PlayerId, Winner


@dataclass(frozen=True)
class GameResult:
    seed: str
    first_player: PlayerId
    winner: Winner
    turn: int


@dataclass(frozen=True)
class MatchSummary:
    games: int
    winners: Dict[Winner, int]
    wins_by_policy: Dict[str, int]
    average_turn: float


def play_game(
    env: MagnateBridgeEnv,
    policies: Mapping[PlayerId, Policy],
    seed: str,
    first_player: PlayerId,
    rng: random.Random,
) -> GameResult:
    step_result = env.reset(seed=seed, first_player=first_player)

    while not step_result.terminal:
        legal = env.legal_actions()
        active_player = legal.active_player_id
        policy = policies[active_player]
        action_key = policy.choose_action_key(step_result.view, legal.actions, rng)
        step_result = env.step(action_key=action_key)

    final_score = step_result.state.get("finalScore", {})
    winner = str(final_score.get("winner", "Draw"))
    if winner not in ("PlayerA", "PlayerB", "Draw"):
        winner = "Draw"

    return GameResult(
        seed=seed,
        first_player=first_player,
        winner=winner,  # type: ignore[arg-type]
        turn=int(step_result.state.get("turn", 0)),
    )


def evaluate_matchup(
    env: MagnateBridgeEnv,
    policy_player_a: Policy,
    policy_player_b: Policy,
    games: int,
    seed_prefix: str,
) -> MatchSummary:
    policies: Dict[PlayerId, Policy] = {
        "PlayerA": policy_player_a,
        "PlayerB": policy_player_b,
    }

    winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    wins_by_policy = {
        policy_player_a.name: 0,
        policy_player_b.name: 0,
    }

    turn_total = 0
    rng = random.Random(0)
    for index in range(games):
        seed = f"{seed_prefix}-{index}"
        first_player: PlayerId = "PlayerA" if index % 2 == 0 else "PlayerB"
        result = play_game(
            env=env,
            policies=policies,
            seed=seed,
            first_player=first_player,
            rng=rng,
        )
        winners[result.winner] += 1
        if result.winner == "PlayerA":
            wins_by_policy[policy_player_a.name] += 1
        elif result.winner == "PlayerB":
            wins_by_policy[policy_player_b.name] += 1
        turn_total += result.turn

    average_turn = (turn_total / games) if games > 0 else 0.0
    return MatchSummary(
        games=games,
        winners=winners,
        wins_by_policy=wins_by_policy,
        average_turn=average_turn,
    )

