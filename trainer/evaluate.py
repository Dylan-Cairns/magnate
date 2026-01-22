from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Mapping

from .basic_policies import Policy
from .env import MagnateBridgeEnv
from .types import PlayerId, Winner

ProgressCallback = Callable[[int, int, Mapping[Winner, int], Mapping[PlayerId, int]], None]


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
    wins_by_seat: Dict[PlayerId, int]
    policy_name_by_seat: Dict[PlayerId, str]
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
        action_key = policy.choose_action_key(
            step_result.view,
            legal.actions,
            rng,
            state=step_result.state,
        )
        step_result = env.step(action_key=action_key)

    final_score = step_result.state.get("finalScore")
    if not isinstance(final_score, dict):
        raise RuntimeError("Terminal state is missing finalScore payload.")
    winner = final_score.get("winner")
    if winner not in ("PlayerA", "PlayerB", "Draw"):
        raise RuntimeError(f"Invalid winner in terminal state: {winner!r}")

    return GameResult(
        seed=seed,
        first_player=first_player,
        winner=winner,
        turn=int(step_result.state.get("turn", 0)),
    )


def evaluate_matchup(
    env: MagnateBridgeEnv,
    policy_player_a: Policy,
    policy_player_b: Policy,
    games: int,
    seed_prefix: str,
    seed_start_index: int = 0,
    progress_every_games: int = 0,
    on_progress: ProgressCallback | None = None,
) -> MatchSummary:
    policies: Dict[PlayerId, Policy] = {
        "PlayerA": policy_player_a,
        "PlayerB": policy_player_b,
    }

    winners: Dict[Winner, int] = {"PlayerA": 0, "PlayerB": 0, "Draw": 0}
    wins_by_seat: Dict[PlayerId, int] = {
        "PlayerA": 0,
        "PlayerB": 0,
    }
    policy_name_by_seat: Dict[PlayerId, str] = {
        "PlayerA": policy_player_a.name,
        "PlayerB": policy_player_b.name,
    }

    turn_total = 0
    rng = random.Random(0)
    for index in range(games):
        seed = f"{seed_prefix}-{seed_start_index + index}"
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
            wins_by_seat["PlayerA"] += 1
        elif result.winner == "PlayerB":
            wins_by_seat["PlayerB"] += 1
        turn_total += result.turn
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
                dict(winners),
                dict(wins_by_seat),
            )

    average_turn = (turn_total / games) if games > 0 else 0.0
    return MatchSummary(
        games=games,
        winners=winners,
        wins_by_seat=wins_by_seat,
        policy_name_by_seat=policy_name_by_seat,
        average_turn=average_turn,
    )
