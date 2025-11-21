from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .env import MagnateBridgeEnv
from .evaluate import MatchSummary, evaluate_matchup
from .policies import Policy
from .types import Winner


@dataclass(frozen=True)
class SideSwappedEvalSummary:
    games_per_side: int
    total_games: int
    candidate_name: str
    opponent_name: str
    leg_candidate_as_player_a: MatchSummary
    leg_candidate_as_player_b: MatchSummary
    winners: Dict[Winner, int]
    candidate_wins: int
    opponent_wins: int
    draws: int
    candidate_win_rate: float
    candidate_win_rate_ci95: tuple[float, float]
    candidate_win_rate_as_player_a: float
    candidate_win_rate_as_player_b: float
    side_gap: float
    average_turn: float

    def to_json(self) -> Dict[str, object]:
        return {
            "gamesPerSide": self.games_per_side,
            "totalGames": self.total_games,
            "candidate": self.candidate_name,
            "opponent": self.opponent_name,
            "winners": dict(self.winners),
            "candidateWins": self.candidate_wins,
            "opponentWins": self.opponent_wins,
            "draws": self.draws,
            "candidateWinRate": self.candidate_win_rate,
            "candidateWinRateCi95": {
                "low": self.candidate_win_rate_ci95[0],
                "high": self.candidate_win_rate_ci95[1],
            },
            "candidateWinRateAsPlayerA": self.candidate_win_rate_as_player_a,
            "candidateWinRateAsPlayerB": self.candidate_win_rate_as_player_b,
            "sideGap": self.side_gap,
            "averageTurn": self.average_turn,
            "legs": {
                "candidateAsPlayerA": _match_summary_to_json(self.leg_candidate_as_player_a),
                "candidateAsPlayerB": _match_summary_to_json(self.leg_candidate_as_player_b),
            },
        }


def evaluate_side_swapped(
    *,
    env: MagnateBridgeEnv,
    candidate_policy: Policy,
    opponent_policy: Policy,
    games_per_side: int,
    seed_prefix: str,
) -> SideSwappedEvalSummary:
    if games_per_side <= 0:
        raise ValueError("games_per_side must be > 0.")

    leg_as_a = evaluate_matchup(
        env=env,
        policy_player_a=candidate_policy,
        policy_player_b=opponent_policy,
        games=games_per_side,
        seed_prefix=seed_prefix,
    )
    leg_as_b = evaluate_matchup(
        env=env,
        policy_player_a=opponent_policy,
        policy_player_b=candidate_policy,
        games=games_per_side,
        seed_prefix=seed_prefix,
    )

    candidate_wins_as_a = leg_as_a.winners["PlayerA"]
    candidate_wins_as_b = leg_as_b.winners["PlayerB"]
    candidate_wins = candidate_wins_as_a + candidate_wins_as_b
    draws = leg_as_a.winners["Draw"] + leg_as_b.winners["Draw"]
    total_games = games_per_side * 2
    opponent_wins = total_games - candidate_wins - draws

    candidate_win_rate = candidate_wins / float(total_games)
    candidate_win_rate_as_a = candidate_wins_as_a / float(games_per_side)
    candidate_win_rate_as_b = candidate_wins_as_b / float(games_per_side)
    ci95 = wilson_interval(candidate_wins, total_games)

    winners: Dict[Winner, int] = {
        "PlayerA": leg_as_a.winners["PlayerA"] + leg_as_b.winners["PlayerA"],
        "PlayerB": leg_as_a.winners["PlayerB"] + leg_as_b.winners["PlayerB"],
        "Draw": draws,
    }
    average_turn = (
        ((leg_as_a.average_turn * games_per_side) + (leg_as_b.average_turn * games_per_side))
        / float(total_games)
    )

    return SideSwappedEvalSummary(
        games_per_side=games_per_side,
        total_games=total_games,
        candidate_name=candidate_policy.name,
        opponent_name=opponent_policy.name,
        leg_candidate_as_player_a=leg_as_a,
        leg_candidate_as_player_b=leg_as_b,
        winners=winners,
        candidate_wins=candidate_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        candidate_win_rate=candidate_win_rate,
        candidate_win_rate_ci95=ci95,
        candidate_win_rate_as_player_a=candidate_win_rate_as_a,
        candidate_win_rate_as_player_b=candidate_win_rate_as_b,
        side_gap=abs(candidate_win_rate_as_a - candidate_win_rate_as_b),
        average_turn=average_turn,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError("trials must be > 0.")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be in [0, trials].")
    if z <= 0:
        raise ValueError("z must be > 0.")

    p_hat = successes / float(trials)
    z2 = z * z
    denom = 1.0 + (z2 / float(trials))
    center = (p_hat + (z2 / (2.0 * float(trials)))) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + (z2 / (4.0 * float(trials)))) / float(trials))
        / denom
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def _match_summary_to_json(summary: MatchSummary) -> Dict[str, object]:
    return {
        "games": summary.games,
        "winners": dict(summary.winners),
        "winsByPolicy": dict(summary.wins_by_policy),
        "averageTurn": summary.average_turn,
    }
