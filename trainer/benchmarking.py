from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .env import MagnateBridgeEnv
from .evaluate import MatchSummary, evaluate_matchup
from .policies import Policy, policy_from_name

BENCHMARK_RANDOM_SEED_PREFIX = "bench-random-holdout"
BENCHMARK_HEURISTIC_SEED_PREFIX = "bench-heuristic-holdout"
BENCHMARK_GAMES = 200
BENCHMARK_HEURISTIC_WEIGHT = 0.7
BENCHMARK_RANDOM_WEIGHT = 0.3


@dataclass(frozen=True)
class BenchmarkSummary:
    games_per_matchup: int
    random: MatchSummary
    heuristic: MatchSummary
    random_win_rate: float
    heuristic_win_rate: float
    selection_score: float

    def to_json(self) -> Dict[str, object]:
        return {
            "gamesPerMatchup": self.games_per_matchup,
            "random": _match_summary_json("random", self.random),
            "heuristic": _match_summary_json("heuristic", self.heuristic),
            "randomWinRate": self.random_win_rate,
            "heuristicWinRate": self.heuristic_win_rate,
            "selectionScore": self.selection_score,
        }


def canonical_selection_score(
    random_summary: MatchSummary,
    heuristic_summary: MatchSummary,
) -> float:
    random_win_rate = player_a_win_rate(random_summary)
    heuristic_win_rate = player_a_win_rate(heuristic_summary)
    return (
        (BENCHMARK_HEURISTIC_WEIGHT * heuristic_win_rate)
        + (BENCHMARK_RANDOM_WEIGHT * random_win_rate)
    )


def player_a_win_rate(summary: MatchSummary) -> float:
    if summary.games <= 0:
        raise ValueError("Benchmark summary must include at least one game.")
    return summary.winners["PlayerA"] / float(summary.games)


def run_canonical_benchmark(
    env: MagnateBridgeEnv,
    candidate_policy: Policy,
    games_per_matchup: int = BENCHMARK_GAMES,
) -> BenchmarkSummary:
    if games_per_matchup <= 0:
        raise ValueError("games_per_matchup must be > 0.")

    random_summary = evaluate_matchup(
        env=env,
        policy_player_a=candidate_policy,
        policy_player_b=policy_from_name("random"),
        games=games_per_matchup,
        seed_prefix=BENCHMARK_RANDOM_SEED_PREFIX,
    )
    heuristic_summary = evaluate_matchup(
        env=env,
        policy_player_a=candidate_policy,
        policy_player_b=policy_from_name("heuristic"),
        games=games_per_matchup,
        seed_prefix=BENCHMARK_HEURISTIC_SEED_PREFIX,
    )

    return BenchmarkSummary(
        games_per_matchup=games_per_matchup,
        random=random_summary,
        heuristic=heuristic_summary,
        random_win_rate=player_a_win_rate(random_summary),
        heuristic_win_rate=player_a_win_rate(heuristic_summary),
        selection_score=canonical_selection_score(random_summary, heuristic_summary),
    )


def benchmark_spec_json() -> Dict[str, object]:
    return {
        "randomSeedPrefix": BENCHMARK_RANDOM_SEED_PREFIX,
        "heuristicSeedPrefix": BENCHMARK_HEURISTIC_SEED_PREFIX,
        "gamesPerMatchup": BENCHMARK_GAMES,
        "selectionWeights": {
            "heuristic": BENCHMARK_HEURISTIC_WEIGHT,
            "random": BENCHMARK_RANDOM_WEIGHT,
        },
    }


def _match_summary_json(opponent: str, summary: MatchSummary) -> Dict[str, object]:
    return {
        "opponent": opponent,
        "games": summary.games,
        "winners": dict(summary.winners),
        "winsByPolicy": dict(summary.wins_by_policy),
        "averageTurn": summary.average_turn,
    }
