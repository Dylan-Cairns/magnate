from __future__ import annotations

import unittest

from trainer.benchmarking import (
    BENCHMARK_HEURISTIC_SEED_PREFIX,
    BENCHMARK_HEURISTIC_WEIGHT,
    BENCHMARK_RANDOM_SEED_PREFIX,
    BENCHMARK_RANDOM_WEIGHT,
    benchmark_spec_json,
    canonical_selection_score,
    player_a_win_rate,
)
from trainer.evaluate import MatchSummary


class BenchmarkingTests(unittest.TestCase):
    def test_player_a_win_rate_requires_positive_games(self) -> None:
        summary = MatchSummary(
            games=0,
            winners={"PlayerA": 0, "PlayerB": 0, "Draw": 0},
            wins_by_policy={},
            average_turn=0.0,
        )
        with self.assertRaises(ValueError):
            player_a_win_rate(summary)

    def test_canonical_selection_score_uses_locked_weights(self) -> None:
        random_summary = MatchSummary(
            games=100,
            winners={"PlayerA": 80, "PlayerB": 20, "Draw": 0},
            wins_by_policy={"candidate": 80, "random": 20},
            average_turn=43.0,
        )
        heuristic_summary = MatchSummary(
            games=100,
            winners={"PlayerA": 40, "PlayerB": 60, "Draw": 0},
            wins_by_policy={"candidate": 40, "heuristic": 60},
            average_turn=43.0,
        )
        expected = (BENCHMARK_HEURISTIC_WEIGHT * 0.40) + (BENCHMARK_RANDOM_WEIGHT * 0.80)
        self.assertAlmostEqual(
            canonical_selection_score(random_summary, heuristic_summary),
            expected,
        )

    def test_benchmark_spec_exposes_locked_seed_prefixes(self) -> None:
        spec = benchmark_spec_json()
        self.assertEqual(spec["randomSeedPrefix"], BENCHMARK_RANDOM_SEED_PREFIX)
        self.assertEqual(spec["heuristicSeedPrefix"], BENCHMARK_HEURISTIC_SEED_PREFIX)


if __name__ == "__main__":
    unittest.main()
