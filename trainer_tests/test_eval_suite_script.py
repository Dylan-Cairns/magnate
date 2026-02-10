from __future__ import annotations

import math
import unittest

from scripts.eval_suite import (
    _merge_shard_results,
    _split_games,
    _sprt_boundaries,
    _sprt_log_likelihood_ratio,
)


class EvalSuiteScriptTests(unittest.TestCase):
    def test_split_games_distributes_remainder(self) -> None:
        self.assertEqual(_split_games(total_games_per_side=10, workers=3), [4, 3, 3])
        self.assertEqual(_split_games(total_games_per_side=6, workers=3), [2, 2, 2])

    def test_merge_shard_results_combines_metrics(self) -> None:
        shard_one = {
            "gamesPerSide": 2,
            "totalGames": 4,
            "candidate": "search",
            "opponent": "heuristic",
            "winners": {"PlayerA": 2, "PlayerB": 1, "Draw": 1},
            "candidateWins": 2,
            "opponentWins": 1,
            "draws": 1,
            "averageTurn": 12.0,
            "legs": {
                "candidateAsPlayerA": {
                    "games": 2,
                    "winners": {"PlayerA": 1, "PlayerB": 0, "Draw": 1},
                    "winsBySeat": {"PlayerA": 1, "PlayerB": 0},
                    "policyBySeat": {"PlayerA": "search", "PlayerB": "heuristic"},
                    "averageTurn": 11.0,
                },
                "candidateAsPlayerB": {
                    "games": 2,
                    "winners": {"PlayerA": 1, "PlayerB": 1, "Draw": 0},
                    "winsBySeat": {"PlayerA": 1, "PlayerB": 1},
                    "policyBySeat": {"PlayerA": "heuristic", "PlayerB": "search"},
                    "averageTurn": 13.0,
                },
            },
        }
        shard_two = {
            "gamesPerSide": 2,
            "totalGames": 4,
            "candidate": "search",
            "opponent": "heuristic",
            "winners": {"PlayerA": 1, "PlayerB": 2, "Draw": 1},
            "candidateWins": 1,
            "opponentWins": 2,
            "draws": 1,
            "averageTurn": 10.0,
            "legs": {
                "candidateAsPlayerA": {
                    "games": 2,
                    "winners": {"PlayerA": 1, "PlayerB": 1, "Draw": 0},
                    "winsBySeat": {"PlayerA": 1, "PlayerB": 1},
                    "policyBySeat": {"PlayerA": "search", "PlayerB": "heuristic"},
                    "averageTurn": 10.0,
                },
                "candidateAsPlayerB": {
                    "games": 2,
                    "winners": {"PlayerA": 0, "PlayerB": 1, "Draw": 1},
                    "winsBySeat": {"PlayerA": 0, "PlayerB": 1},
                    "policyBySeat": {"PlayerA": "heuristic", "PlayerB": "search"},
                    "averageTurn": 10.0,
                },
            },
        }

        merged = _merge_shard_results([shard_one, shard_two])
        self.assertEqual(merged["gamesPerSide"], 4)
        self.assertEqual(merged["totalGames"], 8)
        self.assertEqual(merged["candidateWins"], 3)
        self.assertEqual(merged["opponentWins"], 3)
        self.assertEqual(merged["draws"], 2)
        self.assertAlmostEqual(float(merged["candidateWinRate"]), 0.375)
        self.assertAlmostEqual(float(merged["candidateWinRateAsPlayerA"]), 0.5)
        self.assertAlmostEqual(float(merged["candidateWinRateAsPlayerB"]), 0.5)
        self.assertAlmostEqual(float(merged["sideGap"]), 0.0)
        self.assertEqual(
            merged["legs"]["candidateAsPlayerA"]["winsBySeat"],
            {"PlayerA": 2, "PlayerB": 1},
        )
        self.assertEqual(
            merged["legs"]["candidateAsPlayerA"]["policyBySeat"],
            {"PlayerA": "search", "PlayerB": "heuristic"},
        )
        self.assertEqual(
            merged["legs"]["candidateAsPlayerB"]["winsBySeat"],
            {"PlayerA": 1, "PlayerB": 2},
        )
        self.assertEqual(
            merged["legs"]["candidateAsPlayerB"]["policyBySeat"],
            {"PlayerA": "heuristic", "PlayerB": "search"},
        )

    def test_sprt_boundaries_and_llr_move_in_expected_direction(self) -> None:
        accept, reject = _sprt_boundaries(alpha=0.05, beta=0.10)
        self.assertGreater(accept, 0.0)
        self.assertLess(reject, 0.0)

        llr_strong = _sprt_log_likelihood_ratio(
            successes=60,
            trials=100,
            h0=0.50,
            h1=0.55,
        )
        llr_weak = _sprt_log_likelihood_ratio(
            successes=45,
            trials=100,
            h0=0.50,
            h1=0.55,
        )
        self.assertGreater(llr_strong, 0.0)
        self.assertLess(llr_weak, 0.0)
        self.assertTrue(math.isfinite(llr_strong))
        self.assertTrue(math.isfinite(llr_weak))


if __name__ == "__main__":
    unittest.main()
