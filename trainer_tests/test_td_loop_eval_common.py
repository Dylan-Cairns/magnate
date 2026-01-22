from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.td_loop_eval_common import (
    EvalRow,
    PromotionThresholds,
    build_eval_payload,
    evaluate_promotion_gate,
    pool_eval_rows,
    read_eval_row,
)


class TdLoopEvalCommonTests(unittest.TestCase):
    def _row(
        self,
        *,
        artifact: str,
        opponent_policy: str = "search",
        win_rate: float = 0.60,
        ci_low: float = 0.52,
        ci_high: float = 0.67,
        side_gap: float = 0.04,
        candidate_wins: int = 120,
        opponent_wins: int = 80,
        draws: int = 0,
        total_games: int = 200,
        rate_a: float = 0.62,
        rate_b: float = 0.58,
    ) -> EvalRow:
        return EvalRow(
            artifact=Path(artifact),
            opponent_policy=opponent_policy,
            candidate_win_rate=win_rate,
            ci_low=ci_low,
            ci_high=ci_high,
            side_gap=side_gap,
            candidate_wins=candidate_wins,
            opponent_wins=opponent_wins,
            draws=draws,
            total_games=total_games,
            candidate_win_rate_as_player_a=rate_a,
            candidate_win_rate_as_player_b=rate_b,
        )

    def test_read_eval_row_loads_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "eval.json"
            artifact.write_text(
                json.dumps(
                    {
                        "results": {
                            "candidateWinRate": 0.57,
                            "candidateWinRateCi95": {"low": 0.51, "high": 0.63},
                            "sideGap": 0.02,
                            "candidateWins": 114,
                            "opponentWins": 86,
                            "draws": 0,
                            "totalGames": 200,
                            "candidateWinRateAsPlayerA": 0.58,
                            "candidateWinRateAsPlayerB": 0.56,
                        }
                    }
                ),
                encoding="utf-8",
            )

            row = read_eval_row(artifact, opponent_policy="search")

        self.assertEqual(row.artifact, artifact)
        self.assertEqual(row.opponent_policy, "search")
        self.assertAlmostEqual(row.candidate_win_rate, 0.57)
        self.assertAlmostEqual(row.ci_low, 0.51)
        self.assertAlmostEqual(row.candidate_win_rate_as_player_a, 0.58)
        self.assertEqual(row.candidate_wins, 114)
        self.assertEqual(row.total_games, 200)

    def test_pool_eval_rows_sums_counts_and_recomputes_weighted_side_gap(self) -> None:
        pooled = pool_eval_rows(
            eval_rows=[
                self._row(
                    artifact="window-a.json",
                    candidate_wins=120,
                    opponent_wins=80,
                    total_games=200,
                    rate_a=0.62,
                    rate_b=0.58,
                ),
                self._row(
                    artifact="window-b.json",
                    candidate_wins=110,
                    opponent_wins=90,
                    total_games=200,
                    rate_a=0.59,
                    rate_b=0.56,
                ),
            ],
            opponent_policy="search",
        )

        self.assertEqual(pooled.artifact, Path("pooled"))
        self.assertEqual(pooled.candidate_wins, 230)
        self.assertEqual(pooled.opponent_wins, 170)
        self.assertEqual(pooled.total_games, 400)
        self.assertAlmostEqual(pooled.candidate_win_rate, 0.575)
        self.assertAlmostEqual(pooled.candidate_win_rate_as_player_a, 0.605)
        self.assertAlmostEqual(pooled.candidate_win_rate_as_player_b, 0.57)
        self.assertAlmostEqual(pooled.side_gap, 0.035)
        self.assertGreater(pooled.ci_high, pooled.ci_low)

    def test_pool_eval_rows_rejects_empty_input(self) -> None:
        with self.assertRaises(SystemExit):
            pool_eval_rows(eval_rows=[], opponent_policy="search")

    def test_build_eval_payload_emits_window_and_pooled_sections(self) -> None:
        windows = [
            self._row(
                artifact="window-a.json",
                win_rate=0.55,
                ci_low=0.49,
                ci_high=0.61,
                candidate_wins=110,
                opponent_wins=90,
                total_games=200,
            ),
            self._row(
                artifact="window-b.json",
                win_rate=0.58,
                ci_low=0.52,
                ci_high=0.64,
                candidate_wins=116,
                opponent_wins=84,
                total_games=200,
            ),
        ]
        pooled = self._row(
            artifact="pooled.json",
            win_rate=0.565,
            ci_low=0.52,
            ci_high=0.61,
            candidate_wins=226,
            opponent_wins=174,
            total_games=400,
        )

        payload = build_eval_payload([0, 100], windows, pooled)

        self.assertEqual(payload["windows"][0]["seedStartIndex"], 0)
        self.assertEqual(payload["windows"][1]["artifact"], "window-b.json")
        self.assertAlmostEqual(payload["pooled"]["candidateWinRate"], 0.565)
        self.assertEqual(payload["pooled"]["candidateWins"], 226)

    def test_evaluate_promotion_gate_reports_failed_checks(self) -> None:
        pooled = self._row(
            artifact="pooled.json",
            win_rate=0.54,
            ci_low=0.49,
            ci_high=0.59,
            side_gap=0.09,
            candidate_wins=216,
            opponent_wins=184,
            total_games=400,
            rate_a=0.585,
            rate_b=0.495,
        )
        windows = [
            self._row(
                artifact="window-a.json",
                win_rate=0.56,
                ci_low=0.50,
                ci_high=0.62,
                side_gap=0.05,
            ),
            self._row(
                artifact="window-b.json",
                win_rate=0.52,
                ci_low=0.46,
                ci_high=0.58,
                side_gap=0.12,
            ),
        ]

        gate = evaluate_promotion_gate(
            eval_row=pooled,
            eval_windows=windows,
            thresholds=PromotionThresholds(
                min_win_rate=0.55,
                max_side_gap=0.08,
                min_ci_low=0.50,
                max_window_side_gap=0.10,
            ),
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["checks"]["minWinRate"])
        self.assertFalse(gate["checks"]["maxSideGap"])
        self.assertFalse(gate["checks"]["minCiLow"])
        self.assertFalse(gate["checks"]["maxWindowSideGap"])
        self.assertEqual(gate["windowChecks"][1]["artifact"], "window-b.json")
        self.assertFalse(gate["windowChecks"][1]["maxWindowSideGap"])


if __name__ == "__main__":
    unittest.main()
