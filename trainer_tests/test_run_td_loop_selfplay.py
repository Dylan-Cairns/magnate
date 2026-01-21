from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.opponent_pool import PoolCheckpoint
from scripts.run_td_loop_selfplay import (
    EvalRow,
    _collect_profile_seed_prefix,
    _build_collect_profiles,
    _promotion_decision,
    _validate_args,
    parse_args,
)


class RunTdLoopSelfplayTests(unittest.TestCase):
    def _parse(self, *args: str) -> Namespace:
        with patch.object(sys, "argv", ["run_td_loop_selfplay.py", *args]):
            return parse_args()

    def _args(self) -> Namespace:
        return Namespace(
            collect_games=100,
            collect_selfplay_share=0.60,
            collect_pool_share=0.25,
            collect_search_anchor_share=0.15,
            promotion_min_win_rate=0.55,
            promotion_max_side_gap=0.08,
            promotion_min_ci_low=0.50,
            promotion_max_window_side_gap=0.10,
            promotion_incumbent_min_win_rate=0.52,
            promotion_incumbent_max_side_gap=0.08,
            promotion_incumbent_min_ci_low=0.50,
            promotion_incumbent_max_window_side_gap=0.10,
        )

    def test_parse_args_defaults_to_12_chunks_per_loop(self) -> None:
        args = self._parse()
        self.assertEqual(args.chunks_per_loop, 12)

    def test_build_collect_profiles_includes_selfplay_pool_and_search_anchor(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = PoolCheckpoint(
                run_id="candidate",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=root / "candidate.value.pt",
                opponent_path=root / "candidate.opp.pt",
            )
            candidate.value_path.write_text("v", encoding="utf-8")
            candidate.opponent_path.write_text("o", encoding="utf-8")
            older = PoolCheckpoint(
                run_id="older",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=root / "older.value.pt",
                opponent_path=root / "older.opp.pt",
            )
            older.value_path.write_text("v", encoding="utf-8")
            older.opponent_path.write_text("o", encoding="utf-8")
            profiles = _build_collect_profiles(
                args=args,
                candidate=candidate,
                promoted_pool=[candidate, older],
            )
        self.assertEqual(sum(profile.games for profile in profiles), args.collect_games)
        labels = {profile.label for profile in profiles}
        self.assertIn("selfplay", labels)
        self.assertIn("search-anchor", labels)
        self.assertIn("pool-01", labels)

    def test_build_collect_profiles_folds_pool_share_when_pool_empty_and_selfplay_zero(self) -> None:
        args = Namespace(
            collect_games=100,
            collect_selfplay_share=0.0,
            collect_pool_share=0.5,
            collect_search_anchor_share=0.5,
            promotion_min_win_rate=0.55,
            promotion_max_side_gap=0.08,
            promotion_min_ci_low=0.50,
            promotion_max_window_side_gap=0.10,
            promotion_incumbent_min_win_rate=0.52,
            promotion_incumbent_max_side_gap=0.08,
            promotion_incumbent_min_ci_low=0.50,
            promotion_incumbent_max_window_side_gap=0.10,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = PoolCheckpoint(
                run_id="candidate",
                generated_at_utc="2026-01-01T00:00:00+00:00",
                value_path=root / "candidate.value.pt",
                opponent_path=root / "candidate.opp.pt",
            )
            candidate.value_path.write_text("v", encoding="utf-8")
            candidate.opponent_path.write_text("o", encoding="utf-8")
            profiles = _build_collect_profiles(
                args=args,
                candidate=candidate,
                promoted_pool=[],
            )
        self.assertEqual(sum(profile.games for profile in profiles), args.collect_games)
        by_label = {profile.label: profile.games for profile in profiles}
        self.assertEqual(by_label["search-anchor"], 50)
        self.assertEqual(by_label["selfplay"], 50)

    def test_dual_promotion_gate_requires_both_baseline_and_incumbent(self) -> None:
        args = self._args()
        baseline = EvalRow(
            artifact=Path("baseline.json"),
            opponent_policy="search",
            candidate_win_rate=0.57,
            ci_low=0.52,
            ci_high=0.62,
            side_gap=0.04,
            candidate_wins=456,
            opponent_wins=344,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.59,
            candidate_win_rate_as_player_b=0.55,
        )
        incumbent = EvalRow(
            artifact=Path("incumbent.json"),
            opponent_policy="td-search",
            candidate_win_rate=0.50,
            ci_low=0.46,
            ci_high=0.54,
            side_gap=0.02,
            candidate_wins=400,
            opponent_wins=400,
            draws=0,
            total_games=800,
            candidate_win_rate_as_player_a=0.51,
            candidate_win_rate_as_player_b=0.49,
        )
        result = _promotion_decision(
            baseline_eval=baseline,
            baseline_windows=[baseline],
            incumbent_eval=incumbent,
            incumbent_windows=[incumbent],
            args=args,
        )
        self.assertFalse(result["promoted"])
        self.assertFalse(result["checks"]["candidateVsIncumbent"]["minWinRate"])

    def test_collect_profile_seed_prefix_includes_run_scope(self) -> None:
        prefix_a = _collect_profile_seed_prefix(
            collect_seed_prefix="collect",
            run_id="run-chunk-001",
            profile_label="selfplay",
        )
        prefix_b = _collect_profile_seed_prefix(
            collect_seed_prefix="collect",
            run_id="run-chunk-002",
            profile_label="selfplay",
        )
        self.assertEqual(prefix_a, "collect-run-chunk-001-selfplay")
        self.assertEqual(prefix_b, "collect-run-chunk-002-selfplay")
        self.assertNotEqual(prefix_a, prefix_b)

    def test_validate_args_rejects_train_disable_value(self) -> None:
        args = self._parse("--train-disable-value")
        with self.assertRaises(SystemExit):
            _validate_args(args)

    def test_validate_args_rejects_train_disable_opponent(self) -> None:
        args = self._parse("--train-disable-opponent")
        with self.assertRaises(SystemExit):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
