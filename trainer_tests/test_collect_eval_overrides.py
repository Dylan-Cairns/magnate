from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from scripts import collect_td_self_play, eval_suite


class CollectEvalOverrideTests(unittest.TestCase):
    def test_collect_parse_args_defaults_cache_limits_to_32(self) -> None:
        with patch.object(sys, "argv", ["collect_td_self_play.py", "--player-a-policy", "search", "--player-b-policy", "search"]):
            args = collect_td_self_play.parse_args()
        self.assertEqual(args.transition_cache_limit, 32)
        self.assertEqual(args.legal_actions_cache_limit, 32)
        self.assertEqual(args.observation_cache_limit, 32)

    def test_eval_parse_args_defaults_cache_limits_to_32(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["eval_suite.py", "--mode", "certify", "--candidate-policy", "search", "--opponent-policy", "search"],
        ):
            args = eval_suite.parse_args()
        self.assertEqual(args.transition_cache_limit, 32)
        self.assertEqual(args.legal_actions_cache_limit, 32)
        self.assertEqual(args.observation_cache_limit, 32)

    def test_collect_validate_policy_args_accepts_per_player_td_search_overrides(self) -> None:
        args = Namespace(
            player_a_policy="td-search",
            player_b_policy="search",
            td_value_checkpoint=None,
            td_search_value_checkpoint=None,
            td_search_opponent_checkpoint=None,
            player_a_td_search_value_checkpoint=Path("a.value.pt"),
            player_a_td_search_opponent_checkpoint=Path("a.opp.pt"),
            player_b_td_search_value_checkpoint=None,
            player_b_td_search_opponent_checkpoint=None,
        )
        collect_td_self_play._validate_policy_args(args)

    def test_collect_validate_policy_args_rejects_missing_per_player_td_search_overrides(self) -> None:
        args = Namespace(
            player_a_policy="td-search",
            player_b_policy="search",
            td_value_checkpoint=None,
            td_search_value_checkpoint=None,
            td_search_opponent_checkpoint=None,
            player_a_td_search_value_checkpoint=Path("a.value.pt"),
            player_a_td_search_opponent_checkpoint=None,
            player_b_td_search_value_checkpoint=None,
            player_b_td_search_opponent_checkpoint=None,
        )
        with self.assertRaises(SystemExit):
            collect_td_self_play._validate_policy_args(args)

    def test_eval_validate_policy_args_accepts_per_side_td_search_overrides(self) -> None:
        args = Namespace(
            candidate_policy="td-search",
            opponent_policy="td-search",
            td_value_checkpoint=None,
            td_search_value_checkpoint=None,
            td_search_opponent_checkpoint=None,
            candidate_td_search_value_checkpoint=Path("candidate.value.pt"),
            candidate_td_search_opponent_checkpoint=Path("candidate.opp.pt"),
            opponent_td_search_value_checkpoint=Path("opponent.value.pt"),
            opponent_td_search_opponent_checkpoint=Path("opponent.opp.pt"),
        )
        eval_suite._validate_policy_args(args)

    def test_eval_evaluate_shard_maps_per_side_td_search_payload_keys(self) -> None:
        payload = {
            "gamesPerSide": 20,
            "seedPrefix": "eval",
            "seedStartIndex": 0,
            "candidatePolicy": "td-search",
            "opponentPolicy": "td-search",
            "searchWorlds": 6,
            "searchRollouts": 1,
            "searchDepth": 14,
            "searchMaxRootActions": 6,
            "searchRolloutEpsilon": 0.04,
            "transitionCacheLimit": 64,
            "legalActionsCacheLimit": 32,
            "observationCacheLimit": 16,
            "tdValueCheckpoint": None,
            "tdWorlds": 8,
            "candidateTdSearchValueCheckpoint": "candidate.value.pt",
            "candidateTdSearchOpponentCheckpoint": "candidate.opp.pt",
            "opponentTdSearchValueCheckpoint": "opponent.value.pt",
            "opponentTdSearchOpponentCheckpoint": "opponent.opp.pt",
            "tdSearchOpponentTemperature": 1.0,
            "tdSearchSampleOpponentActions": False,
            "progressEveryGames": 10,
            "progressLogMinutes": 30.0,
            "workerTorchThreads": 1,
            "workerTorchInteropThreads": 1,
            "workerBlasThreads": 1,
        }
        with patch("scripts.eval_suite._run_eval_shard", return_value={"ok": True}) as mocked:
            result = eval_suite._evaluate_shard(payload)
        self.assertEqual(result, {"ok": True})
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["candidate_td_search_value_checkpoint"], Path("candidate.value.pt"))
        self.assertEqual(kwargs["candidate_td_search_opponent_checkpoint"], Path("candidate.opp.pt"))
        self.assertEqual(kwargs["opponent_td_search_value_checkpoint"], Path("opponent.value.pt"))
        self.assertEqual(kwargs["opponent_td_search_opponent_checkpoint"], Path("opponent.opp.pt"))
        self.assertEqual(kwargs["transition_cache_limit"], 64)
        self.assertEqual(kwargs["legal_actions_cache_limit"], 32)
        self.assertEqual(kwargs["observation_cache_limit"], 16)


if __name__ == "__main__":
    unittest.main()
