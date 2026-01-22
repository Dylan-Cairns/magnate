from __future__ import annotations

import unittest
from argparse import Namespace

from scripts.generate_teacher_data import _validate_policy_args


class GenerateTeacherDataScriptTests(unittest.TestCase):
    def test_rejects_teacher_policy_without_root_action_probs(self) -> None:
        args = Namespace(
            teacher_players="both",
            teacher_policy="random",
            opponent_policy=None,
            td_value_checkpoint=None,
            td_search_value_checkpoint=None,
            td_search_opponent_checkpoint=None,
        )
        with self.assertRaises(SystemExit) as context:
            _validate_policy_args(args)
        self.assertIn("root action probabilities", str(context.exception))

    def test_accepts_search_teacher_policy(self) -> None:
        args = Namespace(
            teacher_players="both",
            teacher_policy="search",
            opponent_policy=None,
            td_value_checkpoint=None,
            td_search_value_checkpoint=None,
            td_search_opponent_checkpoint=None,
        )
        _validate_policy_args(args)


if __name__ == "__main__":
    unittest.main()

