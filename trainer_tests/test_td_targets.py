from __future__ import annotations

import unittest

from trainer.td.targets import n_step_bootstrap_targets, td_lambda_targets


class TDTargetsTests(unittest.TestCase):
    def test_n_step_bootstrap_targets_handles_terminal_without_bootstrap(self) -> None:
        targets = n_step_bootstrap_targets(
            rewards=[1.0, 2.0, 3.0],
            dones=[False, False, True],
            next_values=[10.0, 20.0, 30.0],
            gamma=0.5,
            n_steps=2,
        )
        self.assertEqual(len(targets), 3)
        self.assertAlmostEqual(targets[0], 7.0)
        self.assertAlmostEqual(targets[1], 3.5)
        self.assertAlmostEqual(targets[2], 3.0)

    def test_td_lambda_reduces_to_td_zero_when_lambda_zero(self) -> None:
        targets = td_lambda_targets(
            rewards=[1.0, 2.0],
            dones=[False, True],
            next_values=[4.0, 5.0],
            gamma=0.5,
            lambda_=0.0,
        )
        self.assertEqual(len(targets), 2)
        self.assertAlmostEqual(targets[0], 3.0)
        self.assertAlmostEqual(targets[1], 2.0)

    def test_td_lambda_reduces_to_bootstrap_return_when_lambda_one(self) -> None:
        targets = td_lambda_targets(
            rewards=[1.0, 2.0],
            dones=[False, True],
            next_values=[4.0, 5.0],
            gamma=0.5,
            lambda_=1.0,
        )
        self.assertEqual(len(targets), 2)
        self.assertAlmostEqual(targets[0], 2.0)
        self.assertAlmostEqual(targets[1], 2.0)


if __name__ == "__main__":
    unittest.main()
