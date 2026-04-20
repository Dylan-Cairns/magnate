from __future__ import annotations

import unittest

from trainer.search.leaf_evaluator import (
    active_value_to_root_value,
    value_from_player_view,
)


class LeafEvaluatorTests(unittest.TestCase):
    def test_active_value_to_root_value_flips_for_opponent(self) -> None:
        self.assertAlmostEqual(
            active_value_to_root_value(
                active_value=0.4,
                active_player="PlayerA",
                root_player="PlayerA",
            ),
            0.4,
        )
        self.assertAlmostEqual(
            active_value_to_root_value(
                active_value=0.4,
                active_player="PlayerB",
                root_player="PlayerA",
            ),
            -0.4,
        )

    def test_value_from_player_view_favors_stronger_root_position(self) -> None:
        view = {
            "players": [
                {
                    "id": "PlayerA",
                    "resources": {"clubs": 4, "diamonds": 2},
                    "handCount": 3,
                },
                {
                    "id": "PlayerB",
                    "resources": {"clubs": 1, "diamonds": 0},
                    "handCount": 1,
                },
            ],
            "districts": [
                {
                    "stacks": {
                        "PlayerA": {
                            "developed": ["C3", "D4"],
                            "deed": {"cardId": "H5", "progress": 3},
                        },
                        "PlayerB": {
                            "developed": ["S2"],
                            "deed": None,
                        },
                    }
                }
            ],
        }

        player_a_value = value_from_player_view(view, "PlayerA")
        player_b_value = value_from_player_view(view, "PlayerB")

        self.assertGreater(player_a_value, 0.0)
        self.assertLess(player_b_value, 0.0)

    def test_value_from_player_view_requires_districts_list(self) -> None:
        view = {
            "players": [
                {"id": "PlayerA", "resources": {"clubs": 1}, "handCount": 1},
                {"id": "PlayerB", "resources": {"clubs": 1}, "handCount": 1},
            ]
        }

        with self.assertRaises(ValueError):
            value_from_player_view(view, "PlayerA")


if __name__ == "__main__":
    unittest.main()
