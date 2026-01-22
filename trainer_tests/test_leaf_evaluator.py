from __future__ import annotations

import unittest
from typing import cast

from trainer.bridge_payloads import PlayerViewPayload, ResourcePoolPayload
from trainer.search.leaf_evaluator import (
    active_value_to_root_value,
    value_from_player_view,
)


def _resource_pool(
    *,
    moons: int = 0,
    suns: int = 0,
    waves: int = 0,
    leaves: int = 0,
    wyrms: int = 0,
    knots: int = 0,
) -> ResourcePoolPayload:
    return {
        "Moons": moons,
        "Suns": suns,
        "Waves": waves,
        "Leaves": leaves,
        "Wyrms": wyrms,
        "Knots": knots,
    }


def _player_view() -> PlayerViewPayload:
    return {
        "viewerId": "PlayerA",
        "activePlayerId": "PlayerA",
        "turn": 0,
        "phase": "ActionWindow",
        "districts": [
            {
                "id": "D0",
                "markerSuitMask": [],
                "stacks": {
                    "PlayerA": {
                        "developed": ["8", "17"],
                        "deed": {"cardId": "20", "progress": 3, "tokens": {"Wyrms": 2}},
                    },
                    "PlayerB": {
                        "developed": ["6"],
                    },
                },
            }
        ],
        "players": [
            {
                "id": "PlayerA",
                "crowns": [],
                "resources": _resource_pool(moons=4, suns=2),
                "hand": [],
                "handCount": 3,
                "handHidden": False,
            },
            {
                "id": "PlayerB",
                "crowns": [],
                "resources": _resource_pool(moons=1),
                "hand": [],
                "handCount": 1,
                "handHidden": True,
            },
        ],
        "deck": {"drawCount": 12, "discard": [], "reshuffles": 0},
        "cardPlayedThisTurn": False,
        "log": [],
    }


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
        view = _player_view()

        player_a_value = value_from_player_view(view, "PlayerA")
        player_b_value = value_from_player_view(view, "PlayerB")

        self.assertGreater(player_a_value, 0.0)
        self.assertLess(player_b_value, 0.0)

    def test_value_from_player_view_requires_districts_list(self) -> None:
        bad_view = dict(_player_view())
        bad_view.pop("districts")

        with self.assertRaises(KeyError):
            value_from_player_view(cast(PlayerViewPayload, bad_view), "PlayerA")


if __name__ == "__main__":
    unittest.main()
