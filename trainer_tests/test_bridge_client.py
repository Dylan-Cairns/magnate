from __future__ import annotations

import unittest

from trainer.bridge_client import (
    BridgeClient,
    _legal_actions_result_from_bridge,
    _metadata_from_bridge,
    _observation_result_from_bridge,
    _state_result_from_bridge,
)
from trainer.env import MagnateBridgeEnv


class BridgePayloadAdapterTests(unittest.TestCase):
    def test_state_result_adapter_uses_shallow_state_and_view_payloads(self) -> None:
        state: dict[str, object] = {"schemaVersion": 1}
        view: dict[str, object] = {"viewerId": "PlayerA"}

        result = _state_result_from_bridge(
            {"state": state, "view": view, "terminal": False}
        )

        self.assertIs(result.state, state)
        self.assertIs(result.view, view)
        self.assertFalse(result.terminal)

    def test_legal_actions_adapter_preserves_order_and_action_payloads(self) -> None:
        end_turn: dict[str, object] = {"type": "end-turn"}
        trade: dict[str, object] = {"type": "trade", "give": "Moons", "receive": "Suns"}
        payload: dict[str, object] = {
            "actions": [
                {"actionId": "end-turn", "actionKey": "b", "action": end_turn},
                {"actionId": "trade", "actionKey": "a", "action": trade},
            ],
            "activePlayerId": "PlayerA",
            "phase": "ActionWindow",
        }

        result = _legal_actions_result_from_bridge(payload)

        self.assertEqual([action.action_key for action in result.actions], ["b", "a"])
        self.assertIs(result.actions[0].action, end_turn)
        self.assertIs(result.actions[1].action, trade)
        self.assertEqual(result.active_player_id, "PlayerA")
        self.assertEqual(result.phase, "ActionWindow")

    def test_legal_actions_adapter_rejects_malformed_top_level_fields(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "legalActions.actions"):
            _legal_actions_result_from_bridge(
                {"actions": {}, "activePlayerId": "PlayerA", "phase": "ActionWindow"}
            )

        with self.assertRaisesRegex(RuntimeError, "legalActions.activePlayerId"):
            _legal_actions_result_from_bridge(
                {
                    "actions": [],
                    "activePlayerId": "PlayerC",
                    "phase": "ActionWindow",
                }
            )

    def test_observation_adapter_handles_optional_mask(self) -> None:
        view: dict[str, object] = {"viewerId": "PlayerA"}
        self.assertIsNone(
            _observation_result_from_bridge({"view": view}).legal_action_mask
        )

        result = _observation_result_from_bridge(
            {"view": view, "legalActionMask": ["a", "b"]}
        )

        self.assertEqual(result.legal_action_mask, ["a", "b"])
        self.assertIs(result.view, view)

    def test_observation_adapter_rejects_malformed_mask(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "observation.legalActionMask"):
            _observation_result_from_bridge(
                {"view": {"viewerId": "PlayerA"}, "legalActionMask": "a"}
            )

    def test_metadata_adapter_checks_contract_identity(self) -> None:
        metadata = _metadata_from_bridge(
            {"contractName": "magnate_bridge", "contractVersion": "v1"}
        )
        self.assertEqual(metadata["contractName"], "magnate_bridge")
        self.assertEqual(metadata["contractVersion"], "v1")

        with self.assertRaisesRegex(RuntimeError, "metadata.contractVersion"):
            _metadata_from_bridge(
                {"contractName": "magnate_bridge", "contractVersion": "v2"}
            )


class BridgeClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_metadata_contract_identity(self) -> None:
        metadata = self.client.metadata()
        self.assertEqual(metadata["contractName"], "magnate_bridge")
        self.assertEqual(metadata["contractVersion"], "v1")

    def test_legal_actions_canonical_key_order(self) -> None:
        actions = self.client.legal_actions().actions
        keys = [action.action_key for action in actions]
        self.assertEqual(keys, sorted(keys))

    def test_step_by_action_key_advances_log(self) -> None:
        env = MagnateBridgeEnv(client=self.client)
        result = env.reset(seed="bridge-client-step", first_player="PlayerA")
        starting_log_count = len(result.state.get("log", []))
        legal = env.legal_actions()
        self.assertGreater(len(legal.actions), 0)
        step = env.step(action_key=legal.actions[0].action_key)
        self.assertGreater(len(step.state.get("log", [])), starting_log_count)


if __name__ == "__main__":
    unittest.main()

