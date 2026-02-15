from __future__ import annotations

import unittest

from trainer.bridge_client import BridgeClient
from trainer.env import MagnateBridgeEnv


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

