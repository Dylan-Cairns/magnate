from __future__ import annotations

import unittest

from trainer.bridge_client import BridgeClient
from trainer.encoding import (
    ACTION_FEATURE_DIM,
    OBSERVATION_DIM,
    encode_action_candidates,
    encode_observation,
)
from trainer.env import MagnateBridgeEnv


class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = BridgeClient()
        cls.env = MagnateBridgeEnv(client=cls.client)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_observation_vector_has_fixed_size(self) -> None:
        state = self.env.reset(seed="encoding-observation", first_player="PlayerA")
        encoded = encode_observation(state.view)
        self.assertEqual(len(encoded), OBSERVATION_DIM)

    def test_action_features_have_fixed_size(self) -> None:
        self.env.reset(seed="encoding-actions", first_player="PlayerA")
        legal = self.env.legal_actions()
        vectors = encode_action_candidates(legal.actions)
        self.assertEqual(len(vectors), len(legal.actions))
        for vector in vectors:
            self.assertEqual(len(vector), ACTION_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()

