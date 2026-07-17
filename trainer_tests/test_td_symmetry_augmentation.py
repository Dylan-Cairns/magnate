from __future__ import annotations

import math
import random
import unittest
from typing import cast

from trainer.bridge_payloads import GameActionPayload
from trainer.encoding import (
    ACTION_DISTRICT_ID_FEATURE_INDEX,
    ACTION_FEATURE_DIM,
    ACTION_HAS_DISTRICT_FEATURE_INDEX,
    OBSERVATION_DIM,
    OBSERVATION_DISTRICT_COUNT,
    OBSERVATION_DISTRICT_FEATURE_DIM,
    OBSERVATION_GLOBAL_FEATURE_DIM,
    encode_action,
)
from trainer.td.symmetry_augmentation import (
    DISTRICT_AUGMENTATION_NONE,
    DISTRICT_AUGMENTATION_S4,
    DISTRICT_AUGMENTATION_S4_ORBIT,
    PAWN_DISTRICT_PERMUTATIONS,
    augment_opponent_training_batch,
    augment_value_training_batch,
    derive_augmentation_stream_seed,
    inverse_pawn_district_permutation,
    opponent_augmentation_copies_per_sample,
    permute_encoded_action_features,
    permute_encoded_observation,
    permute_opponent_sample,
    permute_value_transition,
)
from trainer.td.train import build_value_sequence_index
from trainer.td.types import OpponentSample, ValueTransition
from trainer.types import KeyedAction


def _observation(*, offset: float = 0.0) -> list[float]:
    values = [offset + (index / 1000.0) for index in range(OBSERVATION_GLOBAL_FEATURE_DIM)]
    for district in range(1, OBSERVATION_DISTRICT_COUNT + 1):
        values.extend(
            offset + district + (index / 100.0) for index in range(OBSERVATION_DISTRICT_FEATURE_DIM)
        )
    assert len(values) == OBSERVATION_DIM
    return values


def _action_features(*, district: int | None) -> list[float]:
    values = [index / 100.0 for index in range(ACTION_FEATURE_DIM)]
    if district is None:
        values[ACTION_DISTRICT_ID_FEATURE_INDEX] = 0.123
        values[ACTION_HAS_DISTRICT_FEATURE_INDEX] = 0.0
    else:
        values[ACTION_DISTRICT_ID_FEATURE_INDEX] = district / OBSERVATION_DISTRICT_COUNT
        values[ACTION_HAS_DISTRICT_FEATURE_INDEX] = 1.0
    return values


def _sequence(episode_id: str, *, offset: float = 0.0) -> list[ValueTransition]:
    first_next = _observation(offset=offset + 1.0)
    return [
        ValueTransition(
            observation=_observation(offset=offset),
            reward=0.25,
            done=False,
            next_observation=first_next,
            player_id="PlayerA",
            episode_id=episode_id,
            timestep=0,
        ),
        ValueTransition(
            observation=first_next,
            reward=1.0,
            done=True,
            next_observation=None,
            player_id="PlayerA",
            episode_id=episode_id,
            timestep=1,
        ),
    ]


class TDSymmetryAugmentationTests(unittest.TestCase):
    def test_action_layout_constants_match_the_canonical_python_encoder(self) -> None:
        action = KeyedAction(
            action_id="buy-deed",
            action_key="test",
            action=cast(
                GameActionPayload,
                {"type": "buy-deed", "cardId": "6", "districtId": "D5"},
            ),
        )
        encoded = encode_action(action)
        self.assertEqual(encoded[ACTION_DISTRICT_ID_FEATURE_INDEX], 1.0)
        self.assertEqual(encoded[ACTION_HAS_DISTRICT_FEATURE_INDEX], 1.0)

    def test_all_24_permutations_are_unique_and_keep_d3_fixed(self) -> None:
        self.assertEqual(len(PAWN_DISTRICT_PERMUTATIONS), 24)
        mappings = {permutation.destination_by_source for permutation in PAWN_DISTRICT_PERMUTATIONS}
        self.assertEqual(len(mappings), 24)
        self.assertIn((0, 1, 2, 3, 4, 5), mappings)
        for permutation in PAWN_DISTRICT_PERMUTATIONS:
            self.assertEqual(permutation.destination_by_source[3], 3)
            self.assertEqual(
                sorted(permutation.destination_by_source[index] for index in (1, 2, 4, 5)),
                [1, 2, 4, 5],
            )

    def test_observation_permutations_move_whole_blocks_and_invert(self) -> None:
        original = _observation()
        global_features = original[:OBSERVATION_GLOBAL_FEATURE_DIM]
        for permutation in PAWN_DISTRICT_PERMUTATIONS:
            transformed = permute_encoded_observation(original, permutation)
            self.assertEqual(transformed[:OBSERVATION_GLOBAL_FEATURE_DIM], global_features)
            for source in range(1, OBSERVATION_DISTRICT_COUNT + 1):
                destination = permutation.destination_by_source[source]
                source_start = OBSERVATION_GLOBAL_FEATURE_DIM + (
                    (source - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
                )
                destination_start = OBSERVATION_GLOBAL_FEATURE_DIM + (
                    (destination - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
                )
                self.assertEqual(
                    transformed[
                        destination_start : destination_start + OBSERVATION_DISTRICT_FEATURE_DIM
                    ],
                    original[source_start : source_start + OBSERVATION_DISTRICT_FEATURE_DIM],
                )
            restored = permute_encoded_observation(
                transformed, inverse_pawn_district_permutation(permutation)
            )
            self.assertEqual(restored, original)

    def test_action_permutations_remap_only_present_district(self) -> None:
        non_district = _action_features(district=None)
        for permutation in PAWN_DISTRICT_PERMUTATIONS:
            self.assertEqual(
                permute_encoded_action_features(non_district, permutation), non_district
            )
            for source in range(1, OBSERVATION_DISTRICT_COUNT + 1):
                original = _action_features(district=source)
                transformed = permute_encoded_action_features(original, permutation)
                expected = list(original)
                expected[ACTION_DISTRICT_ID_FEATURE_INDEX] = (
                    permutation.destination_by_source[source] / OBSERVATION_DISTRICT_COUNT
                )
                self.assertEqual(transformed, expected)

    def test_malformed_encodings_fail_fast(self) -> None:
        permutation = PAWN_DISTRICT_PERMUTATIONS[0]
        with self.assertRaisesRegex(ValueError, "observation length"):
            permute_encoded_observation([0.0] * (OBSERVATION_DIM - 1), permutation)
        invalid_observation = _observation()
        invalid_observation[0] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            permute_encoded_observation(invalid_observation, permutation)
        with self.assertRaisesRegex(ValueError, "action feature length"):
            permute_encoded_action_features([0.0] * (ACTION_FEATURE_DIM - 1), permutation)
        invalid_presence = _action_features(district=1)
        invalid_presence[ACTION_HAS_DISTRICT_FEATURE_INDEX] = 0.5
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            permute_encoded_action_features(invalid_presence, permutation)
        invalid_district = _action_features(district=1)
        invalid_district[ACTION_DISTRICT_ID_FEATURE_INDEX] = 0.11
        with self.assertRaisesRegex(ValueError, "D1-D5"):
            permute_encoded_action_features(invalid_district, permutation)

    def test_value_transition_uses_one_permutation_for_current_and_next(self) -> None:
        transition = _sequence("episode-one")[0]
        permutation = PAWN_DISTRICT_PERMUTATIONS[-1]
        transformed = permute_value_transition(transition, permutation)
        self.assertEqual(
            transformed.observation,
            permute_encoded_observation(transition.observation, permutation),
        )
        self.assertEqual(
            transformed.next_observation,
            permute_encoded_observation(transition.next_observation or (), permutation),
        )
        self.assertEqual(transformed.reward, transition.reward)
        self.assertEqual(transformed.episode_id, transition.episode_id)
        self.assertEqual(transformed.timestep, transition.timestep)

    def test_td_lambda_batch_transforms_entire_sequence_consistently(self) -> None:
        sequence = _sequence("episode-sequence")
        sequence_index = build_value_sequence_index(transitions=sequence)
        result = augment_value_training_batch(
            mode=DISTRICT_AUGMENTATION_S4,
            transitions=[sequence[1], sequence[0]],
            sequence_index=sequence_index,
            rng=random.Random(12),
        )
        self.assertEqual(len(result.permutation_ids), 1)
        permutation = next(
            item for item in PAWN_DISTRICT_PERMUTATIONS if item.id == result.permutation_ids[0]
        )
        self.assertIsNotNone(result.sequence_index)
        assert result.sequence_index is not None
        transformed_sequence = result.sequence_index[("episode-sequence", "PlayerA")]
        self.assertEqual(
            transformed_sequence[0].observation,
            permute_encoded_observation(sequence[0].observation, permutation),
        )
        self.assertEqual(
            transformed_sequence[0].next_observation,
            transformed_sequence[1].observation,
        )
        self.assertIs(result.transitions[0], transformed_sequence[1])
        self.assertIs(result.transitions[1], transformed_sequence[0])

    def test_sequence_index_mismatch_fails_fast(self) -> None:
        sequence = _sequence("episode-sequence")
        mismatched = list(sequence)
        mismatched[0] = ValueTransition(
            observation=_observation(offset=9.0),
            reward=0.0,
            done=False,
            next_observation=_observation(offset=10.0),
            player_id="PlayerA",
            episode_id="episode-sequence",
            timestep=0,
        )
        sequence_index = build_value_sequence_index(transitions=sequence)
        with self.assertRaisesRegex(ValueError, "does not match"):
            augment_value_training_batch(
                mode=DISTRICT_AUGMENTATION_S4,
                transitions=[mismatched[0]],
                sequence_index=sequence_index,
                rng=random.Random(2),
            )

    def test_opponent_targets_and_candidate_order_are_unchanged(self) -> None:
        action_probs = [0.1, 0.2, 0.7]
        sample = OpponentSample(
            observation=_observation(),
            action_features=[
                _action_features(district=1),
                _action_features(district=None),
                _action_features(district=5),
            ],
            action_index=2,
            action_probs=action_probs,
            player_id="PlayerB",
        )
        permutation = PAWN_DISTRICT_PERMUTATIONS[-1]
        transformed = permute_opponent_sample(sample, permutation)
        self.assertEqual(transformed.action_index, 2)
        self.assertIs(transformed.action_probs, action_probs)
        self.assertEqual(transformed.player_id, "PlayerB")
        self.assertEqual(transformed.action_features[1], sample.action_features[1])
        self.assertEqual(
            transformed.action_features[0],
            permute_encoded_action_features(sample.action_features[0], permutation),
        )
        self.assertEqual(
            transformed.action_features[2],
            permute_encoded_action_features(sample.action_features[2], permutation),
        )

    def test_complete_opponent_orbit_uses_every_permutation_in_fixed_order(self) -> None:
        first = OpponentSample(
            observation=_observation(),
            action_features=[
                _action_features(district=1),
                _action_features(district=None),
            ],
            action_index=1,
            action_probs=[0.25, 0.75],
            player_id="PlayerA",
        )
        second = OpponentSample(
            observation=_observation(offset=10.0),
            action_features=[_action_features(district=5)],
            action_index=0,
            action_probs=[1.0],
            player_id="PlayerB",
        )
        rng = random.Random(1234)
        before = rng.getstate()

        result = augment_opponent_training_batch(
            mode=DISTRICT_AUGMENTATION_S4_ORBIT,
            samples=[first, second],
            rng=rng,
        )

        self.assertEqual(len(result), 2 * len(PAWN_DISTRICT_PERMUTATIONS))
        self.assertEqual(rng.getstate(), before)
        for sample_index, source in enumerate((first, second)):
            start = sample_index * len(PAWN_DISTRICT_PERMUTATIONS)
            orbit = result[start : start + len(PAWN_DISTRICT_PERMUTATIONS)]
            expected = [
                permute_opponent_sample(source, permutation)
                for permutation in PAWN_DISTRICT_PERMUTATIONS
            ]
            self.assertEqual(orbit, expected)
            for transformed in orbit:
                self.assertEqual(transformed.action_index, source.action_index)
                self.assertIs(transformed.action_probs, source.action_probs)

        self.assertEqual(
            opponent_augmentation_copies_per_sample(mode=DISTRICT_AUGMENTATION_S4_ORBIT),
            24,
        )

    def test_complete_orbit_is_rejected_for_value_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "opponent-only"):
            augment_value_training_batch(
                mode=DISTRICT_AUGMENTATION_S4_ORBIT,
                transitions=_sequence("value-orbit"),
                sequence_index=None,
                rng=None,
            )

    def test_control_mode_is_an_exact_noop_and_consumes_no_rng(self) -> None:
        transitions = _sequence("control")
        sequence_index = build_value_sequence_index(transitions=transitions)
        samples = [
            OpponentSample(
                observation=_observation(),
                action_features=[_action_features(district=1)],
                action_index=0,
                action_probs=[1.0],
                player_id="PlayerA",
            )
        ]
        rng = random.Random(99)
        before = rng.getstate()
        value_result = augment_value_training_batch(
            mode=DISTRICT_AUGMENTATION_NONE,
            transitions=transitions,
            sequence_index=sequence_index,
            rng=rng,
        )
        opponent_result = augment_opponent_training_batch(
            mode=DISTRICT_AUGMENTATION_NONE,
            samples=samples,
            rng=rng,
        )
        self.assertIs(value_result.transitions, transitions)
        self.assertIs(value_result.sequence_index, sequence_index)
        self.assertIs(opponent_result, samples)
        self.assertEqual(rng.getstate(), before)

    def test_augmentation_rng_does_not_change_replay_sampling(self) -> None:
        control_sampling_rng = random.Random(1729)
        candidate_sampling_rng = random.Random(1729)
        augmentation_rng = random.Random(2718)
        items = list(range(100))
        for _ in range(10):
            control_batch = control_sampling_rng.sample(items, k=12)
            candidate_batch = candidate_sampling_rng.sample(items, k=12)
            self.assertEqual(candidate_batch, control_batch)
            for _item in candidate_batch:
                augmentation_rng.choice(PAWN_DISTRICT_PERMUTATIONS)

    def test_augmentation_seeds_are_reproducible_and_stream_specific(self) -> None:
        value_seed = derive_augmentation_stream_seed(base_seed=20260714, stream="value")
        self.assertEqual(
            value_seed,
            derive_augmentation_stream_seed(base_seed=20260714, stream="value"),
        )
        self.assertNotEqual(
            value_seed,
            derive_augmentation_stream_seed(base_seed=20260714, stream="opponent"),
        )
        first = augment_opponent_training_batch(
            mode=DISTRICT_AUGMENTATION_S4,
            samples=[
                OpponentSample(
                    observation=_observation(),
                    action_features=[_action_features(district=1)],
                    action_index=0,
                    action_probs=[1.0],
                    player_id="PlayerA",
                )
            ],
            rng=random.Random(value_seed),
        )
        second = augment_opponent_training_batch(
            mode=DISTRICT_AUGMENTATION_S4,
            samples=[
                OpponentSample(
                    observation=_observation(),
                    action_features=[_action_features(district=1)],
                    action_index=0,
                    action_probs=[1.0],
                    player_id="PlayerA",
                )
            ],
            rng=random.Random(value_seed),
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
