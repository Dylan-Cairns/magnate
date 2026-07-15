from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from trainer.encoding import (
    ACTION_DISTRICT_ID_FEATURE_INDEX,
    ACTION_FEATURE_DIM,
    ACTION_HAS_DISTRICT_FEATURE_INDEX,
    OBSERVATION_DIM,
    OBSERVATION_DISTRICT_COUNT,
    OBSERVATION_DISTRICT_FEATURE_DIM,
    OBSERVATION_GLOBAL_FEATURE_DIM,
)
from trainer.types import PlayerId

from .types import OpponentSample, ValueTransition

DISTRICT_AUGMENTATION_NONE = "none"
DISTRICT_AUGMENTATION_S4 = "pawn-district-s4-d3-fixed-v1"
DISTRICT_AUGMENTATION_MODES = frozenset(
    (DISTRICT_AUGMENTATION_NONE, DISTRICT_AUGMENTATION_S4)
)
PAWN_DISTRICT_NUMBERS = (1, 2, 4, 5)
_IDENTITY_DESTINATION_BY_SOURCE = (0, 1, 2, 3, 4, 5)
_EPSILON = 1e-9

SequenceKey: TypeAlias = tuple[str, PlayerId]


@dataclass(frozen=True)
class PawnDistrictPermutation:
    id: str
    destination_by_source: tuple[int, ...]


@dataclass(frozen=True)
class ValueAugmentationBatch:
    transitions: Sequence[ValueTransition]
    sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]] | None
    permutation_ids: tuple[str, ...]


def create_pawn_district_permutations() -> tuple[PawnDistrictPermutation, ...]:
    out: list[PawnDistrictPermutation] = []
    for destinations in itertools.permutations(PAWN_DISTRICT_NUMBERS):
        destination_by_source = list(_IDENTITY_DESTINATION_BY_SOURCE)
        for source, destination in zip(PAWN_DISTRICT_NUMBERS, destinations, strict=True):
            destination_by_source[source] = destination
        mapping = tuple(destination_by_source)
        out.append(
            PawnDistrictPermutation(
                id=",".join(
                    f"D{source}>D{mapping[source]}" for source in PAWN_DISTRICT_NUMBERS
                ),
                destination_by_source=mapping,
            )
        )
    return tuple(out)


PAWN_DISTRICT_PERMUTATIONS = create_pawn_district_permutations()


def inverse_pawn_district_permutation(
    permutation: PawnDistrictPermutation,
) -> PawnDistrictPermutation:
    _validate_permutation(permutation)
    destination_by_source = list(_IDENTITY_DESTINATION_BY_SOURCE)
    for source in PAWN_DISTRICT_NUMBERS:
        destination_by_source[permutation.destination_by_source[source]] = source
    mapping = tuple(destination_by_source)
    return PawnDistrictPermutation(
        id=",".join(f"D{source}>D{mapping[source]}" for source in PAWN_DISTRICT_NUMBERS),
        destination_by_source=mapping,
    )


def permute_encoded_observation(
    observation: Sequence[float],
    permutation: PawnDistrictPermutation,
) -> list[float]:
    _validate_observation(observation)
    _validate_permutation(permutation)
    result = list(observation)
    for source in range(1, OBSERVATION_DISTRICT_COUNT + 1):
        destination = permutation.destination_by_source[source]
        source_offset = OBSERVATION_GLOBAL_FEATURE_DIM + (
            (source - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
        )
        destination_offset = OBSERVATION_GLOBAL_FEATURE_DIM + (
            (destination - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
        )
        result[
            destination_offset : destination_offset + OBSERVATION_DISTRICT_FEATURE_DIM
        ] = observation[source_offset : source_offset + OBSERVATION_DISTRICT_FEATURE_DIM]
    return result


def permute_encoded_action_features(
    action_features: Sequence[float],
    permutation: PawnDistrictPermutation,
) -> list[float]:
    _validate_action_features(action_features)
    _validate_permutation(permutation)
    result = list(action_features)
    has_district = float(result[ACTION_HAS_DISTRICT_FEATURE_INDEX])
    if math.isclose(has_district, 0.0, abs_tol=_EPSILON):
        return result
    if not math.isclose(has_district, 1.0, abs_tol=_EPSILON):
        raise ValueError("Action has-district feature must be encoded as 0 or 1.")

    encoded_district = float(result[ACTION_DISTRICT_ID_FEATURE_INDEX])
    source = round(encoded_district * OBSERVATION_DISTRICT_COUNT)
    if (
        source < 1
        or source > OBSERVATION_DISTRICT_COUNT
        or not math.isclose(
            encoded_district,
            source / OBSERVATION_DISTRICT_COUNT,
            abs_tol=_EPSILON,
        )
    ):
        raise ValueError("Action district feature does not encode D1-D5.")
    result[ACTION_DISTRICT_ID_FEATURE_INDEX] = (
        permutation.destination_by_source[source] / OBSERVATION_DISTRICT_COUNT
    )
    return result


def permute_value_transition(
    transition: ValueTransition,
    permutation: PawnDistrictPermutation,
) -> ValueTransition:
    return ValueTransition(
        observation=permute_encoded_observation(transition.observation, permutation),
        reward=transition.reward,
        done=transition.done,
        next_observation=(
            permute_encoded_observation(transition.next_observation, permutation)
            if transition.next_observation is not None
            else None
        ),
        player_id=transition.player_id,
        episode_id=transition.episode_id,
        timestep=transition.timestep,
    )


def permute_opponent_sample(
    sample: OpponentSample,
    permutation: PawnDistrictPermutation,
) -> OpponentSample:
    if len(sample.action_features) == 0:
        raise ValueError("Opponent sample must contain at least one candidate action.")
    if len(sample.action_features) != len(sample.action_probs):
        raise ValueError("Opponent sample action_features/action_probs length mismatch.")
    if sample.action_index < 0 or sample.action_index >= len(sample.action_features):
        raise ValueError("Opponent sample action_index is out of bounds.")
    return OpponentSample(
        observation=permute_encoded_observation(sample.observation, permutation),
        action_features=[
            permute_encoded_action_features(features, permutation)
            for features in sample.action_features
        ],
        action_index=sample.action_index,
        action_probs=sample.action_probs,
        player_id=sample.player_id,
    )


def augment_value_training_batch(
    *,
    mode: str,
    transitions: Sequence[ValueTransition],
    sequence_index: Mapping[SequenceKey, Sequence[ValueTransition]] | None,
    rng: random.Random | None,
) -> ValueAugmentationBatch:
    _validate_mode(mode)
    if mode == DISTRICT_AUGMENTATION_NONE:
        return ValueAugmentationBatch(
            transitions=transitions,
            sequence_index=sequence_index,
            permutation_ids=(),
        )
    augmentation_rng = _require_rng(rng)

    transformed_transitions: list[ValueTransition] = []
    transformed_sequences: dict[SequenceKey, tuple[ValueTransition, ...]] = {}
    permutations_by_key: dict[object, PawnDistrictPermutation] = {}
    permutation_ids: list[str] = []

    for row_index, transition in enumerate(transitions):
        sequence_key = _optional_sequence_key(transition)
        lookup_key: object = sequence_key if sequence_key is not None else ("row", row_index)
        permutation = permutations_by_key.get(lookup_key)
        if permutation is None:
            permutation = augmentation_rng.choice(PAWN_DISTRICT_PERMUTATIONS)
            permutations_by_key[lookup_key] = permutation
            permutation_ids.append(permutation.id)

        if sequence_index is not None:
            if sequence_key is None or transition.timestep is None:
                raise ValueError(
                    "Sequence-aware value augmentation requires episode_id and timestep."
                )
            sequence = sequence_index.get(sequence_key)
            if sequence is None:
                raise ValueError(
                    "Missing sequence for augmented value transition. "
                    f"episodeId={sequence_key[0]!r} playerId={sequence_key[1]}"
                )
            if transition.timestep < 0 or transition.timestep >= len(sequence):
                raise ValueError("Augmented value transition timestep is out of range.")
            if sequence[transition.timestep] != transition:
                raise ValueError(
                    "Sampled value transition does not match its sequence-index row."
                )
            if sequence_key not in transformed_sequences:
                transformed_sequences[sequence_key] = tuple(
                    permute_value_transition(item, permutation) for item in sequence
                )
            transformed_transitions.append(
                transformed_sequences[sequence_key][transition.timestep]
            )
        else:
            transformed_transitions.append(
                permute_value_transition(transition, permutation)
            )

    return ValueAugmentationBatch(
        transitions=transformed_transitions,
        sequence_index=transformed_sequences if sequence_index is not None else None,
        permutation_ids=tuple(permutation_ids),
    )


def augment_opponent_training_batch(
    *,
    mode: str,
    samples: Sequence[OpponentSample],
    rng: random.Random | None,
) -> Sequence[OpponentSample]:
    _validate_mode(mode)
    if mode == DISTRICT_AUGMENTATION_NONE:
        return samples
    augmentation_rng = _require_rng(rng)
    return [
        permute_opponent_sample(
            sample,
            augmentation_rng.choice(PAWN_DISTRICT_PERMUTATIONS),
        )
        for sample in samples
    ]


def derive_augmentation_stream_seed(*, base_seed: int, stream: str) -> int:
    if not stream:
        raise ValueError("augmentation stream must not be empty.")
    digest = hashlib.sha256(f"{base_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _optional_sequence_key(transition: ValueTransition) -> SequenceKey | None:
    if transition.episode_id is None:
        return None
    return (transition.episode_id, transition.player_id)


def _validate_observation(observation: Sequence[float]) -> None:
    if len(observation) != OBSERVATION_DIM:
        raise ValueError(
            "Symmetry observation length mismatch. "
            f"expected={OBSERVATION_DIM} actual={len(observation)}."
        )
    if (
        OBSERVATION_GLOBAL_FEATURE_DIM
        + OBSERVATION_DISTRICT_COUNT * OBSERVATION_DISTRICT_FEATURE_DIM
        != OBSERVATION_DIM
    ):
        raise ValueError("Symmetry observation layout constants do not match encoding dimension.")
    if not all(math.isfinite(float(value)) for value in observation):
        raise ValueError("Symmetry observation must contain finite values.")


def _validate_action_features(action_features: Sequence[float]) -> None:
    if len(action_features) != ACTION_FEATURE_DIM:
        raise ValueError(
            "Symmetry action feature length mismatch. "
            f"expected={ACTION_FEATURE_DIM} actual={len(action_features)}."
        )
    if not all(math.isfinite(float(value)) for value in action_features):
        raise ValueError("Symmetry action features must contain finite values.")


def _validate_permutation(permutation: PawnDistrictPermutation) -> None:
    if len(permutation.destination_by_source) != OBSERVATION_DISTRICT_COUNT + 1:
        raise ValueError("Pawn district permutation must index D1-D5.")
    if permutation.destination_by_source[3] != 3:
        raise ValueError("Pawn district permutation must keep D3 fixed.")
    destinations = sorted(
        permutation.destination_by_source[source] for source in PAWN_DISTRICT_NUMBERS
    )
    if destinations != list(PAWN_DISTRICT_NUMBERS):
        raise ValueError("Pawn district permutation must be a bijection over D1,D2,D4,D5.")


def _validate_mode(mode: str) -> None:
    if mode not in DISTRICT_AUGMENTATION_MODES:
        raise ValueError(
            f"district augmentation mode must be one of {sorted(DISTRICT_AUGMENTATION_MODES)}."
        )


def _require_rng(rng: random.Random | None) -> random.Random:
    if rng is None:
        raise ValueError("S4 district augmentation requires a dedicated RNG.")
    return rng
