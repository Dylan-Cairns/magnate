import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import type { LoadedTdGuidanceModel } from '../policies/tdGuidanceModel';
import {
  ACTION_DISTRICT_ID_FEATURE_INDEX,
  ACTION_FEATURE_DIM,
  ACTION_HAS_DISTRICT_FEATURE_INDEX,
  OBSERVATION_DIM,
  OBSERVATION_DISTRICT_FEATURE_DIM,
  OBSERVATION_GLOBAL_FEATURE_DIM,
} from '../policies/trainingEncoding';
import type { TdReplayOpponentSamplePayload } from './types';
import {
  createPawnDistrictPermutations,
  permuteEncodedActionFeatures,
  permuteEncodedObservation,
  runTdSymmetryAudit,
  sampleOpponentReplayDirectory,
} from './tdSymmetry';

describe('TD district symmetry', () => {
  it('creates all 24 Pawn-district permutations while fixing D3', () => {
    const permutations = createPawnDistrictPermutations();

    expect(permutations).toHaveLength(24);
    expect(new Set(permutations.map((entry) => entry.id))).toHaveLength(24);
    for (const permutation of permutations) {
      expect(permutation.destinationBySource[3]).toBe(3);
      expect(
        [1, 2, 4, 5]
          .map((source) => permutation.destinationBySource[source])
          .sort((left, right) => left - right)
      ).toEqual([1, 2, 4, 5]);
    }
  });

  it('moves complete observation blocks and matching action district features', () => {
    const observation = makeObservation();
    const swapD1D5 = createPawnDistrictPermutations().find(
      (entry) =>
        entry.destinationBySource[1] === 5 &&
        entry.destinationBySource[5] === 1 &&
        entry.destinationBySource[2] === 2 &&
        entry.destinationBySource[4] === 4
    );
    expect(swapD1D5).toBeDefined();

    const transformed = permuteEncodedObservation(observation, swapD1D5!);
    expect(firstDistrictFeature(transformed, 1)).toBe(5);
    expect(firstDistrictFeature(transformed, 5)).toBe(1);
    expect(firstDistrictFeature(transformed, 3)).toBe(3);

    const districtAction = makeActionFeatures(1);
    const transformedAction = permuteEncodedActionFeatures(
      districtAction,
      swapD1D5!
    );
    expect(transformedAction[ACTION_DISTRICT_ID_FEATURE_INDEX]).toBe(1);

    const nonDistrictAction = new Array<number>(ACTION_FEATURE_DIM).fill(0);
    expect(permuteEncodedActionFeatures(nonDistrictAction, swapD1D5!)).toEqual(
      nonDistrictAction
    );
  });

  it('reports zero drift for a symmetric model', () => {
    const run = runTdSymmetryAudit({
      samples: [makeReplaySample()],
      rowsScanned: 1,
      sourceFiles: ['shard-000.opponent.jsonl'],
      samplingSeed: 'test',
      requestedSampleSize: 1,
      modelPackId: 'symmetric-mock',
      modelIndexPath: 'mock-index.json',
      model: makeSymmetricModel(),
    });

    expect(run.aggregate.samplesWithAnyTopActionFlip).toBe(0);
    expect(run.aggregate.jensenShannonDivergence.max).toBe(0);
    expect(run.aggregate.maxProbabilityDelta.max).toBe(0);
    expect(run.aggregate.maxCenteredLogitDelta.max).toBe(0);
    expect(run.aggregate.valueAbsoluteDelta.max).toBe(0);
  });

  it('detects a model that directly favors physical district numbers', () => {
    const run = runTdSymmetryAudit({
      samples: [makeReplaySample()],
      rowsScanned: 1,
      sourceFiles: ['shard-000.opponent.jsonl'],
      samplingSeed: 'test',
      requestedSampleSize: 1,
      modelPackId: 'biased-mock',
      modelIndexPath: 'mock-index.json',
      model: makeDistrictBiasedModel(),
    });

    expect(run.aggregate.samplesWithAnyTopActionFlip).toBe(1);
    expect(run.aggregate.maxProbabilityDelta.max).toBeGreaterThan(0);
    expect(run.aggregate.maxCenteredLogitDelta.max).toBeGreaterThan(0);
    expect(run.aggregate.valueAbsoluteDelta.max).toBeGreaterThan(0);
    expect(
      run.pawnSlotEffects.find((entry) => entry.districtId === 'D5')
        ?.meanCenteredLogit
    ).toBeGreaterThan(
      run.pawnSlotEffects.find((entry) => entry.districtId === 'D1')
        ?.meanCenteredLogit ?? Number.POSITIVE_INFINITY
    );
  });

  it('samples all replay files deterministically', async () => {
    const directory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-symmetry-')
    );
    try {
      const rows = Array.from({ length: 10 }, (_unused, index) => ({
        ...makeRow(),
        actionIndex: index % 2,
      }));
      await writeFile(
        path.join(directory, 'shard-000.opponent.jsonl'),
        `${rows
          .slice(0, 5)
          .map((row) => JSON.stringify(row))
          .join('\n')}\n`,
        'utf8'
      );
      await writeFile(
        path.join(directory, 'shard-001.opponent.jsonl'),
        `${rows
          .slice(5)
          .map((row) => JSON.stringify(row))
          .join('\n')}\n`,
        'utf8'
      );

      const first = await sampleOpponentReplayDirectory(directory, 4, 'seed');
      const second = await sampleOpponentReplayDirectory(directory, 4, 'seed');

      expect(first.rowsScanned).toBe(10);
      expect(first.files).toHaveLength(2);
      expect(
        first.samples.map((sample) => [sample.sourcePath, sample.sourceLine])
      ).toEqual(
        second.samples.map((sample) => [sample.sourcePath, sample.sourceLine])
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

function makeReplaySample() {
  return {
    sourcePath: 'shard-000.opponent.jsonl',
    sourceLine: 1,
    row: makeRow(),
  };
}

function makeRow(): TdReplayOpponentSamplePayload {
  return {
    observation: makeObservation(),
    actionFeatures: [makeActionFeatures(1), makeActionFeatures(5)],
    actionIndex: 1,
    actionProbs: [0.5, 0.5],
    playerId: 'PlayerA',
  };
}

function makeObservation(): number[] {
  const observation = new Array<number>(OBSERVATION_DIM).fill(0);
  for (let district = 1; district <= 5; district += 1) {
    observation[
      OBSERVATION_GLOBAL_FEATURE_DIM +
        (district - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
    ] = district;
  }
  return observation;
}

function firstDistrictFeature(
  observation: readonly number[],
  district: number
): number {
  return observation[
    OBSERVATION_GLOBAL_FEATURE_DIM +
      (district - 1) * OBSERVATION_DISTRICT_FEATURE_DIM
  ];
}

function makeActionFeatures(district: number): number[] {
  const features = new Array<number>(ACTION_FEATURE_DIM).fill(0);
  features[0] = 1;
  features[ACTION_DISTRICT_ID_FEATURE_INDEX] = district / 5;
  features[ACTION_HAS_DISTRICT_FEATURE_INDEX] = 1;
  return features;
}

function makeSymmetricModel(): LoadedTdGuidanceModel {
  return {
    valueScorer: {
      observationDim: OBSERVATION_DIM,
      predict(observation) {
        return observation.reduce((sum, value) => sum + value, 0);
      },
    },
    opponentScorer: {
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
      logits(_observation, actionFeatures) {
        return Float32Array.from(actionFeatures.map((features) => features[0]));
      },
    },
  };
}

function makeDistrictBiasedModel(): LoadedTdGuidanceModel {
  return {
    valueScorer: {
      observationDim: OBSERVATION_DIM,
      predict(observation) {
        return firstDistrictFeature(observation, 5);
      },
    },
    opponentScorer: {
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
      logits(_observation, actionFeatures) {
        return Float32Array.from(
          actionFeatures.map(
            (features) => features[ACTION_DISTRICT_ID_FEATURE_INDEX]
          )
        );
      },
    },
  };
}
