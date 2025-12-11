import { describe, expect, it } from 'vitest';

import { ENCODING_VERSION, OBSERVATION_DIM } from './trainingEncoding';
import {
  createTdValueNetworkFromWeights,
  parseModelPackManifest,
  type TdValueModelPackManifest,
} from './tdValueModelPack';

function makeManifest(hiddenDim: number): TdValueModelPackManifest {
  return {
    schemaVersion: 1,
    packId: 'test-pack',
    label: 'Test Pack',
    createdAtUtc: '2026-03-05T00:00:00Z',
    model: {
      modelType: 'td-value-v1',
      checkpointType: 'magnate_td_value_v1',
      encodingVersion: ENCODING_VERSION,
      observationDim: OBSERVATION_DIM,
      hiddenDim,
      weightsPath: 'weights.json',
      requiredStateDictKeys: [
        'encoder.0.weight',
        'encoder.0.bias',
        'encoder.2.weight',
        'encoder.2.bias',
        'encoder.4.weight',
        'encoder.4.bias',
      ],
    },
    source: {
      runId: 'run-001',
      valueCheckpoint: 'artifacts/run-001/value.pt',
      checkpointMetadata: { step: 1 },
    },
  };
}

describe('td value model pack', () => {
  it('parses manifest with expected schema and dimensions', () => {
    const manifest = parseModelPackManifest(makeManifest(2));
    expect(manifest.model.encodingVersion).toBe(ENCODING_VERSION);
    expect(manifest.model.observationDim).toBe(OBSERVATION_DIM);
    expect(manifest.model.hiddenDim).toBe(2);
  });

  it('runs a deterministic forward pass from exported tensors', () => {
    const manifest = makeManifest(2);
    const w1 = new Array<number>(OBSERVATION_DIM * 2).fill(0);
    w1[0] = 1;
    w1[OBSERVATION_DIM + 1] = 1;
    const w2 = [1, 0, 0, 1];
    const w3 = [1, -1];
    const weights = {
      schemaVersion: 1,
      tensors: {
        'encoder.0.weight': { shape: [2, OBSERVATION_DIM], values: w1 },
        'encoder.0.bias': { shape: [2], values: [0, 0] },
        'encoder.2.weight': { shape: [2, 2], values: w2 },
        'encoder.2.bias': { shape: [2], values: [0, 0] },
        'encoder.4.weight': { shape: [1, 2], values: w3 },
        'encoder.4.bias': { shape: [1], values: [0] },
      },
    };

    const network = createTdValueNetworkFromWeights(manifest, weights);
    const observation = new Array<number>(OBSERVATION_DIM).fill(0);
    observation[0] = 0.4;
    observation[1] = 0.1;
    const score = network.predict(observation);
    const expected =
      Math.tanh(Math.tanh(0.4)) - Math.tanh(Math.tanh(0.1));
    expect(score).toBeCloseTo(expected, 6);
  });
});
