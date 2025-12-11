import { describe, expect, it } from 'vitest';

import { ACTION_FEATURE_DIM, ENCODING_VERSION, OBSERVATION_DIM } from './trainingEncoding';
import {
  TdSearchOpponentNetwork,
  parseTdSearchModelPackManifest,
} from './tdSearchModelPack';

function makeManifest(hiddenDim: number) {
  return {
    schemaVersion: 1,
    packId: 'td-search-test-pack',
    label: 'TD Search Test Pack',
    createdAtUtc: '2026-03-05T00:00:00Z',
    model: {
      modelType: 'td-search-v1',
      weightsPath: 'weights.json',
      value: {
        checkpointType: 'magnate_td_value_v1',
        encodingVersion: ENCODING_VERSION,
        observationDim: OBSERVATION_DIM,
        hiddenDim,
        requiredStateDictKeys: [
          'encoder.0.weight',
          'encoder.0.bias',
          'encoder.2.weight',
          'encoder.2.bias',
          'encoder.4.weight',
          'encoder.4.bias',
        ],
      },
      opponent: {
        checkpointType: 'magnate_td_opponent_v1',
        encodingVersion: ENCODING_VERSION,
        observationDim: OBSERVATION_DIM,
        actionFeatureDim: ACTION_FEATURE_DIM,
        hiddenDim,
        requiredStateDictKeys: [
          'obs_encoder.0.weight',
          'obs_encoder.0.bias',
          'obs_encoder.2.weight',
          'obs_encoder.2.bias',
          'action_encoder.0.weight',
          'action_encoder.0.bias',
          'policy_head.0.weight',
          'policy_head.0.bias',
          'policy_head.2.weight',
          'policy_head.2.bias',
        ],
      },
    },
    source: {
      runId: 'run-001',
      valueCheckpoint: 'artifacts/value.pt',
      opponentCheckpoint: 'artifacts/opponent.pt',
      checkpointMetadata: { step: 1 },
    },
  };
}

describe('td search model pack', () => {
  it('parses manifest with expected schema and dimensions', () => {
    const manifest = parseTdSearchModelPackManifest(makeManifest(2));
    expect(manifest.model.modelType).toBe('td-search-v1');
    expect(manifest.model.value.observationDim).toBe(OBSERVATION_DIM);
    expect(manifest.model.opponent.actionFeatureDim).toBe(ACTION_FEATURE_DIM);
  });

  it('computes deterministic opponent logits from configured tensors', () => {
    const hiddenDim = 2;
    const obsW1 = new Array<number>(hiddenDim * OBSERVATION_DIM).fill(0);
    const obsW2 = [1, 0, 0, 1];
    const actionW = new Array<number>(hiddenDim * ACTION_FEATURE_DIM).fill(0);
    actionW[0] = 1;
    actionW[ACTION_FEATURE_DIM + 1] = 1;
    const headW1 = new Array<number>(hiddenDim * hiddenDim * 3).fill(0);
    headW1[2] = 1;
    headW1[(hiddenDim * 3) + 3] = 1;
    const headW2 = [1, -1];

    const model = new TdSearchOpponentNetwork({
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
      hiddenDim,
      obsW1: new Float32Array(obsW1),
      obsB1: new Float32Array([0, 0]),
      obsW2: new Float32Array(obsW2),
      obsB2: new Float32Array([0, 0]),
      actionW: new Float32Array(actionW),
      actionB: new Float32Array([0, 0]),
      headW1: new Float32Array(headW1),
      headB1: new Float32Array([0, 0]),
      headW2: new Float32Array(headW2),
      headB2: new Float32Array([0]),
    });

    const observation = new Array<number>(OBSERVATION_DIM).fill(0);
    observation[0] = 0.3;
    observation[1] = 0.1;
    const actionA = new Array<number>(ACTION_FEATURE_DIM).fill(0);
    actionA[0] = 0.5;
    actionA[1] = 0.2;
    const actionB = new Array<number>(ACTION_FEATURE_DIM).fill(0);
    actionB[0] = 0.1;
    actionB[1] = 0.4;

    const logits = model.logits(observation, [actionA, actionB]);
    const expectedA = Math.tanh(Math.tanh(0.5)) - Math.tanh(Math.tanh(0.2));
    const expectedB = Math.tanh(Math.tanh(0.1)) - Math.tanh(Math.tanh(0.4));
    expect(logits[0]).toBeCloseTo(expectedA, 6);
    expect(logits[1]).toBeCloseTo(expectedB, 6);
  });
});
