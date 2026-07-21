import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { createSession } from '../engine/session';
import {
  ACTION_FEATURE_DIM,
  ENCODING_VERSION,
  OBSERVATION_DIM,
  encodeActionCandidates,
} from './trainingEncoding';
import {
  TD_ROOT_MODEL_TYPE,
  TdRootOpponentNetwork,
  parseTdRootModelPackManifest,
  type TdRootModelPackManifest,
} from './tdRootModelPack';

function makeManifest(modelType = TD_ROOT_MODEL_TYPE): TdRootModelPackManifest {
  return {
    schemaVersion: 1,
    packId: 'td-root-test-pack',
    label: 'TD Root Test Pack',
    createdAtUtc: '2026-06-18T00:00:00Z',
    model: {
      modelType,
      weightsPath: 'weights.json',
      value: {
        checkpointType: 'magnate_td_value_v1',
        encodingVersion: ENCODING_VERSION,
        observationDim: OBSERVATION_DIM,
        hiddenDim: 2,
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
        hiddenDim: 2,
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
    source: {},
  };
}

describe('td root model pack', () => {
  it('parses the canonical TD-root model type', () => {
    const manifest = parseTdRootModelPackManifest(makeManifest());

    expect(manifest.model.modelType).toBe('td-root-search-v1');
    expect(manifest.model.value.observationDim).toBe(OBSERVATION_DIM);
    expect(manifest.model.opponent.actionFeatureDim).toBe(ACTION_FEATURE_DIM);
  });

  it('rejects legacy TD search model packs', () => {
    expect(() =>
      parseTdRootModelPackManifest(makeManifest('td-search-v1'))
    ).toThrow('td-search-v1');
  });

  it('runs opponent logits from exported tensors', () => {
    const model = makeTinyOpponentNetwork();

    const logits = model.logits(
      [0.5, 0.25],
      [
        [0.1, 0.2],
        [0.9, 0.2],
      ]
    );

    expect(logits).toHaveLength(2);
    expect(logits[0]).toBeCloseTo(expectedTinyOpponentLogit(0.5, 0.1), 6);
    expect(logits[1]).toBeCloseTo(expectedTinyOpponentLogit(0.5, 0.9), 6);
    expect(logits[1]).toBeGreaterThan(logits[0]);
  });

  it('keeps repeated opponent logits deterministic', () => {
    const model = makeTinyOpponentNetwork();
    const observation = [0.5, 0.25];
    const actions = [
      [0.1, 0.2],
      [0.9, 0.2],
    ];

    const first = model.logits(observation, actions);
    const second = model.logits(observation, actions);

    expect(Array.from(second)).toEqual(Array.from(first));
  });

  it('does not leak action embeddings between opponent candidates', () => {
    const model = makeTinyOpponentNetwork();
    const observation = [0.5, 0.25];
    const actions = [
      [0.1, 0.2],
      [0.9, 0.2],
      [0.1, 0.8],
    ];

    const batch = model.logits(observation, actions);
    const firstSolo = model.logits(observation, [actions[0]]);
    const secondSolo = model.logits(observation, [actions[1]]);
    const thirdSolo = model.logits(observation, [actions[2]]);

    expect(batch[0]).toBeCloseTo(firstSolo[0], 6);
    expect(batch[1]).toBeCloseTo(secondSolo[0], 6);
    expect(batch[2]).toBeCloseTo(thirdSolo[0], 6);
  });

  it('rejects opponent inputs with wrong dimensions', () => {
    const model = makeTinyOpponentNetwork();

    expect(() => model.logits([0.5], [[0.1, 0.2]])).toThrow(
      'observation length mismatch'
    );
    expect(() => model.logits([0.5, 0.25], [[0.1]])).toThrow(
      'action feature length mismatch'
    );
  });

  it('scores raw actions the same as pre-encoded action features', () => {
    const model = makeFullActionOpponentNetwork();
    const observation = new Array<number>(OBSERVATION_DIM).fill(0);
    observation[0] = 0.5;
    observation[1] = 0.25;
    const state = createSession('td-root-raw-action-logits', 'PlayerA');
    const actions = legalActions(state);

    const fromFeatures = model.logits(
      observation,
      encodeActionCandidates(actions)
    );
    const fromActions = model.logitsForActions(observation, actions);

    expect(fromActions).toHaveLength(fromFeatures.length);
    for (let index = 0; index < fromFeatures.length; index += 1) {
      expect(fromActions[index]).toBeCloseTo(fromFeatures[index], 6);
    }
  });

  it('keeps paired raw-action logits bit-identical to scalar scoring', () => {
    const model = makeFullActionOpponentNetwork();
    const observationA = new Array<number>(OBSERVATION_DIM).fill(0);
    observationA[0] = 0.5;
    observationA[1] = 0.25;
    const observationB = new Array<number>(OBSERVATION_DIM).fill(0);
    observationB[0] = -0.125;
    observationB[1] = 0.75;
    const stateA = createSession('td-root-pair-a', 'PlayerA');
    const stateB = createSession('td-root-pair-b', 'PlayerB');
    const actionsA = legalActions(stateA);
    const actionsB = legalActions(stateB).slice(0, 2);

    const scalarA = model.logitsForActions(observationA, actionsA);
    const scalarB = model.logitsForActions(observationB, actionsB);
    const [pairedA, pairedB] = model.logitsForActionPair(
      observationA,
      actionsA,
      observationB,
      actionsB
    );

    expect(Array.from(pairedA)).toEqual(Array.from(scalarA));
    expect(Array.from(pairedB)).toEqual(Array.from(scalarB));
  });

  it('falls back safely when one paired action list is empty', () => {
    const model = makeFullActionOpponentNetwork();
    const observationA = new Array<number>(OBSERVATION_DIM).fill(0.25);
    const observationB = new Array<number>(OBSERVATION_DIM).fill(-0.25);
    const state = createSession('td-root-pair-empty', 'PlayerA');
    const actions = legalActions(state);

    const scalar = model.logitsForActions(observationA, actions);
    const [pairedA, pairedB] = model.logitsForActionPair(
      observationA,
      actions,
      observationB,
      []
    );

    expect(Array.from(pairedA)).toEqual(Array.from(scalar));
    expect(pairedB).toHaveLength(0);
  });
});

function makeTinyOpponentNetwork(): TdRootOpponentNetwork {
  return new TdRootOpponentNetwork({
    observationDim: 2,
    actionFeatureDim: 2,
    hiddenDim: 2,
    obsW1: Float32Array.from([1, 0, 0, 1]),
    obsB1: Float32Array.from([0, 0]),
    obsW2: Float32Array.from([1, 0, 0, 1]),
    obsB2: Float32Array.from([0, 0]),
    actionW: Float32Array.from([1, 0, 0, 1]),
    actionB: Float32Array.from([0, 0]),
    headW1: Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    headB1: Float32Array.from([0, 0]),
    headW2: Float32Array.from([1, 1]),
    headB2: Float32Array.from([0]),
  });
}

function expectedTinyOpponentLogit(
  observation0: number,
  action0: number
): number {
  const observationEmbedding0 = Math.tanh(Math.tanh(observation0));
  const actionEmbedding0 = Math.tanh(action0);
  return Math.tanh(observationEmbedding0) + Math.tanh(actionEmbedding0);
}

function makeFullActionOpponentNetwork(): TdRootOpponentNetwork {
  const obsW1 = new Array<number>(OBSERVATION_DIM * 2).fill(0);
  obsW1[0] = 1;
  obsW1[OBSERVATION_DIM + 1] = 1;
  const actionW = new Array<number>(ACTION_FEATURE_DIM * 2).fill(0);
  actionW[0] = 1;
  actionW[ACTION_FEATURE_DIM + 1] = 1;
  return new TdRootOpponentNetwork({
    observationDim: OBSERVATION_DIM,
    actionFeatureDim: ACTION_FEATURE_DIM,
    hiddenDim: 2,
    obsW1: Float32Array.from(obsW1),
    obsB1: Float32Array.from([0, 0]),
    obsW2: Float32Array.from([1, 0, 0, 1]),
    obsB2: Float32Array.from([0, 0]),
    actionW: Float32Array.from(actionW),
    actionB: Float32Array.from([0, 0]),
    headW1: Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    headB1: Float32Array.from([0, 0]),
    headW2: Float32Array.from([1, 1]),
    headB2: Float32Array.from([0]),
  });
}
