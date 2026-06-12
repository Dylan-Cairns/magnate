import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';

import {
  ACTION_FEATURE_DIM,
  encodeAction,
  encodeActionCandidates,
  encodeActionInto,
  encodeObservation,
  OBSERVATION_DIM,
} from './trainingEncoding';

describe('training encoding', () => {
  it('encodes active-player view with stable observation dimension', () => {
    const state = createSession('encoding-test-seed', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const observation = encodeObservation(view);
    expect(observation).toHaveLength(OBSERVATION_DIM);
  });

  it('encodes legal action candidates with stable action dimension', () => {
    const state = createSession('encoding-action-seed', 'PlayerA');
    const actions = legalActions(state);
    const encoded = encodeActionCandidates(actions);
    expect(encoded.length).toBe(actions.length);
    expect(encoded.length).toBeGreaterThan(0);
    for (const vector of encoded) {
      expect(vector).toHaveLength(ACTION_FEATURE_DIM);
    }
  });

  it('encodes actions equivalently into a reusable output vector', () => {
    const state = createSession('encoding-action-into-seed', 'PlayerA');
    const actions = legalActions(state);
    const output = new Float32Array(ACTION_FEATURE_DIM);

    for (const action of actions) {
      output.fill(0.5);
      encodeActionInto(action, output);
      const encoded = encodeAction(action);
      expect(output).toHaveLength(encoded.length);
      encoded.forEach((value, index) => {
        expect(output[index]).toBeCloseTo(value, 6);
      });
    }
  });
});
