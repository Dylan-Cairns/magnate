import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';

import {
  ACTION_FEATURE_DIM,
  encodeActionCandidates,
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
});
