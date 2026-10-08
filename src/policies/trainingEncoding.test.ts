import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';

import { legalActions } from '../engine/actionBuilders';
import { legalActionsCanonical } from '../engine/actionSurface';
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
  it('preserves the pre-refactor seeded observation and canonical action encodings', () => {
    const state = createSession('value-refactor-compatibility', 'PlayerA');
    const encoded = {
      observation: encodeObservation(toPlayerView(state, 'PlayerA')),
      actions: encodeActionCandidates(
        legalActionsCanonical(state).map((entry) => entry.action)
      ),
    };
    // Captured before moving the value definitions; independent of shared constants.
    expect(
      createHash('sha256').update(JSON.stringify(encoded)).digest('hex')
    ).toBe('d6f7bac21fb9cb1ab0e701c538765aefc3eb267ce97065f20c939ebb930f16de');
  });

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
