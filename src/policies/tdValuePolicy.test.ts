import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { TdValueScorer } from './tdValueModelPack';
import { createTdValuePolicy } from './tdValuePolicy';
import { OBSERVATION_DIM } from './trainingEncoding';

function constantScorer(): TdValueScorer {
  return {
    observationDim: OBSERVATION_DIM,
    predict() {
      return 0;
    },
  };
}

describe('td value policy', () => {
  it('selects a legal action and resolves score ties by action key', async () => {
    const state = createSession('td-value-policy-ties', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    expect(actions.length).toBeGreaterThan(1);

    let loadCount = 0;
    const policy = createTdValuePolicy({
      worlds: 4,
      loadModel: async () => {
        loadCount += 1;
        return constantScorer();
      },
    });
    const selected = await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-value-policy-rng'),
    });
    const expected = [...actions].sort((left, right) =>
      actionStableKey(left).localeCompare(actionStableKey(right))
    )[0];

    expect(selected).toBeDefined();
    expect(actionStableKey(selected!)).toBe(actionStableKey(expected));

    await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-value-policy-rng-2'),
    });
    expect(loadCount).toBe(1);
  });
});
