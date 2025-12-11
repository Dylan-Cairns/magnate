import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import { createTdSearchPolicy } from './tdSearchPolicy';
import { ACTION_FEATURE_DIM, OBSERVATION_DIM } from './trainingEncoding';

describe('td search policy', () => {
  it('selects a legal action and caches model loader', async () => {
    const state = createSession('td-search-browser-policy', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    expect(actions.length).toBeGreaterThan(1);

    let loadCount = 0;
    const policy = createTdSearchPolicy({
      worlds: 1,
      rollouts: 1,
      depth: 2,
      maxRootActions: 3,
      rolloutEpsilon: 0.0,
      loadModel: async () => {
        loadCount += 1;
        return {
          manifest: {
            schemaVersion: 1,
            packId: 'td-search-test',
            label: 'td-search-test',
            createdAtUtc: '2026-03-05T00:00:00Z',
            model: {
              modelType: 'td-search-v1',
              weightsPath: 'weights.json',
              value: {
                checkpointType: 'magnate_td_value_v1',
                encodingVersion: 2,
                observationDim: OBSERVATION_DIM,
                hiddenDim: 8,
                requiredStateDictKeys: [],
              },
              opponent: {
                checkpointType: 'magnate_td_opponent_v1',
                encodingVersion: 2,
                observationDim: OBSERVATION_DIM,
                actionFeatureDim: ACTION_FEATURE_DIM,
                hiddenDim: 8,
                requiredStateDictKeys: [],
              },
            },
            source: {},
          },
          valueScorer: {
            observationDim: OBSERVATION_DIM,
            hiddenDim: 8,
            predict() {
              return 0;
            },
          },
          opponentScorer: {
            observationDim: OBSERVATION_DIM,
            actionFeatureDim: ACTION_FEATURE_DIM,
            logits(_observation, actionFeatures) {
              return new Float32Array(actionFeatures.length);
            },
          },
        };
      },
    });

    const selected = await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-search-rng'),
    });
    expect(selected).toBeDefined();
    const selectedKey = actionStableKey(selected!);
    const legalKeys = new Set(actions.map((action) => actionStableKey(action)));
    expect(legalKeys.has(selectedKey)).toBe(true);

    await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-search-rng-2'),
    });
    expect(loadCount).toBe(1);
  });
});
