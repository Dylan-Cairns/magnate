import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import {
  createTdRootSearchPolicy,
  createTdRootSearchRolloutGuidance,
  createTdRootSearchRootGuide,
} from './tdRootSearchPolicy';
import { ACTION_FEATURE_DIM, OBSERVATION_DIM } from './trainingEncoding';

describe('td root search policy', () => {
  it('uses TD logits for root ranking and priors', () => {
    const state = createSession('td-root-search-guide', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    const keyed = toKeyedActions(actions);
    const preferredIndex = Math.min(1, keyed.length - 1);
    const preferredKey = keyed[preferredIndex].actionKey;

    const guide = createTdRootSearchRootGuide({
      state,
      view,
      candidateActions: actions,
      worldStates: [state],
      rootPlayer: view.activePlayerId,
      model: fakeModel({
        predict() {
          throw new Error('Root guide should not call the value model.');
        },
        logits(_observation, actionFeatures) {
          return Float32Array.from(
            actionFeatures.map((_features, index) =>
              index === preferredIndex ? 3 : 0
            )
          );
        },
      }),
    });

    expect(guide.rankedRootActions[0].actionKey).toBe(preferredKey);
    expect(guide.rootPriorByKey.get(preferredKey)).toBeGreaterThan(
      Math.max(
        ...keyed
          .filter((candidate) => candidate.actionKey !== preferredKey)
          .map(
            (candidate) => guide.rootPriorByKey.get(candidate.actionKey) ?? 0
          )
      )
    );
  });

  it('selects a legal action and caches the model loader', async () => {
    const state = createSession('td-root-search-policy-cache', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    expect(actions.length).toBeGreaterThan(1);

    let loadCount = 0;
    const policy = createTdRootSearchPolicy({
      worlds: 1,
      rollouts: 1,
      depth: 2,
      maxRootActions: 3,
      rolloutEpsilon: 0,
      loadModel: async () => {
        loadCount += 1;
        return fakeModel({
          predict() {
            return 0;
          },
          logits(_observation, actionFeatures) {
            return new Float32Array(actionFeatures.length);
          },
        });
      },
    });

    const selected = await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-root-search-policy-rng'),
      randomSeed: 'td-root-search-policy-rng',
    });
    expect(selected).toBeDefined();
    const legalKeys = new Set(actions.map((action) => actionStableKey(action)));
    expect(legalKeys.has(actionStableKey(selected!))).toBe(true);

    await policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed('td-root-search-policy-rng-2'),
      randomSeed: 'td-root-search-policy-rng-2',
    });
    expect(loadCount).toBe(1);
  });

  it('uses TD rollout guidance for leaf values and playout action logits', () => {
    const state = createSession('td-root-search-rollout-guide', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    const keyed = toKeyedActions(actions);
    const preferred = keyed[keyed.length - 1];
    const guidance = createTdRootSearchRolloutGuidance({
      model: fakeModel({
        predict() {
          return 0.6;
        },
        logits(_observation, actionFeatures) {
          return Float32Array.from(
            actionFeatures.map((_features, index) =>
              index === keyed.length - 1 ? 5 : 0
            )
          );
        },
      }),
    });

    expect(
      guidance.evaluateLeaf?.({
        state,
        rootPlayer: view.activePlayerId,
      })
    ).toBe(0.6);
    expect(
      actionStableKey(
        guidance.chooseRolloutAction!({
          state,
          actions,
          decisionPlayer: view.activePlayerId,
          rootPlayer: view.activePlayerId,
          random: rngFromSeed('td-root-search-rollout-guide-rng'),
          config: {
            worlds: 1,
            rollouts: 1,
            depth: 1,
            maxRootActions: 1,
            rolloutEpsilon: 0,
          },
        })!
      )
    ).toBe(preferred.actionKey);
  });
});

function fakeModel({
  predict,
  logits,
}: {
  predict: (observation: readonly number[]) => number;
  logits: (
    observation: readonly number[],
    actionFeatures: readonly number[][]
  ) => Float32Array;
}): LoadedTdGuidanceModel {
  return {
    valueScorer: {
      observationDim: OBSERVATION_DIM,
      predict,
    },
    opponentScorer: {
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
      logits,
    },
  };
}
