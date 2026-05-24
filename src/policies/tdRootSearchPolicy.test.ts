import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import {
  createTdRootSearchPolicy,
  createTdRootSearchRootGuide,
} from './tdRootSearchPolicy';
import { ACTION_FEATURE_DIM, OBSERVATION_DIM } from './trainingEncoding';

describe('td root search policy', () => {
  it('uses TD value scores for root ranking and TD logits for priors', () => {
    const state = createSession('td-root-search-guide', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    const keyed = toKeyedActions(actions);
    const preferredIndex = 0;
    const preferredKey = keyed[preferredIndex].actionKey;
    const activeValueSignByAction = keyed.map((candidate) => {
      const next = stepToDecision(state, candidate.action);
      if (isTerminal(next)) {
        throw new Error(
          'TD root search guide test expected non-terminal roots.'
        );
      }
      return next.players[next.activePlayerIndex]?.id === view.activePlayerId
        ? 1
        : -1;
    });
    let valueCallCount = 0;

    const guide = createTdRootSearchRootGuide({
      state,
      view,
      candidateActions: actions,
      worldStates: [state],
      rootPlayer: view.activePlayerId,
      model: fakeModel({
        predict() {
          const rootValue = valueCallCount === preferredIndex ? 0.8 : -0.2;
          const value = rootValue * activeValueSignByAction[valueCallCount];
          valueCallCount += 1;
          return value;
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
