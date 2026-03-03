import { describe, expect, it } from 'vitest';

import { makeGameState } from '../engine/__tests__/fixtures';
import { rngFromSeed } from '../engine/rng';
import { policyRandomForState, policyRandomSeedForState } from './policyRandom';

describe('policy random helpers', () => {
  it('derives the same RNG stream from the exported state seed helper', () => {
    const state = makeGameState({ seed: 'policy-random-seed-test', turn: 4 });
    const random = policyRandomForState(state, 'rollout-eval-search');
    const fromSeed = rngFromSeed(
      policyRandomSeedForState(state, 'rollout-eval-search')
    );

    expect([random(), random(), random()]).toEqual([
      fromSeed(),
      fromSeed(),
      fromSeed(),
    ]);
  });
});
