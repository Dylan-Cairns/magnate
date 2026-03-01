import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import { createSearchPolicy } from './searchPolicy';

describe('search policy', () => {
  it('chooses deterministically for the same state and RNG seed', async () => {
    const state = createSession('search-policy-deterministic', 'PlayerB');
    const view = toPlayerView(state, 'PlayerB');
    const actions = legalActions(state);
    const policy = createSearchPolicy({
      worlds: 2,
      rollouts: 1,
      depth: 6,
      maxRootActions: 4,
      rolloutEpsilon: 0.1,
    });

    const first = await Promise.resolve(
      policy.selectAction({
        state,
        view,
        legalActions: actions,
        random: rngFromSeed('search-test-rng'),
      })
    );
    const second = await Promise.resolve(
      policy.selectAction({
        state,
        view,
        legalActions: actions,
        random: rngFromSeed('search-test-rng'),
      })
    );

    expect(first).toBeDefined();
    expect(second).toBeDefined();
    expect(actionStableKey(first!)).toBe(actionStableKey(second!));
  });

  it('always returns a legal action', async () => {
    const policy = createSearchPolicy({
      worlds: 2,
      rollouts: 1,
      depth: 6,
      maxRootActions: 4,
      rolloutEpsilon: 0.1,
    });
    const seeds = ['search-legal-1', 'search-legal-2', 'search-legal-3'];

    for (const seed of seeds) {
      const state = createSession(seed, 'PlayerB');
      const view = toPlayerView(state, 'PlayerB');
      const actions = legalActions(state);
      const legalKeys = new Set(actions.map((action) => actionStableKey(action)));
      const selected = await Promise.resolve(
        policy.selectAction({
          state,
          view,
          legalActions: actions,
          random: rngFromSeed(`search-rng-${seed}`),
        })
      );

      expect(selected).toBeDefined();
      expect(legalKeys.has(actionStableKey(selected!))).toBe(true);
    }
  });

  it('throws when determinization accounting is inconsistent', () => {
    const state = createSession('search-policy-invalid-view', 'PlayerB');
    const view = toPlayerView(state, 'PlayerB');
    const brokenView = {
      ...view,
      deck: {
        ...view.deck,
        drawCount: view.deck.drawCount + 1,
      },
    };
    const actions = legalActions(state);
    const policy = createSearchPolicy();

    expect(() =>
      policy.selectAction({
        state,
        view: brokenView,
        legalActions: actions,
        random: rngFromSeed('search-invalid-rng'),
      })
    ).toThrow('Search determinization mismatch');
  });
});
