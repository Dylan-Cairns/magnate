import { describe, expect, it } from 'vitest';

import { createPolicyFromBotSpec, parseBotSpec } from './botSpec';

describe('bot specs', () => {
  it('parses a complete rollout-search spec', () => {
    expect(
      parseBotSpec({
        id: 'search-small',
        kind: 'search',
        config: {
          worlds: 2,
          rollouts: 1,
          depth: 4,
          maxRootActions: 3,
          rolloutEpsilon: 0,
        },
      })
    ).toEqual({
      id: 'search-small',
      kind: 'search',
      config: {
        worlds: 2,
        rollouts: 1,
        depth: 4,
        maxRootActions: 3,
        rolloutEpsilon: 0,
      },
    });
  });

  it('constructs policies for deterministic bot kinds', () => {
    expect(
      createPolicyFromBotSpec({ id: 'heuristic-test', kind: 'heuristic' })
        .selectAction
    ).toBeTypeOf('function');
    expect(
      createPolicyFromBotSpec({ id: 'random-test', kind: 'random' })
        .selectAction
    ).toBeTypeOf('function');
  });

  it('rejects incomplete search configs', () => {
    expect(() =>
      parseBotSpec({
        id: 'broken-search',
        kind: 'search',
        config: {
          worlds: 2,
        },
      })
    ).toThrow('rollouts');
  });
});
