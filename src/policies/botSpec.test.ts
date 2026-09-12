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

  it('parses a rollout-search spec with heuristic v2', () => {
    expect(
      parseBotSpec({
        id: 'search-v2',
        kind: 'search',
        config: {
          worlds: 2,
          rollouts: 1,
          depth: 4,
          maxRootActions: 3,
          rolloutEpsilon: 0,
          heuristic: 'v2',
        },
      })
    ).toEqual({
      id: 'search-v2',
      kind: 'search',
      config: {
        worlds: 2,
        rollouts: 1,
        depth: 4,
        maxRootActions: 3,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
    });
  });

  it('parses a complete TD-root rollout-search spec', () => {
    expect(
      parseBotSpec({
        id: 'td-root-search-small',
        kind: 'td-root-search',
        modelIndexPath: 'model-packs/index.json',
        config: {
          worlds: 2,
          rollouts: 1,
          depth: 4,
          maxRootActions: 3,
          rolloutEpsilon: 0,
        },
      })
    ).toEqual({
      id: 'td-root-search-small',
      kind: 'td-root-search',
      modelIndexPath: 'model-packs/index.json',
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
    expect(
      createPolicyFromBotSpec({
        id: 'td-root-test',
        kind: 'td-root-search',
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 1,
          maxRootActions: 1,
          rolloutEpsilon: 0,
        },
      }).selectAction
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

  it('rejects unsupported search heuristic versions', () => {
    expect(() =>
      parseBotSpec({
        id: 'broken-search',
        kind: 'search',
        config: {
          worlds: 2,
          rollouts: 1,
          depth: 4,
          maxRootActions: 3,
          rolloutEpsilon: 0,
          heuristic: 'v3',
        },
      })
    ).toThrow('heuristic');
  });
});
