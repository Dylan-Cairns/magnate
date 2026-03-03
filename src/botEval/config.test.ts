import { describe, expect, it } from 'vitest';

import { parseHeadToHeadConfig } from './config';

describe('head-to-head config parsing', () => {
  it('resolves catalog profile references and arbitrary specs', () => {
    const config = parseHeadToHeadConfig({
      schemaVersion: 1,
      runLabel: 'config-test',
      seedPrefix: 'config-test',
      gamesPerSide: 2,
      candidate: {
        id: 'small-search',
        kind: 'search',
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 2,
          maxRootActions: 2,
          rolloutEpsilon: 0,
        },
      },
      opponent: {
        profileId: 'heuristic',
      },
    });

    expect(config.candidate.id).toBe('small-search');
    expect(config.opponent).toEqual({
      id: 'heuristic',
      kind: 'heuristic',
    });
  });

  it('rejects duplicate bot ids', () => {
    expect(() =>
      parseHeadToHeadConfig({
        schemaVersion: 1,
        runLabel: 'duplicate-test',
        seedPrefix: 'duplicate-test',
        gamesPerSide: 1,
        candidate: { profileId: 'heuristic' },
        opponent: { profileId: 'heuristic' },
      })
    ).toThrow('distinct');
  });
});
