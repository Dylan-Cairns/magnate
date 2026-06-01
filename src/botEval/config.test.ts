import { describe, expect, it } from 'vitest';

import { parseHeadToHeadConfig, parseRolloutSearchSweepConfig } from './config';

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

  it('parses rollout-search sweep configs', () => {
    const config = parseRolloutSearchSweepConfig(sweepConfig());

    expect(config.opponent.id).toBe('heuristic');
    expect(config.candidates.map((candidate) => candidate.id)).toEqual([
      'search-one',
    ]);
  });

  it('rejects rollout-search sweeps without candidates', () => {
    expect(() =>
      parseRolloutSearchSweepConfig({
        ...sweepConfig(),
        candidates: [],
      })
    ).toThrow('at least one');
  });

  it('rejects non-search sweep candidates', () => {
    expect(() =>
      parseRolloutSearchSweepConfig({
        ...sweepConfig(),
        candidates: [{ id: 'heuristic-candidate', kind: 'heuristic' }],
      })
    ).toThrow('kind search');
  });

  it('rejects duplicate sweep bot ids', () => {
    expect(() =>
      parseRolloutSearchSweepConfig({
        ...sweepConfig(),
        candidates: [
          searchSpec('duplicate-search'),
          searchSpec('duplicate-search'),
        ],
      })
    ).toThrow('distinct');
  });
});

function sweepConfig() {
  return {
    schemaVersion: 1,
    runLabel: 'sweep-test',
    seedPrefix: 'sweep-test',
    gamesPerSide: 1,
    opponent: { profileId: 'heuristic' },
    candidates: [searchSpec('search-one')],
  };
}

function searchSpec(id: string) {
  return {
    id,
    kind: 'search',
    config: {
      worlds: 1,
      rollouts: 1,
      depth: 1,
      maxRootActions: 1,
      rolloutEpsilon: 0,
    },
  };
}
