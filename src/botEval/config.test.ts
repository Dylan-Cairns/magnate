import { describe, expect, it } from 'vitest';

import {
  parseHeadToHeadConfig,
  parseRolloutSearchSweepConfig,
  parseTdReplayConfig,
} from './config';

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
        profileId: 'rollout-search-v2-easy',
      },
    });

    expect(config.candidate.id).toBe('small-search');
    expect(config.opponent).toEqual({
      id: 'rollout-search-v2-easy',
      kind: 'search',
      config: {
        worlds: 20,
        rollouts: 1,
        depth: 80,
        maxRootActions: 10,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
    });
  });

  it('rejects duplicate bot ids', () => {
    expect(() =>
      parseHeadToHeadConfig({
        schemaVersion: 1,
        runLabel: 'duplicate-test',
        seedPrefix: 'duplicate-test',
        gamesPerSide: 1,
        candidate: { profileId: 'rollout-search-v2-easy' },
        opponent: { profileId: 'rollout-search-v2-easy' },
      })
    ).toThrow('distinct');
  });

  it('parses rollout-search sweep configs', () => {
    const config = parseRolloutSearchSweepConfig(sweepConfig());

    expect(config.opponent.id).toBe('rollout-search-v2-easy');
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

  it('parses TD replay configs and resolves profile specs without using profile policies', () => {
    const config = parseTdReplayConfig({
      schemaVersion: 1,
        runLabel: 'td-replay-test',
        seedPrefix: 'td-replay-test',
        games: 2,
        playerA: { profileId: 'rollout-search-v2-medium' },
        playerB: { id: 'random-b', kind: 'random' },
      });

    expect(config.playerA).toEqual({
      id: 'rollout-search-v2-medium',
      kind: 'search',
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
    });
    expect(config.playerB).toEqual({ id: 'random-b', kind: 'random' });
  });

  it('rejects TD replay configs with td-root-search specs', () => {
    expect(() =>
      parseTdReplayConfig({
        schemaVersion: 1,
        runLabel: 'td-replay-test',
        seedPrefix: 'td-replay-test',
        games: 1,
        playerA: {
          id: 'td-root-inline',
          kind: 'td-root-search',
          config: {
            worlds: 1,
            rollouts: 1,
            depth: 1,
            maxRootActions: 1,
            rolloutEpsilon: 0,
          },
        },
        playerB: { id: 'random-b', kind: 'random' },
      })
    ).toThrow('td-root-search is not supported');
  });

  it('rejects TD replay configs without positive games', () => {
    expect(() =>
      parseTdReplayConfig({
        schemaVersion: 1,
        runLabel: 'td-replay-test',
        seedPrefix: 'td-replay-test',
        games: 0,
        playerA: { id: 'random-a', kind: 'random' },
        playerB: { id: 'random-b', kind: 'random' },
      })
    ).toThrow('positive integer');
  });
});

function sweepConfig() {
  return {
    schemaVersion: 1,
    runLabel: 'sweep-test',
    seedPrefix: 'sweep-test',
    gamesPerSide: 1,
    opponent: { profileId: 'rollout-search-v2-easy' },
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
