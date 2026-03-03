import { describe, expect, it } from 'vitest';

import { runRolloutSearchSweep } from './sweep';
import type { HeadToHeadSummary, RolloutSearchSweepConfig } from './types';

describe('rollout-search sweeps', () => {
  it('runs candidates sequentially with one shared seed prefix', async () => {
    const capturedSeedPrefixes: string[] = [];
    let activeCalls = 0;
    let maxActiveCalls = 0;

    const run = await runRolloutSearchSweep(sweepConfig(), {
      async runMatchup(config) {
        capturedSeedPrefixes.push(config.seedPrefix);
        activeCalls += 1;
        maxActiveCalls = Math.max(maxActiveCalls, activeCalls);
        await new Promise((resolve) => setTimeout(resolve, 0));
        activeCalls -= 1;
        return {
          config,
          summary: {} as HeadToHeadSummary,
          games: [],
        };
      },
    });

    expect(run.matchups).toHaveLength(2);
    expect(capturedSeedPrefixes).toEqual([
      'shared-sweep-seed',
      'shared-sweep-seed',
    ]);
    expect(maxActiveCalls).toBe(1);
  });
});

function sweepConfig(): RolloutSearchSweepConfig {
  return {
    schemaVersion: 1,
    runLabel: 'sequential-sweep',
    seedPrefix: 'shared-sweep-seed',
    gamesPerSide: 1,
    opponent: {
      id: 'random-opponent',
      kind: 'random',
    },
    candidates: [searchSpec('search-one'), searchSpec('search-two')],
  };
}

function searchSpec(
  id: string
): RolloutSearchSweepConfig['candidates'][number] {
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
