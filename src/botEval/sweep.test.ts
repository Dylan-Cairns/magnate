import { describe, expect, it } from 'vitest';

import { resolveEvaluationExecution } from './execution';
import { runRolloutSearchSweep } from './sweep';
import type { HeadToHeadSummary, RolloutSearchSweepConfig } from './types';

describe('rollout-search sweeps', () => {
  it('runs candidates sequentially with one shared seed prefix', async () => {
    const capturedSeedPrefixes: string[] = [];
    const capturedWorkers: number[] = [];
    const progressEvents: string[] = [];
    const persistedCandidateIds: string[] = [];
    let activeCalls = 0;
    let maxActiveCalls = 0;

    const run = await runRolloutSearchSweep(sweepConfig(), {
      async runMatchup(config, dependencies) {
        capturedSeedPrefixes.push(config.seedPrefix);
        capturedWorkers.push(dependencies?.workers ?? 1);
        activeCalls += 1;
        maxActiveCalls = Math.max(maxActiveCalls, activeCalls);
        dependencies?.onProgress?.({
          type: 'pair-completed',
          candidateId: config.candidate.id,
          pairNumber: 1,
          workerId: 1,
          completedPairs: 1,
          totalPairs: 1,
          completedGames: 2,
          totalGames: 2,
          elapsedMs: 1,
          gamesPerMinute: 120_000,
          etaMs: 0,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        activeCalls -= 1;
        return {
          config,
          execution: resolveEvaluationExecution(
            dependencies?.workers ?? 1,
            config.gamesPerSide
          ),
          summary: {} as HeadToHeadSummary,
          games: [],
        };
      },
      workers: 4,
      onProgress(progress) {
        progressEvents.push(progress.type);
      },
      async onCandidateCompleted(completed) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        persistedCandidateIds.push(completed.matchup.config.candidate.id);
        expect(completed.run.matchups).toHaveLength(completed.candidateIndex);
      },
    });

    expect(run.matchups).toHaveLength(2);
    expect(capturedSeedPrefixes).toEqual([
      'shared-sweep-seed',
      'shared-sweep-seed',
    ]);
    expect(maxActiveCalls).toBe(1);
    expect(capturedWorkers).toEqual([4, 4]);
    expect(run.execution).toEqual({
      requestedWorkers: 4,
      workers: 1,
      parallelUnit: 'paired-seed',
      latencyMode: 'isolated',
    });
    expect(persistedCandidateIds).toEqual(['search-one', 'search-two']);
    expect(progressEvents).toEqual([
      'sweep-started',
      'candidate-started',
      'pair-completed',
      'candidate-completed',
      'candidate-started',
      'pair-completed',
      'candidate-completed',
      'sweep-completed',
    ]);
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
