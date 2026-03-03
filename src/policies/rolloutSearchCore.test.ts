import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import {
  runRolloutSearchTask,
  selectRolloutSearchActionParallel,
  selectRolloutSearchActionSync,
  type RolloutSearchVisitResult,
  type RolloutSearchWorkerTask,
} from './rolloutSearchCore';
import type { SearchPolicyConfig } from './searchConfig';
import type { SearchDecisionDiagnostics } from './types';

const TEST_CONFIG: SearchPolicyConfig = {
  worlds: 2,
  rollouts: 1,
  depth: 4,
  maxRootActions: 3,
  rolloutEpsilon: 0.05,
};

describe('rollout search core', () => {
  it('matches the seeded serial planner when parallel batch size is one', async () => {
    const fixture = selectionFixture('rollout-core-seeded-serial');
    const randomSeed = 'rollout-core-seeded-rng';
    const serialDiagnostics: SearchDecisionDiagnostics[] = [];
    const parallelDiagnostics: SearchDecisionDiagnostics[] = [];

    const serial = selectRolloutSearchActionSync({
      ...fixture,
      config: TEST_CONFIG,
      random: rngFromSeed(randomSeed),
      randomSeed,
      onSearchDiagnostics(value) {
        serialDiagnostics.push(value);
      },
    });
    const parallel = await selectRolloutSearchActionParallel({
      ...fixture,
      config: TEST_CONFIG,
      random: rngFromSeed(randomSeed),
      randomSeed,
      batchSize: 1,
      parallelWorkers: 1,
      runBatch: runTasks,
      onSearchDiagnostics(value) {
        parallelDiagnostics.push(value);
      },
    });

    expect(actionStableKey(parallel!)).toBe(actionStableKey(serial!));
    expect(parallelDiagnostics).toEqual(serialDiagnostics);
  });

  it('merges parallel batch results independently of response ordering', async () => {
    const fixture = selectionFixture('rollout-core-response-order');
    const randomSeed = 'rollout-core-response-order-rng';

    const ordered = await selectRolloutSearchActionParallel({
      ...fixture,
      config: TEST_CONFIG,
      random: rngFromSeed(randomSeed),
      randomSeed,
      batchSize: 4,
      parallelWorkers: 2,
      runBatch: runTasks,
    });
    const reversed = await selectRolloutSearchActionParallel({
      ...fixture,
      config: TEST_CONFIG,
      random: rngFromSeed(randomSeed),
      randomSeed,
      batchSize: 4,
      parallelWorkers: 2,
      async runBatch(tasks) {
        return (await runTasks(tasks)).slice().reverse();
      },
    });

    expect(actionStableKey(reversed!)).toBe(actionStableKey(ordered!));
  });

  it('rejects worker results that do not match scheduled visits', async () => {
    const fixture = selectionFixture('rollout-core-bad-result');

    await expect(
      selectRolloutSearchActionParallel({
        ...fixture,
        config: TEST_CONFIG,
        random: rngFromSeed('rollout-core-bad-result-rng'),
        randomSeed: 'rollout-core-bad-result-rng',
        batchSize: 2,
        parallelWorkers: 2,
        async runBatch(tasks) {
          const results = await runTasks(tasks);
          return [
            {
              ...results[0],
              actionKey: 'not-the-scheduled-action',
            },
            ...results.slice(1),
          ];
        },
      })
    ).rejects.toThrow('not-the-scheduled-action');
  });
});

async function runTasks(
  tasks: readonly RolloutSearchWorkerTask[]
): Promise<readonly RolloutSearchVisitResult[]> {
  return tasks.map((task) => runRolloutSearchTask(task));
}

function selectionFixture(seed: string) {
  const state = createSession(seed, 'PlayerB');
  const view = toPlayerView(state, 'PlayerB');
  const candidateActions = legalActions(state);
  if (candidateActions.length < 2) {
    throw new Error('rollout search core test requires multiple actions.');
  }
  return {
    state,
    view,
    candidateActions,
  };
}
