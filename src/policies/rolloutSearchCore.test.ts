import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import { rankHeuristicV2Actions } from './heuristicScorerV2';
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
    expect(serialDiagnostics).toHaveLength(1);
    expect(serialDiagnostics[0].selectedActionKey).toBe(
      actionStableKey(serial!)
    );
    expectRootActionDiagnosticsAreConsistent(serialDiagnostics[0]);
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

  it('uses an injected root guide for expansion order and priors', () => {
    const fixture = selectionFixture('rollout-core-custom-root-guide');
    const keyed = toKeyedActions(fixture.candidateActions);
    const rankedRootActions = keyed
      .slice()
      .reverse()
      .map(({ action, actionKey }) => ({ action, actionKey }));
    const rootPriorByKey = new Map(
      rankedRootActions.map((candidate, index) => [
        candidate.actionKey,
        index === 0 ? 0.8 : 0.2 / (rankedRootActions.length - 1),
      ])
    );
    const diagnostics: SearchDecisionDiagnostics[] = [];

    selectRolloutSearchActionSync({
      ...fixture,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
      },
      random: rngFromSeed('rollout-core-custom-root-guide-rng'),
      randomSeed: 'rollout-core-custom-root-guide-rng',
      createRootGuide() {
        return {
          rankedRootActions,
          rootPriorByKey,
        };
      },
      onSearchDiagnostics(value) {
        diagnostics.push(value);
      },
    });

    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0].rootActions.map((entry) => entry.actionKey)).toEqual(
      rankedRootActions
        .slice(0, diagnostics[0].expandedRootActions)
        .map((entry) => entry.actionKey)
    );
    expect(diagnostics[0].rootActions[0].prior).toBe(0.8);
  });

  it('uses heuristic v2 for configured root expansion order', () => {
    const fixture = selectionFixture('rollout-core-heuristic-v2-root');
    const diagnostics: SearchDecisionDiagnostics[] = [];

    selectRolloutSearchActionSync({
      ...fixture,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 3,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
      random: rngFromSeed('rollout-core-heuristic-v2-root-rng'),
      randomSeed: 'rollout-core-heuristic-v2-root-rng',
      onSearchDiagnostics(value) {
        diagnostics.push(value);
      },
    });

    const expected = rankHeuristicV2Actions(fixture.candidateActions, {
      state: fixture.state,
      view: fixture.view,
    })
      .slice(0, diagnostics[0].expandedRootActions)
      .map((entry) => entry.actionKey);
    expect(diagnostics[0].rootActions.map((entry) => entry.actionKey)).toEqual(
      expected
    );
    expect(diagnostics[0].heuristic).toBe('v2');
  });

  it('runs rollout tasks with heuristic v2 playout selection configured', () => {
    const fixture = selectionFixture('rollout-core-heuristic-v2-task');
    const rootAction = toKeyedActions(fixture.candidateActions)[0];
    const result = runRolloutSearchTask({
      kind: 'rollout-search',
      visitIndex: 0,
      world: fixture.state,
      rootPlayer: fixture.view.activePlayerId,
      rootAction: rootAction.action,
      rootActionKey: rootAction.actionKey,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 4,
        maxRootActions: 3,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
      randomSeed: 'rollout-core-heuristic-v2-task-rng',
    });

    expect(result.actionKey).toBe(rootAction.actionKey);
    expect(result.simulatedActionSteps).toBeGreaterThan(0);
  });
});

function expectRootActionDiagnosticsAreConsistent(
  diagnostics: SearchDecisionDiagnostics
): void {
  expect(diagnostics.rootActions).toHaveLength(diagnostics.expandedRootActions);
  expect(
    diagnostics.rootActions.reduce((total, entry) => total + entry.visits, 0)
  ).toBe(diagnostics.rootVisitBudget);
  expect(
    diagnostics.rootActions.reduce(
      (total, entry) => total + entry.terminalRollouts,
      0
    )
  ).toBe(diagnostics.terminalRollouts);
  expect(diagnostics.terminalRate).toBe(
    diagnostics.terminalRollouts / diagnostics.rootVisitBudget
  );

  const selected = diagnostics.rootActions.find(
    (entry) => entry.actionKey === diagnostics.selectedActionKey
  );
  expect(selected).toBeDefined();
  expect(diagnostics.selectedActionVisits).toBe(selected!.visits);
  expect(diagnostics.selectedActionMeanValue).toBe(selected!.meanValue);
  expect(diagnostics.selectedActionTerminalRollouts).toBe(
    selected!.terminalRollouts
  );
  expect(diagnostics.selectedActionTerminalRate).toBe(selected!.terminalRate);

  for (const rootAction of diagnostics.rootActions) {
    expect(Number.isFinite(rootAction.meanValue)).toBe(true);
    expect(rootAction.terminalRate).toBe(
      rootAction.visits === 0
        ? 0
        : rootAction.terminalRollouts / rootAction.visits
    );
  }
}

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
