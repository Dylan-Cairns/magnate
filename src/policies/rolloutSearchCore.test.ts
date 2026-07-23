import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import {
  createSession,
  stepKnownLegalActionToDecision,
  stepKnownLegalActionToDecisionForSimulation,
} from '../engine/session';
import { toPlayerView } from '../engine/view';
import { rankHeuristicV2Actions } from './heuristicScorerV2';
import {
  rolloutSearchScenarioSeeds,
  runRolloutSearchTask,
  runRolloutSearchTaskResumable,
  selectRolloutSearchActionParallel,
  selectRolloutSearchActionSync,
  type RolloutSearchTraceStep,
  type RolloutSearchVisitResult,
  type RolloutSearchWorkerContext,
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
  it('exposes the exact deterministic seed pair for a common scenario', () => {
    expect(rolloutSearchScenarioSeeds('shared-root', 12)).toEqual({
      engineSeed: 'shared-root:engine-scenario:12',
      rolloutRandomSeed: 'shared-root:rollout-scenario:12',
    });
    expect(() => rolloutSearchScenarioSeeds('shared-root', -1)).toThrow(
      'nonnegative safe integer'
    );
  });

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
    expect(serialDiagnostics[0].guidance).toBe('heuristic');
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
      async runBatch(tasks, context) {
        return (await runTasks(tasks, context)).slice().reverse();
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
        async runBatch(tasks, context) {
          const results = await runTasks(tasks, context);
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
    expect(diagnostics[0].guidance).toBe('custom');
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
    expect(diagnostics[0].guidance).toBe('heuristic');
  });

  it('runs rollout tasks with heuristic v2 playout selection configured', () => {
    const fixture = selectionFixture('rollout-core-heuristic-v2-task');
    const rootAction = toKeyedActions(fixture.candidateActions)[0];
    const context: RolloutSearchWorkerContext = {
      contextId: 'rollout-core-heuristic-v2-task-context',
      worldStates: [fixture.state],
    };
    const result = runRolloutSearchTask(
      {
        kind: 'rollout-search',
        contextId: context.contextId,
        visitIndex: 0,
        actionVisitIndex: 0,
        scenarioIndex: 0,
        worldIndex: 0,
        engineSeed: 'rollout-core-heuristic-v2-task-engine',
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
      },
      context.worldStates
    );

    expect(result.actionKey).toBe(rootAction.actionKey);
    expect(result.simulatedActionSteps).toBeGreaterThan(0);
  });

  it('uses injected guidance for non-terminal leaf evaluation', () => {
    const fixture = selectionFixture('rollout-core-guided-leaf');
    const rootAction = nonTerminalRootAction(fixture);
    const context: RolloutSearchWorkerContext = {
      contextId: 'rollout-core-guided-leaf-context',
      worldStates: [fixture.state],
    };
    const result = runRolloutSearchTask(
      {
        kind: 'rollout-search',
        contextId: context.contextId,
        visitIndex: 0,
        actionVisitIndex: 0,
        scenarioIndex: 0,
        worldIndex: 0,
        engineSeed: 'rollout-core-guided-leaf-engine',
        rootPlayer: fixture.view.activePlayerId,
        rootAction: rootAction.action,
        rootActionKey: rootAction.actionKey,
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 0,
          maxRootActions: 1,
          rolloutEpsilon: 0,
        },
        randomSeed: 'rollout-core-guided-leaf-rng',
      },
      context.worldStates,
      undefined,
      {
        evaluateLeaf() {
          return 0.75;
        },
      }
    );

    expect(result.score).toBe(0.75);
    expect(result.terminatedBeforeDepthLimit).toBe(false);
  });

  it('uses injected guidance for rollout playout action selection', () => {
    const fixture = selectionFixture('rollout-core-guided-playout');
    const rootAction = rootActionWithFollowup(fixture);
    const context: RolloutSearchWorkerContext = {
      contextId: 'rollout-core-guided-playout-context',
      worldStates: [fixture.state],
    };
    let playoutCalls = 0;
    const result = runRolloutSearchTask(
      {
        kind: 'rollout-search',
        contextId: context.contextId,
        visitIndex: 0,
        actionVisitIndex: 0,
        scenarioIndex: 0,
        worldIndex: 0,
        engineSeed: 'rollout-core-guided-playout-engine',
        rootPlayer: fixture.view.activePlayerId,
        rootAction: rootAction.action,
        rootActionKey: rootAction.actionKey,
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 1,
          maxRootActions: 1,
          rolloutEpsilon: 0,
        },
        randomSeed: 'rollout-core-guided-playout-rng',
      },
      context.worldStates,
      undefined,
      {
        chooseRolloutAction({ actions }) {
          playoutCalls += 1;
          return actions[actions.length - 1];
        },
        evaluateLeaf() {
          return 0.25;
        },
      }
    );

    expect(playoutCalls).toBeGreaterThan(0);
    expect(result.score).toBe(0.25);
    expect(result.simulatedActionSteps).toBeGreaterThanOrEqual(2);
  });

  it('simulation stepping preserves gameplay state while suppressing logs', () => {
    const fixture = selectionFixture('rollout-core-no-log-step');
    const rootAction = fixture.candidateActions.find((action) =>
      ['buy-deed', 'develop-outright', 'sell-card'].includes(action.type)
    );
    if (!rootAction) {
      throw new Error('Expected a card-play action in the fixture.');
    }

    const logged = stepKnownLegalActionToDecision(fixture.state, rootAction);
    const unlogged = stepKnownLegalActionToDecisionForSimulation(
      fixture.state,
      rootAction
    );

    expect(logged.log.length).toBeGreaterThan(fixture.state.log.length);
    expect(unlogged.log).toEqual(fixture.state.log);
    expect(withoutLog(unlogged)).toEqual(withoutLog(logged));
  });

  it('assigns common random scenarios by action-local visit index', async () => {
    const fixture = selectionFixture('rollout-core-common-scenarios');
    const keyed = toKeyedActions(fixture.candidateActions).slice(0, 2);
    const capturedTasks: RolloutSearchWorkerTask[] = [];

    await selectRolloutSearchActionParallel({
      ...fixture,
      candidateActions: keyed.map((entry) => entry.action),
      config: {
        worlds: 2,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
      },
      random: rngFromSeed('rollout-core-common-scenarios-rng'),
      randomSeed: 'rollout-core-common-scenarios-rng',
      batchSize: 2,
      parallelWorkers: 1,
      createRootGuide() {
        return {
          rankedRootActions: keyed.map(({ action, actionKey }) => ({
            action,
            actionKey,
          })),
          rootPriorByKey: new Map(
            keyed.map(({ actionKey }) => [actionKey, 0.5])
          ),
        };
      },
      async runBatch(tasks, context) {
        capturedTasks.push(...tasks);
        return runTasks(tasks, context);
      },
    });

    const byAction = new Map<string, RolloutSearchWorkerTask[]>();
    for (const task of capturedTasks) {
      byAction.set(task.rootActionKey, [
        ...(byAction.get(task.rootActionKey) ?? []),
        task,
      ]);
    }
    const firstActionTasks = byAction.get(keyed[0].actionKey) ?? [];
    const secondActionTasks = byAction.get(keyed[1].actionKey) ?? [];

    expect(firstActionTasks[0]).toMatchObject({
      actionVisitIndex: 0,
      scenarioIndex: 0,
      worldIndex: 0,
    });
    expect(secondActionTasks[0]).toMatchObject({
      actionVisitIndex: 0,
      scenarioIndex: 0,
      worldIndex: 0,
    });
    expect(firstActionTasks[0].engineSeed).toBe(
      secondActionTasks[0].engineSeed
    );
    expect(firstActionTasks[0].randomSeed).toBe(
      secondActionTasks[0].randomSeed
    );

    const laterScenario = capturedTasks.find(
      (task) => task.scenarioIndex === 1
    );
    expect(laterScenario).toMatchObject({
      actionVisitIndex: 1,
      worldIndex: 1,
    });
  });

  it('runs simulated engine seeds without mutating shared world states', () => {
    const fixture = selectionFixture('rollout-core-simulated-engine-seed');
    const cardPlayAction = fixture.candidateActions.find((action) =>
      ['buy-deed', 'develop-outright', 'sell-card'].includes(action.type)
    );
    if (!cardPlayAction) {
      throw new Error('Expected a card-play action in the fixture.');
    }
    const firstActionState = stepKnownLegalActionToDecision(
      fixture.state,
      cardPlayAction
    );
    const actionsAfterCardPlay = legalActions(firstActionState);
    const endTurn = actionsAfterCardPlay.find(
      (action) => action.type === 'end-turn'
    );
    if (!endTurn) {
      throw new Error('Expected an end-turn action after first card play.');
    }
    const context: RolloutSearchWorkerContext = {
      contextId: 'rollout-core-simulated-engine-seed-context',
      worldStates: [firstActionState],
    };
    const originalSeed = firstActionState.seed;
    const originalRngCursor = firstActionState.rngCursor;

    runRolloutSearchTask(
      {
        kind: 'rollout-search',
        contextId: context.contextId,
        visitIndex: 0,
        actionVisitIndex: 0,
        scenarioIndex: 0,
        worldIndex: 0,
        engineSeed: 'rollout-core-simulated-engine-seed-engine',
        rootPlayer: fixture.view.activePlayerId,
        rootAction: endTurn,
        rootActionKey: actionStableKey(endTurn),
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 1,
          maxRootActions: 1,
          rolloutEpsilon: 0,
        },
        randomSeed: 'rollout-core-simulated-engine-seed-playout',
      },
      context.worldStates
    );

    expect(firstActionState.seed).toBe(originalSeed);
    expect(firstActionState.rngCursor).toBe(originalRngCursor);
  });

  it('keeps rollout task results independent of source world logs', () => {
    const fixture = selectionFixture('rollout-core-log-independent-task');
    const rootAction = toKeyedActions(fixture.candidateActions)[0];
    const baseTask: RolloutSearchWorkerTask = {
      kind: 'rollout-search',
      contextId: 'rollout-core-log-independent-task-context',
      visitIndex: 0,
      actionVisitIndex: 0,
      scenarioIndex: 0,
      worldIndex: 0,
      engineSeed: 'rollout-core-log-independent-task-engine',
      rootPlayer: fixture.view.activePlayerId,
      rootAction: rootAction.action,
      rootActionKey: rootAction.actionKey,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 4,
        maxRootActions: 1,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
      randomSeed: 'rollout-core-log-independent-task-playout',
    };
    const noisyWorld = {
      ...fixture.state,
      log: [
        ...fixture.state.log,
        {
          turn: fixture.state.turn,
          phase: fixture.state.phase,
          player: fixture.view.activePlayerId,
          summary: 'test-only source log entry',
        },
      ],
    };

    const cleanResult = runRolloutSearchTask(baseTask, [fixture.state]);
    const noisyResult = runRolloutSearchTask(baseTask, [noisyWorld]);

    expect(noisyResult).toEqual(cleanResult);
  });

  it('traces a rollout without changing its result or shared source world', () => {
    const fixture = selectionFixture('rollout-core-trace');
    const rootAction = toKeyedActions(fixture.candidateActions)[0];
    const sourceWorld = structuredClone(fixture.state);
    const task: RolloutSearchWorkerTask = {
      kind: 'rollout-search',
      contextId: 'rollout-core-trace-context',
      visitIndex: 0,
      actionVisitIndex: 0,
      scenarioIndex: 0,
      worldIndex: 0,
      engineSeed: 'rollout-core-trace-engine',
      rootPlayer: fixture.view.activePlayerId,
      rootAction: rootAction.action,
      rootActionKey: rootAction.actionKey,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 4,
        maxRootActions: 1,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
      randomSeed: 'rollout-core-trace-playout',
    };
    const clean = runRolloutSearchTask(task, [fixture.state]);
    const steps: RolloutSearchTraceStep[] = [];
    const traced = runRolloutSearchTask(
      task,
      [fixture.state],
      undefined,
      undefined,
      {
        onStep(step) {
          steps.push(structuredClone(step));
          // Detached snapshots keep diagnostic consumers behavior-neutral.
          step.stateAfter.seed = 'trace-only-mutation';
        },
      }
    );

    expect(traced).toEqual(clean);
    expect(fixture.state).toEqual(sourceWorld);
    expect(steps).toHaveLength(traced.simulatedActionSteps);
    expect(steps[0]).toMatchObject({
      stepIndex: 0,
      decisionPlayer: fixture.view.activePlayerId,
      actionKey: rootAction.actionKey,
    });
    for (const [index, step] of steps.entries()) {
      expect(step.stepIndex).toBe(index);
      expect(step.legalActionKeys).toContain(step.actionKey);
      if (index > 0) {
        expect(step.stateBefore).toEqual(steps[index - 1].stateAfter);
      }
    }
  });

  it('keeps resumable scalar results and traces identical to the legacy task oracle', () => {
    const fixture = selectionFixture('rollout-core-resumable-parity');
    const rootAction = rootActionWithFollowup(fixture);
    const task: RolloutSearchWorkerTask = {
      kind: 'rollout-search',
      contextId: 'rollout-core-resumable-parity-context',
      visitIndex: 7,
      actionVisitIndex: 3,
      scenarioIndex: 3,
      worldIndex: 0,
      engineSeed: 'rollout-core-resumable-parity-engine',
      rootPlayer: fixture.view.activePlayerId,
      rootAction: rootAction.action,
      rootActionKey: rootAction.actionKey,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 7,
        maxRootActions: 1,
        rolloutEpsilon: 0.25,
        heuristic: 'v2',
      },
      randomSeed: 'rollout-core-resumable-parity-playout',
    };
    const guidance = {
      chooseRolloutAction({ actions }: { actions: readonly unknown[] }) {
        return actions[actions.length - 1] as (typeof task)['rootAction'];
      },
      evaluateLeaf() {
        return 0.321;
      },
    };
    const legacyTrace: RolloutSearchTraceStep[] = [];
    const resumableTrace: RolloutSearchTraceStep[] = [];

    const legacy = runRolloutSearchTask(
      task,
      [fixture.state],
      undefined,
      guidance,
      { onStep: (step) => legacyTrace.push(step) }
    );
    const resumable = runRolloutSearchTaskResumable(
      task,
      [fixture.state],
      undefined,
      guidance,
      { onStep: (step) => resumableTrace.push(step) }
    );

    expect(resumable).toEqual(legacy);
    expect(resumableTrace).toEqual(legacyTrace);
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
  tasks: readonly RolloutSearchWorkerTask[],
  context: RolloutSearchWorkerContext
): Promise<readonly RolloutSearchVisitResult[]> {
  return tasks.map((task) => runRolloutSearchTask(task, context.worldStates));
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

function nonTerminalRootAction(fixture: ReturnType<typeof selectionFixture>) {
  const keyed = toKeyedActions(fixture.candidateActions);
  const match = keyed.find(
    (candidate) =>
      !isTerminal(
        stepKnownLegalActionToDecisionForSimulation(
          fixture.state,
          candidate.action
        )
      )
  );
  if (!match) {
    throw new Error('rollout search core test requires a non-terminal action.');
  }
  return match;
}

function rootActionWithFollowup(fixture: ReturnType<typeof selectionFixture>) {
  const keyed = toKeyedActions(fixture.candidateActions);
  const match = keyed.find((candidate) => {
    const next = stepKnownLegalActionToDecisionForSimulation(
      fixture.state,
      candidate.action
    );
    return !isTerminal(next) && legalActions(next).length > 0;
  });
  if (!match) {
    throw new Error(
      'rollout search core test requires a root action with a follow-up decision.'
    );
  }
  return match;
}

function withoutLog<T extends { log: unknown }>(state: T): Omit<T, 'log'> {
  const rest = { ...state };
  delete (rest as { log?: unknown }).log;
  return rest;
}
