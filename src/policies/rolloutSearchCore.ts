import { legalActions as legalActionsForState } from '../engine/actionBuilders';
import { toKeyedActions, type KeyedAction } from '../engine/actionSurface';
import { rngFromSeed, type RandomFn } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { stepKnownLegalActionToDecision } from '../engine/session';
import type {
  GameAction,
  GameState,
  PlayerId,
  PlayerView,
} from '../engine/types';
import { sampleHiddenWorldStates } from './determinization';
import {
  bestHeuristicAction,
  heuristicPriorsByKey,
  rankHeuristicActions,
} from './heuristicScorer';
import {
  bestHeuristicV2Action,
  heuristicV2PriorsByKey,
  rankHeuristicV2Actions,
} from './heuristicScorerV2';
import type {
  SearchHeuristicVersion,
  SearchPolicyConfig,
} from './searchConfig';
import {
  createSearchDecisionDiagnostics,
  progressiveTargetActionCount,
  safeDiv,
  selectBestRootActionKey,
  selectRootUcbAction,
} from './searchRoot';
import {
  evaluateSearchLeafState,
  evaluateSearchTerminalState,
} from './searchStateEvaluator';
import type {
  SearchDecisionDiagnostics,
  SearchRootActionDiagnostics,
} from './types';

export interface RolloutSearchSelectionInput {
  state: GameState;
  view: PlayerView;
  candidateActions: readonly GameAction[];
  config: SearchPolicyConfig;
  random: RandomFn;
  randomSeed?: string;
  createRootGuide?: RolloutSearchRootGuideFactory;
  onSearchDiagnostics?: (diagnostics: SearchDecisionDiagnostics) => void;
  onProgress?: () => void;
}

export interface RolloutSearchParallelSelectionInput extends RolloutSearchSelectionInput {
  batchSize: number;
  parallelWorkers: number;
  runBatch: (
    tasks: readonly RolloutSearchWorkerTask[],
    context: RolloutSearchWorkerContext
  ) => Promise<readonly RolloutSearchVisitResult[]>;
}

export interface RolloutSearchWorkerContext {
  contextId: string;
  worldStates: readonly GameState[];
}

export interface RolloutSearchWorkerTask {
  kind: 'rollout-search';
  contextId: string;
  visitIndex: number;
  worldIndex: number;
  rootPlayer: PlayerId;
  rootAction: GameAction;
  rootActionKey: string;
  config: SearchPolicyConfig;
  randomSeed?: string;
}

export interface RolloutSearchVisitResult {
  kind: 'rollout-search';
  visitIndex: number;
  actionKey: string;
  score: number;
  simulatedActionSteps: number;
  terminatedBeforeDepthLimit: boolean;
}

export interface RolloutSearchRankedRootAction {
  action: GameAction;
  actionKey: string;
}

export interface RolloutSearchRootGuide {
  rankedRootActions: readonly RolloutSearchRankedRootAction[];
  rootPriorByKey: ReadonlyMap<string, number>;
}

export interface RolloutSearchRootGuideInput {
  state: GameState;
  view: PlayerView;
  candidateActions: readonly GameAction[];
  worldStates: readonly GameState[];
  rootPlayer: PlayerId;
}

export type RolloutSearchRootGuideFactory = (
  input: RolloutSearchRootGuideInput
) => RolloutSearchRootGuide;

interface ScheduledVisit {
  actionKey: string;
}

interface RolloutSearchFinalResult {
  action: GameAction;
  actionKey: string;
  diagnostics: SearchDecisionDiagnostics;
}

export function selectRolloutSearchActionSync(
  input: RolloutSearchSelectionInput
): GameAction | undefined {
  if (input.candidateActions.length === 0) {
    return undefined;
  }
  if (input.candidateActions.length === 1) {
    return input.candidateActions[0];
  }

  const session = createRolloutSearchSession(input);
  if (!session) {
    return rankHeuristicActions(input.candidateActions, {
      state: input.state,
      view: input.view,
    })[0].action;
  }

  const workerContext = session.workerContext(
    input.randomSeed
      ? rolloutSearchWorkerContextId(input.randomSeed)
      : 'rollout-search-sync'
  );
  while (session.hasUnscheduledVisits()) {
    input.onProgress?.();
    const visitIndex = session.nextVisitIndex();
    const task = session.nextTask(
      workerContext.contextId,
      randomSeedForVisit(input.randomSeed, visitIndex)
    );
    const result = runRolloutSearchTask(
      task,
      workerContext.worldStates,
      task.randomSeed ? undefined : input.random
    );
    session.mergeResult(result);
  }

  const finalResult = session.finish({
    parallelWorkers: input.randomSeed ? 1 : undefined,
    parallelBatches: input.randomSeed ? session.mergedVisitCount() : undefined,
    parallelBatchSize: input.randomSeed ? 1 : undefined,
  });
  input.onSearchDiagnostics?.(finalResult.diagnostics);
  return finalResult.action;
}

export async function selectRolloutSearchActionParallel(
  input: RolloutSearchParallelSelectionInput
): Promise<GameAction | undefined> {
  if (input.candidateActions.length === 0) {
    return undefined;
  }
  if (input.candidateActions.length === 1) {
    return input.candidateActions[0];
  }
  if (input.randomSeed === undefined) {
    throw new Error('Parallel rollout search requires a randomSeed.');
  }
  if (!Number.isInteger(input.batchSize) || input.batchSize <= 0) {
    throw new Error('Parallel rollout search batchSize must be positive.');
  }
  if (!Number.isInteger(input.parallelWorkers) || input.parallelWorkers <= 0) {
    throw new Error(
      'Parallel rollout search parallelWorkers must be positive.'
    );
  }

  const session = createRolloutSearchSession(input);
  if (!session) {
    return rankHeuristicActions(input.candidateActions, {
      state: input.state,
      view: input.view,
    })[0].action;
  }

  const workerContext = session.workerContext(
    rolloutSearchWorkerContextId(input.randomSeed)
  );
  let batches = 0;
  while (session.hasUnscheduledVisits()) {
    const tasks: RolloutSearchWorkerTask[] = [];
    while (tasks.length < input.batchSize && session.hasUnscheduledVisits()) {
      input.onProgress?.();
      const visitIndex = session.nextVisitIndex();
      tasks.push(
        session.nextTask(
          workerContext.contextId,
          randomSeedForVisit(input.randomSeed, visitIndex)
        )
      );
    }
    batches += 1;
    const results = await input.runBatch(tasks, workerContext);
    if (results.length !== tasks.length) {
      throw new Error(
        `Parallel rollout search expected ${String(tasks.length)} results but received ${String(results.length)}.`
      );
    }
    for (const result of [...results].sort(
      (left, right) => left.visitIndex - right.visitIndex
    )) {
      session.mergeResult(result);
    }
  }

  const finalResult = session.finish({
    parallelWorkers: input.parallelWorkers,
    parallelBatches: batches,
    parallelBatchSize: input.batchSize,
  });
  input.onSearchDiagnostics?.(finalResult.diagnostics);
  return finalResult.action;
}

export function runRolloutSearchTask(
  task: RolloutSearchWorkerTask,
  worldStates: readonly GameState[],
  fallbackRandom?: RandomFn
): RolloutSearchVisitResult {
  const random = task.randomSeed
    ? rngFromSeed(task.randomSeed)
    : fallbackRandom;
  if (!random) {
    throw new Error('Rollout search task requires randomSeed or fallback RNG.');
  }
  const world = worldStates[task.worldIndex];
  if (!world) {
    throw new Error(
      `Rollout search task references unknown world index ${String(task.worldIndex)}.`
    );
  }

  const rollout = runRollout(
    world,
    task.rootPlayer,
    task.rootAction,
    task.config,
    random
  );
  return {
    kind: 'rollout-search',
    visitIndex: task.visitIndex,
    actionKey: task.rootActionKey,
    score: rollout.score,
    simulatedActionSteps: rollout.simulatedActionSteps,
    terminatedBeforeDepthLimit: rollout.terminatedBeforeDepthLimit,
  };
}

export function rolloutSearchRootBudget(
  config: SearchPolicyConfig,
  sampledWorldCount: number
): number {
  const visitCount = Math.max(1, sampledWorldCount * config.rollouts);
  return visitCount * Math.max(1, config.maxRootActions);
}

function createRolloutSearchSession({
  state,
  view,
  candidateActions,
  config,
  random,
  createRootGuide,
}: RolloutSearchSelectionInput): RolloutSearchSession | null {
  const rootPlayer = view.activePlayerId;
  const worldStates = sampleHiddenWorldStates({
    state,
    view,
    rootPlayer,
    worldCount: config.worlds,
    random,
    errorPrefix: 'Search',
  });
  if (worldStates.length === 0) {
    return null;
  }
  const rootGuide =
    createRootGuide?.({
      state,
      view,
      candidateActions,
      worldStates,
      rootPlayer,
    }) ??
    createHeuristicRolloutSearchRootGuide({
      state,
      view,
      candidateActions,
      heuristic: config.heuristic,
    });

  return new RolloutSearchSession({
    candidateActions,
    rankedRootActions: rootGuide.rankedRootActions,
    rootPriorByKey: rootGuide.rootPriorByKey,
    worldStates,
    rootPlayer,
    config,
  });
}

export function createHeuristicRolloutSearchRootGuide({
  state,
  view,
  candidateActions,
  heuristic = 'v1',
}: Pick<
  RolloutSearchSelectionInput,
  'state' | 'view' | 'candidateActions'
> & {
  heuristic?: SearchHeuristicVersion;
}): RolloutSearchRootGuide {
  const heuristicContext = { state, view };
  return {
    rankedRootActions: rankActionsByHeuristic(
      candidateActions,
      heuristicContext,
      heuristic
    ),
    rootPriorByKey: priorsByHeuristic(
      candidateActions,
      heuristicContext,
      heuristic
    ),
  };
}

class RolloutSearchSession {
  private readonly actionByKey: ReadonlyMap<string, GameAction>;
  private readonly rankedRootActions: readonly RolloutSearchRankedRootAction[];
  private readonly rootPriorByKey: ReadonlyMap<string, number>;
  private readonly worldStates: readonly GameState[];
  private readonly rootPlayer: PlayerId;
  private readonly config: SearchPolicyConfig;
  private readonly rootBudget: number;
  private readonly rootVisits = new Map<string, number>();
  private readonly rootValueSum = new Map<string, number>();
  private readonly rootTerminalRollouts = new Map<string, number>();
  private readonly pendingVisits = new Map<string, number>();
  private readonly scheduledVisits = new Map<number, ScheduledVisit>();
  private readonly mergedVisits = new Set<number>();
  private readonly expandedKeys: string[];
  private readonly pendingUnvisited: string[];
  private scheduledVisitCount = 0;
  private mergedVisitTotal = 0;
  private simulatedActionSteps = 0;
  private terminalRollouts = 0;

  constructor({
    candidateActions,
    rankedRootActions,
    rootPriorByKey,
    worldStates,
    rootPlayer,
    config,
  }: {
    candidateActions: readonly GameAction[];
    rankedRootActions: readonly RolloutSearchRankedRootAction[];
    rootPriorByKey: ReadonlyMap<string, number>;
    worldStates: readonly GameState[];
    rootPlayer: PlayerId;
    config: SearchPolicyConfig;
  }) {
    this.actionByKey = new Map(
      rankedRootActions.map((candidate) => [
        candidate.actionKey,
        candidate.action,
      ])
    );
    this.rankedRootActions = rankedRootActions;
    this.rootPriorByKey = rootPriorByKey;
    this.worldStates = worldStates;
    this.rootPlayer = rootPlayer;
    this.config = config;
    this.rootBudget = rolloutSearchRootBudget(config, worldStates.length);

    for (const candidate of rankedRootActions) {
      this.rootVisits.set(candidate.actionKey, 0);
      this.rootValueSum.set(candidate.actionKey, 0);
      this.rootTerminalRollouts.set(candidate.actionKey, 0);
      this.pendingVisits.set(candidate.actionKey, 0);
    }

    const expandedCount = Math.min(
      rankedRootActions.length,
      config.maxRootActions
    );
    this.expandedKeys = rankedRootActions
      .slice(0, expandedCount)
      .map((candidate) => candidate.actionKey);
    this.pendingUnvisited = [...this.expandedKeys];

    if (candidateActions.length !== rankedRootActions.length) {
      throw new Error(
        'Rollout search root ranking did not preserve action count.'
      );
    }
  }

  hasUnscheduledVisits(): boolean {
    return this.scheduledVisitCount < this.rootBudget;
  }

  nextVisitIndex(): number {
    return this.scheduledVisitCount;
  }

  mergedVisitCount(): number {
    return this.mergedVisitTotal;
  }

  workerContext(contextId: string): RolloutSearchWorkerContext {
    return {
      contextId,
      worldStates: this.worldStates,
    };
  }

  nextTask(contextId: string, randomSeed?: string): RolloutSearchWorkerTask {
    if (!this.hasUnscheduledVisits()) {
      throw new Error('Rollout search has no unscheduled visits.');
    }

    const visitIndex = this.scheduledVisitCount;
    const targetCount = progressiveTargetActionCount(
      this.rankedRootActions.length,
      this.config.maxRootActions,
      visitIndex
    );
    while (this.expandedKeys.length < targetCount) {
      const nextKey =
        this.rankedRootActions[this.expandedKeys.length].actionKey;
      this.expandedKeys.push(nextKey);
      this.pendingUnvisited.push(nextKey);
    }

    const actionKey =
      this.pendingUnvisited.length > 0
        ? this.pendingUnvisited.shift()!
        : selectRootUcbAction(
            this.expandedKeys,
            this.rootVisits,
            this.rootValueSum,
            this.rootPriorByKey,
            visitIndex,
            1,
            this.pendingVisits
          );
    const worldIndex = visitIndex % this.worldStates.length;

    this.scheduledVisitCount += 1;
    this.scheduledVisits.set(visitIndex, { actionKey });
    this.pendingVisits.set(
      actionKey,
      (this.pendingVisits.get(actionKey) ?? 0) + 1
    );
    const rootAction = this.actionByKey.get(actionKey);
    if (!rootAction) {
      throw new Error(
        `Rollout search scheduled unknown root action key ${actionKey}.`
      );
    }

    return {
      kind: 'rollout-search',
      contextId,
      visitIndex,
      worldIndex,
      rootPlayer: this.rootPlayer,
      rootAction,
      rootActionKey: actionKey,
      config: this.config,
      ...(randomSeed ? { randomSeed } : {}),
    };
  }

  mergeResult(result: RolloutSearchVisitResult): void {
    if (result.kind !== 'rollout-search') {
      throw new Error(`Unsupported search result kind: ${result.kind}`);
    }
    const scheduled = this.scheduledVisits.get(result.visitIndex);
    if (!scheduled) {
      throw new Error(
        `Rollout search received result for unscheduled visit ${String(result.visitIndex)}.`
      );
    }
    if (this.mergedVisits.has(result.visitIndex)) {
      throw new Error(
        `Rollout search received duplicate result for visit ${String(result.visitIndex)}.`
      );
    }
    if (scheduled.actionKey !== result.actionKey) {
      throw new Error(
        `Rollout search visit ${String(result.visitIndex)} returned ${result.actionKey} but expected ${scheduled.actionKey}.`
      );
    }

    this.mergedVisits.add(result.visitIndex);
    this.mergedVisitTotal += 1;
    this.pendingVisits.set(
      result.actionKey,
      Math.max(0, (this.pendingVisits.get(result.actionKey) ?? 0) - 1)
    );
    this.rootVisits.set(
      result.actionKey,
      (this.rootVisits.get(result.actionKey) ?? 0) + 1
    );
    this.rootValueSum.set(
      result.actionKey,
      (this.rootValueSum.get(result.actionKey) ?? 0) + result.score
    );
    this.simulatedActionSteps += result.simulatedActionSteps;
    if (result.terminatedBeforeDepthLimit) {
      this.rootTerminalRollouts.set(
        result.actionKey,
        (this.rootTerminalRollouts.get(result.actionKey) ?? 0) + 1
      );
      this.terminalRollouts += 1;
    }
  }

  finish({
    parallelWorkers,
    parallelBatches,
    parallelBatchSize,
  }: {
    parallelWorkers?: number;
    parallelBatches?: number;
    parallelBatchSize?: number;
  }): RolloutSearchFinalResult {
    if (this.scheduledVisitCount !== this.rootBudget) {
      throw new Error('Rollout search finished before scheduling all visits.');
    }
    if (this.mergedVisitTotal !== this.rootBudget) {
      throw new Error('Rollout search finished before merging all visits.');
    }

    const actionKey = selectBestRootActionKey({
      expandedKeys: this.expandedKeys,
      visitsByKey: this.rootVisits,
      valueSumByKey: this.rootValueSum,
      priorsByKey: this.rootPriorByKey,
    });
    const action = this.actionByKey.get(actionKey);
    if (!action) {
      throw new Error(
        `Search policy selected root action key that is no longer legal: ${actionKey}.`
      );
    }

    return {
      action,
      actionKey,
      diagnostics: createSearchDecisionDiagnostics({
        config: this.config,
        legalRootActions: this.rankedRootActions.length,
        expandedRootActions: this.expandedKeys.length,
        rootVisitBudget: this.rootBudget,
        simulatedActionSteps: this.simulatedActionSteps,
        terminalRollouts: this.terminalRollouts,
        selectedActionKey: actionKey,
        rootActions: this.rootActionDiagnostics(),
        parallelWorkers,
        parallelBatches,
        parallelBatchSize,
      }),
    };
  }

  private rootActionDiagnostics(): SearchRootActionDiagnostics[] {
    return this.expandedKeys.map((actionKey) => {
      const visits = this.rootVisits.get(actionKey) ?? 0;
      const terminalRollouts = this.rootTerminalRollouts.get(actionKey) ?? 0;
      return {
        actionKey,
        visits,
        meanValue: safeDiv(this.rootValueSum.get(actionKey) ?? 0, visits),
        terminalRollouts,
        terminalRate: safeDiv(terminalRollouts, visits),
        prior: this.rootPriorByKey.get(actionKey) ?? 0,
      };
    });
  }
}

function runRollout(
  initialState: GameState,
  rootPlayer: PlayerId,
  rootAction: GameAction,
  config: SearchPolicyConfig,
  random: RandomFn
): {
  score: number;
  simulatedActionSteps: number;
  terminatedBeforeDepthLimit: boolean;
} {
  let state = stepKnownLegalActionToDecision(initialState, rootAction);
  let depth = 0;
  let simulatedActionSteps = 1;

  while (!isTerminal(state) && depth < config.depth) {
    const actions = legalActionsForState(state);
    if (actions.length === 0) {
      break;
    }
    const nextAction = chooseRolloutAction(
      state,
      actions,
      random,
      config
    );
    state = stepKnownLegalActionToDecision(state, nextAction);
    depth += 1;
    simulatedActionSteps += 1;
  }

  const terminatedBeforeDepthLimit = isTerminal(state) && depth < config.depth;
  if (isTerminal(state)) {
    return {
      score: evaluateSearchTerminalState(state, rootPlayer),
      simulatedActionSteps,
      terminatedBeforeDepthLimit,
    };
  }
  return {
    score: evaluateSearchLeafState(state, rootPlayer),
    simulatedActionSteps,
    terminatedBeforeDepthLimit,
  };
}

function chooseRolloutAction(
  state: GameState,
  actions: readonly GameAction[],
  random: RandomFn,
  config: SearchPolicyConfig
): GameAction {
  if (random() < config.rolloutEpsilon) {
    const keyed = toKeyedActions(actions);
    return keyed[Math.floor(random() * keyed.length)].action;
  }
  const best = bestActionByHeuristic(actions, { state }, config.heuristic);
  if (!best) {
    throw new Error('Rollout search could not select from legal actions.');
  }
  return best.action;
}

function bestActionByHeuristic(
  actions: readonly GameAction[],
  context: Parameters<typeof bestHeuristicAction>[1],
  heuristic: SearchHeuristicVersion | undefined
): KeyedAction | undefined {
  if (heuristic === 'v2') {
    return bestHeuristicV2Action(actions, context);
  }
  return bestHeuristicAction(actions, context);
}

function rankActionsByHeuristic(
  actions: readonly GameAction[],
  context: Parameters<typeof rankHeuristicActions>[1],
  heuristic: SearchHeuristicVersion | undefined
): KeyedAction[] {
  if (heuristic === 'v2') {
    return rankHeuristicV2Actions(actions, context);
  }
  return rankHeuristicActions(actions, context);
}

function priorsByHeuristic(
  actions: readonly GameAction[],
  context: Parameters<typeof heuristicPriorsByKey>[1],
  heuristic: SearchHeuristicVersion | undefined
): Map<string, number> {
  if (heuristic === 'v2') {
    return heuristicV2PriorsByKey(actions, context);
  }
  return heuristicPriorsByKey(actions, context);
}

function randomSeedForVisit(
  randomSeed: string | undefined,
  visitIndex: number
): string | undefined {
  return randomSeed === undefined
    ? undefined
    : `${randomSeed}:rollout-visit:${String(visitIndex)}`;
}

function rolloutSearchWorkerContextId(randomSeed: string): string {
  return `${randomSeed}:rollout-search-worlds`;
}
