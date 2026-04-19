import type { SearchPolicyConfig } from './searchConfig';
import type {
  SearchDecisionDiagnostics,
  SearchRootActionDiagnostics,
} from './types';

export function progressiveTargetActionCount(
  totalActions: number,
  initialActions: number,
  visits: number
): number {
  if (totalActions <= 0) {
    return 0;
  }
  const base = Math.max(1, Math.min(initialActions, totalActions));
  const widened = base + Math.floor(Math.sqrt(Math.max(0, visits) + 1));
  return Math.min(totalActions, widened);
}

export function selectRootUcbAction(
  actionKeys: readonly string[],
  visitsByKey: ReadonlyMap<string, number>,
  valueSumByKey: ReadonlyMap<string, number>,
  priorsByKey: ReadonlyMap<string, number>,
  totalVisits: number,
  cPuct = 1,
  pendingVisitsByKey: ReadonlyMap<string, number> = new Map()
): string {
  if (actionKeys.length === 0) {
    throw new Error('selectRootUcbAction requires at least one action key.');
  }
  if (!(cPuct > 0)) {
    throw new Error('selectRootUcbAction requires cPuct > 0.');
  }

  const sqrtParent = Math.sqrt(Math.max(0, totalVisits) + 1);
  let bestActionKey = actionKeys[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const actionKey of actionKeys) {
    const visits = visitsByKey.get(actionKey) ?? 0;
    const pendingVisits = pendingVisitsByKey.get(actionKey) ?? 0;
    const effectiveVisits = visits + pendingVisits;
    const valueSum = valueSumByKey.get(actionKey) ?? 0;
    const q = safeDiv(valueSum, visits);
    const prior = priorsByKey.get(actionKey) ?? 0;
    const score = q + (cPuct * prior * sqrtParent) / (1 + effectiveVisits);
    if (
      score > bestScore ||
      (approximatelyEqual(score, bestScore) &&
        actionKey.localeCompare(bestActionKey) < 0)
    ) {
      bestActionKey = actionKey;
      bestScore = score;
    }
  }
  return bestActionKey;
}

export function selectBestRootActionKey({
  expandedKeys,
  visitsByKey,
  valueSumByKey,
  priorsByKey,
}: {
  expandedKeys: readonly string[];
  visitsByKey: ReadonlyMap<string, number>;
  valueSumByKey: ReadonlyMap<string, number>;
  priorsByKey: ReadonlyMap<string, number>;
}): string {
  if (expandedKeys.length === 0) {
    throw new Error('selectBestRootActionKey requires expanded actions.');
  }

  let bestActionKey = expandedKeys[0];
  let bestVisits = visitsByKey.get(bestActionKey) ?? 0;
  let bestValue = safeDiv(valueSumByKey.get(bestActionKey) ?? 0, bestVisits);
  let bestPrior = priorsByKey.get(bestActionKey) ?? 0;
  for (const actionKey of expandedKeys.slice(1)) {
    const visits = visitsByKey.get(actionKey) ?? 0;
    const value = safeDiv(valueSumByKey.get(actionKey) ?? 0, visits);
    const prior = priorsByKey.get(actionKey) ?? 0;
    if (
      visits > bestVisits ||
      (visits === bestVisits &&
        (value > bestValue ||
          (approximatelyEqual(value, bestValue) &&
            (prior > bestPrior ||
              (approximatelyEqual(prior, bestPrior) &&
                actionKey.localeCompare(bestActionKey) < 0)))))
    ) {
      bestActionKey = actionKey;
      bestVisits = visits;
      bestValue = value;
      bestPrior = prior;
    }
  }
  return bestActionKey;
}

export function createSearchDecisionDiagnostics({
  config,
  legalRootActions,
  expandedRootActions,
  rootVisitBudget,
  simulatedActionSteps,
  terminalRollouts,
  selectedActionKey,
  rootActions,
  parallelWorkers,
  parallelBatches,
  parallelBatchSize,
}: {
  config: SearchPolicyConfig;
  legalRootActions: number;
  expandedRootActions: number;
  rootVisitBudget: number;
  simulatedActionSteps: number;
  terminalRollouts: number;
  selectedActionKey: string;
  rootActions: readonly SearchRootActionDiagnostics[];
  parallelWorkers?: number;
  parallelBatches?: number;
  parallelBatchSize?: number;
}): SearchDecisionDiagnostics {
  const selectedRootAction = rootActions.find(
    (entry) => entry.actionKey === selectedActionKey
  );
  if (!selectedRootAction) {
    throw new Error(
      `Search diagnostics selected action missing from root actions: ${selectedActionKey}.`
    );
  }
  return {
    kind: 'search',
    heuristic: config.heuristic ?? 'v1',
    stochasticSimulation: 'common-random-scenarios-v1',
    legalRootActions,
    expandedRootActions,
    rootVisitBudget,
    configProxyCost: rootVisitBudget * config.depth,
    maxSimulatedActionSteps: rootVisitBudget * (config.depth + 1),
    simulatedActionSteps,
    terminalRollouts,
    terminalRate: safeDiv(terminalRollouts, rootVisitBudget),
    selectedActionKey,
    selectedActionVisits: selectedRootAction.visits,
    selectedActionMeanValue: selectedRootAction.meanValue,
    selectedActionTerminalRollouts: selectedRootAction.terminalRollouts,
    selectedActionTerminalRate: selectedRootAction.terminalRate,
    rootActions: rootActions.map((entry) => ({ ...entry })),
    ...(parallelWorkers !== undefined ? { parallelWorkers } : {}),
    ...(parallelBatches !== undefined ? { parallelBatches } : {}),
    ...(parallelBatchSize !== undefined ? { parallelBatchSize } : {}),
  };
}

export function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

export function safeDiv(total: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return total / count;
}
