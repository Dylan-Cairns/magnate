import type { SearchDecisionDiagnostics } from '../policies/types';
import {
  ROOT_ACTION_COUNT_BUCKETS,
  type ConfidenceInterval,
  type LatencySummary,
  type RootActionCountBucket,
  type SearchWorkSummary,
} from './types';

export interface SearchDecisionLatencySample {
  legalRootActions: number;
  latencyMs: number;
}

export function wilsonInterval(
  successes: number,
  trials: number,
  z = 1.96
): ConfidenceInterval {
  if (!Number.isInteger(successes) || successes < 0 || successes > trials) {
    throw new Error('successes must be an integer in [0, trials].');
  }
  if (!Number.isInteger(trials) || trials <= 0) {
    throw new Error('trials must be a positive integer.');
  }
  if (!Number.isFinite(z) || z <= 0) {
    throw new Error('z must be a positive finite number.');
  }

  const pHat = successes / trials;
  const zSquared = z * z;
  const denominator = 1 + zSquared / trials;
  const center = (pHat + zSquared / (2 * trials)) / denominator;
  const margin =
    (z * Math.sqrt((pHat * (1 - pHat) + zSquared / (4 * trials)) / trials)) /
    denominator;

  return {
    low: Math.max(0, center - margin),
    high: Math.min(1, center + margin),
  };
}

export function summarizeLatencies(values: readonly number[]): LatencySummary {
  if (values.length === 0) {
    return {
      actions: 0,
      meanMs: 0,
      p50Ms: 0,
      p95Ms: 0,
      maxMs: 0,
    };
  }

  const sorted = [...values].sort((left, right) => left - right);
  return {
    actions: sorted.length,
    meanMs: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    p50Ms: nearestRankPercentile(sorted, 0.5),
    p95Ms: nearestRankPercentile(sorted, 0.95),
    maxMs: sorted[sorted.length - 1],
  };
}

export function rootActionCountBucket(
  legalRootActions: number
): RootActionCountBucket {
  if (!Number.isInteger(legalRootActions) || legalRootActions < 2) {
    throw new Error('legalRootActions must be an integer >= 2.');
  }
  if (legalRootActions <= 4) {
    return '2-4';
  }
  if (legalRootActions <= 8) {
    return '5-8';
  }
  if (legalRootActions <= 16) {
    return '9-16';
  }
  if (legalRootActions <= 32) {
    return '17-32';
  }
  if (legalRootActions <= 64) {
    return '33-64';
  }
  return '65+';
}

export function summarizeSearchWork(
  diagnostics: readonly SearchDecisionDiagnostics[]
): SearchWorkSummary {
  const summary: SearchWorkSummary = {
    searchedDecisions: diagnostics.length,
    rootVisits: 0,
    configProxyCost: 0,
    maxSimulatedActionSteps: 0,
    simulatedActionSteps: 0,
    stepUtilization: 0,
    meanSimulatedActionSteps: 0,
    terminalRollouts: 0,
    terminalRate: 0,
    meanSelectedActionValue: 0,
    meanSelectedActionVisits: 0,
    meanSelectedActionTerminalRate: 0,
  };
  for (const diagnostic of diagnostics) {
    summary.rootVisits += diagnostic.rootVisitBudget;
    summary.configProxyCost += diagnostic.configProxyCost;
    summary.maxSimulatedActionSteps += diagnostic.maxSimulatedActionSteps;
    summary.simulatedActionSteps += diagnostic.simulatedActionSteps;
    summary.terminalRollouts += diagnostic.terminalRollouts;
    summary.meanSelectedActionValue += diagnostic.selectedActionMeanValue;
    summary.meanSelectedActionVisits += diagnostic.selectedActionVisits;
    summary.meanSelectedActionTerminalRate +=
      diagnostic.selectedActionTerminalRate;
  }
  summary.stepUtilization = safeDiv(
    summary.simulatedActionSteps,
    summary.maxSimulatedActionSteps
  );
  summary.meanSimulatedActionSteps = safeDiv(
    summary.simulatedActionSteps,
    summary.searchedDecisions
  );
  summary.terminalRate = safeDiv(summary.terminalRollouts, summary.rootVisits);
  summary.meanSelectedActionValue = safeDiv(
    summary.meanSelectedActionValue,
    summary.searchedDecisions
  );
  summary.meanSelectedActionVisits = safeDiv(
    summary.meanSelectedActionVisits,
    summary.searchedDecisions
  );
  summary.meanSelectedActionTerminalRate = safeDiv(
    summary.meanSelectedActionTerminalRate,
    summary.searchedDecisions
  );
  return summary;
}

export function summarizeSearchLatenciesByRootActionCount(
  samples: readonly SearchDecisionLatencySample[]
): Record<RootActionCountBucket, LatencySummary> {
  const valuesByBucket = new Map<RootActionCountBucket, number[]>(
    ROOT_ACTION_COUNT_BUCKETS.map((bucket) => [bucket, []])
  );
  for (const sample of samples) {
    valuesByBucket
      .get(rootActionCountBucket(sample.legalRootActions))!
      .push(sample.latencyMs);
  }
  return Object.fromEntries(
    ROOT_ACTION_COUNT_BUCKETS.map((bucket) => [
      bucket,
      summarizeLatencies(valuesByBucket.get(bucket) ?? []),
    ])
  ) as Record<RootActionCountBucket, LatencySummary>;
}

function nearestRankPercentile(
  sortedValues: readonly number[],
  percentile: number
): number {
  if (sortedValues.length === 0) {
    return 0;
  }
  const index = Math.max(
    0,
    Math.min(
      sortedValues.length - 1,
      Math.ceil(percentile * sortedValues.length) - 1
    )
  );
  return sortedValues[index];
}

function safeDiv(total: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return total / count;
}
