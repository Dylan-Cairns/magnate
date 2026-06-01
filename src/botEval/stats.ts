import type { ConfidenceInterval, LatencySummary } from './types';

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
