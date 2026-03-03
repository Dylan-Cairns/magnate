import { describe, expect, it } from 'vitest';

import {
  rootActionCountBucket,
  summarizeLatencies,
  summarizeSearchWork,
  wilsonInterval,
} from './stats';

describe('bot evaluation stats', () => {
  it('computes a Wilson interval within probability bounds', () => {
    const interval = wilsonInterval(6, 10);

    expect(interval.low).toBeGreaterThan(0);
    expect(interval.high).toBeLessThan(1);
    expect(interval.low).toBeLessThan(0.6);
    expect(interval.high).toBeGreaterThan(0.6);
  });

  it('summarizes latency samples with nearest-rank percentiles', () => {
    expect(summarizeLatencies([5, 1, 2, 10, 3])).toEqual({
      actions: 5,
      meanMs: 4.2,
      p50Ms: 3,
      p95Ms: 10,
      maxMs: 10,
    });
  });

  it('classifies stable legal-root-action bucket boundaries', () => {
    expect(rootActionCountBucket(4)).toBe('2-4');
    expect(rootActionCountBucket(5)).toBe('5-8');
    expect(rootActionCountBucket(64)).toBe('33-64');
    expect(rootActionCountBucket(65)).toBe('65+');
  });

  it('summarizes rollout-search work counters', () => {
    expect(
      summarizeSearchWork([
        {
          kind: 'search',
          legalRootActions: 5,
          expandedRootActions: 5,
          rootVisitBudget: 10,
          configProxyCost: 30,
          maxSimulatedActionSteps: 40,
          simulatedActionSteps: 25,
          terminalRollouts: 2,
        },
      ])
    ).toEqual({
      searchedDecisions: 1,
      rootVisits: 10,
      configProxyCost: 30,
      maxSimulatedActionSteps: 40,
      simulatedActionSteps: 25,
      stepUtilization: 0.625,
      meanSimulatedActionSteps: 25,
      terminalRollouts: 2,
    });
  });
});
