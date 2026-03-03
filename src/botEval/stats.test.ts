import { describe, expect, it } from 'vitest';

import { summarizeLatencies, wilsonInterval } from './stats';

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
});
