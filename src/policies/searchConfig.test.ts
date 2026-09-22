import { describe, expect, it } from 'vitest';

import {
  DEFAULT_COURT_VALUE_SCALE,
  resolveSearchConfig,
} from './searchConfig';

describe('search policy config', () => {
  it('defaults courtValueScale to the deployed valuation', () => {
    expect(resolveSearchConfig().courtValueScale).toBe(
      DEFAULT_COURT_VALUE_SCALE
    );
    expect(
      resolveSearchConfig({
        worlds: 10,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      }).courtValueScale
    ).toBe(DEFAULT_COURT_VALUE_SCALE);
  });

  it('preserves an explicit disabled court term', () => {
    expect(
      resolveSearchConfig({ courtValueScale: 0, heuristic: 'v2' })
        .courtValueScale
    ).toBe(0);
  });

  it('accepts scales above one for tuning', () => {
    expect(
      resolveSearchConfig({ courtValueScale: 1.5, heuristic: 'v2' })
        .courtValueScale
    ).toBe(1.5);
  });

  it('rejects invalid court value scales', () => {
    expect(() => resolveSearchConfig({ courtValueScale: -0.1 })).toThrow(
      'courtValueScale'
    );
    expect(() => resolveSearchConfig({ courtValueScale: Number.NaN })).toThrow(
      'courtValueScale'
    );
    expect(() =>
      resolveSearchConfig({ courtValueScale: Number.POSITIVE_INFINITY })
    ).toThrow('courtValueScale');
  });
});
