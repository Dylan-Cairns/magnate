import { describe, expect, it } from 'vitest';

import {
  DEFAULT_DEED_POTENTIAL_BASE,
  resolveSearchConfig,
} from './searchConfig';

describe('search policy config', () => {
  it('defaults deedPotentialBase to the deployed floor', () => {
    expect(resolveSearchConfig().deedPotentialBase).toBe(
      DEFAULT_DEED_POTENTIAL_BASE
    );
    expect(
      resolveSearchConfig({
        worlds: 10,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      }).deedPotentialBase
    ).toBe(DEFAULT_DEED_POTENTIAL_BASE);
  });

  it('preserves an explicit legacy zero base', () => {
    expect(
      resolveSearchConfig({ deedPotentialBase: 0, heuristic: 'v2' })
        .deedPotentialBase
    ).toBe(0);
  });

  it('rejects out-of-range deed potential bases', () => {
    expect(() => resolveSearchConfig({ deedPotentialBase: -0.1 })).toThrow(
      'deedPotentialBase'
    );
    expect(() => resolveSearchConfig({ deedPotentialBase: 1.1 })).toThrow(
      'deedPotentialBase'
    );
    expect(() => resolveSearchConfig({ deedPotentialBase: Number.NaN })).toThrow(
      'deedPotentialBase'
    );
  });
});
