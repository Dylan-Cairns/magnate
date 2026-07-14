import { describe, expect, it } from 'vitest';

import { canonicalDeedProgressRatio } from './deedProgress';

describe('deedProgress helpers', () => {
  it('computes canonical progress ratios with clamping', () => {
    expect(canonicalDeedProgressRatio(0, 6)).toBe(0);
    expect(canonicalDeedProgressRatio(1, 6)).toBe(1 / 6);
    expect(canonicalDeedProgressRatio(3, 6)).toBe(0.5);
    expect(canonicalDeedProgressRatio(8, 6)).toBe(1);
    expect(canonicalDeedProgressRatio(3, 0)).toBe(0);
  });

});
