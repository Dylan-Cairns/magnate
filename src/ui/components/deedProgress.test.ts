import { describe, expect, it } from 'vitest';

import {
  buildDeedProgressArcPath,
  canonicalDeedProgressRatio,
  shouldAnimateDeedProgress,
  tweenAnimatedDeedProgressRatio,
} from './deedProgress';

describe('deedProgress helpers', () => {
  it('computes canonical progress ratios with clamping', () => {
    expect(canonicalDeedProgressRatio(0, 6)).toBe(0);
    expect(canonicalDeedProgressRatio(1, 6)).toBe(1 / 6);
    expect(canonicalDeedProgressRatio(3, 6)).toBe(0.5);
    expect(canonicalDeedProgressRatio(8, 6)).toBe(1);
    expect(canonicalDeedProgressRatio(3, 0)).toBe(0);
  });

  it('builds deterministic arc geometry for partial progress', () => {
    expect(buildDeedProgressArcPath(0)).toBeNull();
    expect(buildDeedProgressArcPath(1)).toBeNull();
    expect(buildDeedProgressArcPath(1 / 6)).toBe(
      'M 18 2 A 16 16 0 0 1 31.85640646055102 10'
    );
    expect(buildDeedProgressArcPath(0.5)).toBe('M 18 2 A 16 16 0 0 1 18 34');
  });

  it('animates only for upward progress when motion is allowed', () => {
    expect(shouldAnimateDeedProgress(0, 1 / 6)).toBe(true);
    expect(shouldAnimateDeedProgress(1 / 6, 0.5)).toBe(true);
    expect(shouldAnimateDeedProgress(5 / 6, 1)).toBe(true);
    expect(shouldAnimateDeedProgress(0.5, 1 / 6)).toBe(false);
    expect(shouldAnimateDeedProgress(0.5, 0.5)).toBe(false);
  });

  it('tween lands exactly on target at duration end', () => {
    expect(tweenAnimatedDeedProgressRatio(0, 1 / 6, 0, 260)).toBe(0);
    expect(tweenAnimatedDeedProgressRatio(0, 1 / 6, 260, 260)).toBe(1 / 6);
    expect(tweenAnimatedDeedProgressRatio(1 / 6, 0.5, 260, 260)).toBe(0.5);
    expect(tweenAnimatedDeedProgressRatio(5 / 6, 1, 260, 260)).toBe(1);
  });
});
