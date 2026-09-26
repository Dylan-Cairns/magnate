import { afterEach, describe, expect, it } from 'vitest';

import {
  clearAllDeedTokenLayouts,
  commitDeedTokenLayout,
  layoutDeedTokensBySide,
  planDeedTokenLayout,
  resetDeedTokenLayout,
} from './deedTokenLayout';

describe('deedTokenLayout', () => {
  afterEach(() => {
    clearAllDeedTokenLayouts();
  });

  it('keeps first human suit on its original side when a second suit is added', () => {
    const first = layoutDeedTokensBySide('6', 'human', [
      { suit: 'Suns', count: 1 },
    ]);
    expect(first.left.map((entry) => entry.suit)).toEqual(['Suns']);
    expect(first.right).toHaveLength(0);

    const second = layoutDeedTokensBySide('6', 'human', [
      { suit: 'Moons', count: 1 },
      { suit: 'Suns', count: 1 },
    ]);
    expect(second.left.map((entry) => entry.suit)).toEqual(['Suns']);
    expect(second.right.map((entry) => entry.suit)).toEqual(['Moons']);
  });

  it('keeps first bot suit on its original side when a second suit is added', () => {
    const first = layoutDeedTokensBySide('7', 'bot', [
      { suit: 'Leaves', count: 1 },
    ]);
    expect(first.left).toHaveLength(0);
    expect(first.right.map((entry) => entry.suit)).toEqual(['Leaves']);

    const second = layoutDeedTokensBySide('7', 'bot', [
      { suit: 'Leaves', count: 1 },
      { suit: 'Wyrms', count: 1 },
    ]);
    expect(second.right.map((entry) => entry.suit)).toEqual(['Leaves']);
    expect(second.left.map((entry) => entry.suit)).toEqual(['Wyrms']);
  });

  it('resets per-card side memory when explicitly reset', () => {
    layoutDeedTokensBySide('8', 'human', [{ suit: 'Knots', count: 1 }]);
    resetDeedTokenLayout('8', 'human');

    const next = layoutDeedTokensBySide('8', 'human', [
      { suit: 'Waves', count: 1 },
    ]);
    expect(next.left.map((entry) => entry.suit)).toEqual(['Waves']);
    expect(next.right).toHaveLength(0);
  });

  it('plans a layout without recording assignments in shared memory', () => {
    const planned = planDeedTokenLayout('9', 'human', [
      { suit: 'Suns', count: 1 },
    ]);
    expect(planned.left.map((entry) => entry.suit)).toEqual(['Suns']);

    // The pull render was pure: a later commit for the same card starts fresh,
    // so the first committed suit still takes the default side.
    const committed = commitDeedTokenLayout('9', 'human', [
      { suit: 'Moons', count: 1 },
    ]);
    expect(committed.left.map((entry) => entry.suit)).toEqual(['Moons']);
    expect(committed.right).toHaveLength(0);
  });

  it('plans from committed memory so sides stay stable across a later render', () => {
    commitDeedTokenLayout('10', 'human', [{ suit: 'Suns', count: 1 }]);

    const planned = planDeedTokenLayout('10', 'human', [
      { suit: 'Moons', count: 1 },
      { suit: 'Suns', count: 1 },
    ]);
    expect(planned.left.map((entry) => entry.suit)).toEqual(['Suns']);
    expect(planned.right.map((entry) => entry.suit)).toEqual(['Moons']);
  });

  it('resets shared memory when commit is asked to reset while empty', () => {
    commitDeedTokenLayout('11', 'human', [{ suit: 'Suns', count: 1 }]);

    const reset = commitDeedTokenLayout('11', 'human', [], {
      resetWhenEmpty: true,
    });
    expect(reset).toEqual({ left: [], right: [] });

    const next = commitDeedTokenLayout('11', 'human', [
      { suit: 'Waves', count: 1 },
    ]);
    expect(next.left.map((entry) => entry.suit)).toEqual(['Waves']);
  });
});
