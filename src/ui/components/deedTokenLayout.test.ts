import { afterEach, describe, expect, it } from 'vitest';

import {
  clearAllDeedTokenLayouts,
  layoutDeedTokensBySide,
  resetDeedTokenLayout,
} from './deedTokenLayout';

describe('deedTokenLayout', () => {
  afterEach(() => {
    clearAllDeedTokenLayouts();
  });

  it('keeps first human suit on its original side when a second suit is added', () => {
    const first = layoutDeedTokensBySide('6', 'human', [{ suit: 'Suns', count: 1 }]);
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
    const first = layoutDeedTokensBySide('7', 'bot', [{ suit: 'Leaves', count: 1 }]);
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

    const next = layoutDeedTokensBySide('8', 'human', [{ suit: 'Waves', count: 1 }]);
    expect(next.left.map((entry) => entry.suit)).toEqual(['Waves']);
    expect(next.right).toHaveLength(0);
  });
});

