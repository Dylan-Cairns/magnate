import { describe, expect, it } from 'vitest';

import type { CardFlight } from '../animations/types';
import { turnCycleStartDelayForTransition } from './useGameAnimations';

const DRAW_FLIGHT: CardFlight = {
  id: 'draw-1',
  variant: 'draw',
  visual: 'back',
  cardId: '6',
  isDeed: false,
  perspective: 'human',
  startX: 10,
  startY: 20,
  endX: 30,
  endY: 40,
  startWidth: 50,
  startHeight: 70,
  endWidth: 50,
  endHeight: 70,
  delayMs: 0,
};

describe('turnCycleStartDelayForTransition', () => {
  it('delays turn-cycle visuals until end-turn draw flights settle', () => {
    expect(
      turnCycleStartDelayForTransition({ type: 'end-turn' }, [DRAW_FLIGHT])
    ).toBe(300);
    expect(turnCycleStartDelayForTransition({ type: 'end-turn' }, [])).toBe(0);
    expect(
      turnCycleStartDelayForTransition({ type: 'sell-card', cardId: '6' }, [
        DRAW_FLIGHT,
      ])
    ).toBe(0);
  });
});
