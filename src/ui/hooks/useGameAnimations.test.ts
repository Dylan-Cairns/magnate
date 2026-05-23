import { describe, expect, it } from 'vitest';

import { makeGameState, makePlayer } from '../../engine/__tests__/fixtures';
import type { CardFlight } from '../animations/types';
import {
  activeHighlightOverrideForTransition,
  turnCycleStartDelayForTransition,
} from './useGameAnimations';

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

describe('activeHighlightOverrideForTransition', () => {
  it('holds the outgoing player highlight while their draw flight settles', () => {
    const previousState = makeGameState({
      players: [
        makePlayer('PlayerA', { hand: ['6'] }),
        makePlayer('PlayerB', { hand: ['7'] }),
      ],
      activePlayerIndex: 1,
    });

    expect(
      activeHighlightOverrideForTransition(
        { type: 'end-turn' },
        previousState,
        [DRAW_FLIGHT]
      )
    ).toBe('PlayerB');
  });

  it('does not override highlights without an end-turn draw flight', () => {
    const previousState = makeGameState();

    expect(
      activeHighlightOverrideForTransition(
        { type: 'end-turn' },
        previousState,
        []
      )
    ).toBeNull();
    expect(
      activeHighlightOverrideForTransition(
        { type: 'sell-card', cardId: '6' },
        previousState,
        [DRAW_FLIGHT]
      )
    ).toBeNull();
  });

  it('delays turn-cycle visuals until end-turn draw flights settle', () => {
    expect(
      turnCycleStartDelayForTransition({ type: 'end-turn' }, [DRAW_FLIGHT])
    ).toBe(300);
    expect(
      turnCycleStartDelayForTransition({ type: 'end-turn' }, [])
    ).toBe(0);
    expect(
      turnCycleStartDelayForTransition(
        { type: 'sell-card', cardId: '6' },
        [DRAW_FLIGHT]
      )
    ).toBe(0);
  });
});
