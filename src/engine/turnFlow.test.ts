import { describe, expect, it } from 'vitest';

import {
  PLAYER_A,
  PLAYER_B,
  makeGameState,
  makePlayer,
} from './__tests__/fixtures';
import { advanceToDecision } from './turnFlow';

describe('advanceToDecision', () => {
  it('returns the same state when already at a decision phase', () => {
    const state = makeGameState({ phase: 'PlayCard' });
    expect(advanceToDecision(state)).toBe(state);
  });

  it('auto-advances StartTurn through CollectIncome into OptionalTrade', () => {
    const state = makeGameState({
      phase: 'StartTurn',
      turn: 5,
      activePlayerIndex: 1,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(5);
    expect(advanced.activePlayerIndex).toBe(1);
  });

  it('resolves DrawCard by drawing to active player hand and handing off turn', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 1,
      activePlayerIndex: 0,
      players: [
        makePlayer(PLAYER_A, { hand: ['6'] }),
        makePlayer(PLAYER_B, { hand: ['7'] }),
      ],
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(2);
    expect(advanced.activePlayerIndex).toBe(1);
    expect(advanced.players[0].hand).toEqual(['6', '6']);
    expect(advanced.deck.draw).toEqual(['7', '8']);
  });

  it('on first second-exhaustion draw, enters final-turn countdown and continues', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 10,
      activePlayerIndex: 0,
      deck: { draw: [], discard: [], reshuffles: 1 },
      exhaustionStage: 1,
      finalTurnsRemaining: undefined,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(11);
    expect(advanced.activePlayerIndex).toBe(1);
    expect(advanced.exhaustionStage).toBe(2);
    expect(advanced.finalTurnsRemaining).toBe(1);
  });

  it('ends the game when final-turn countdown reaches zero on DrawCard', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 11,
      activePlayerIndex: 1,
      deck: { draw: [], discard: [], reshuffles: 2 },
      exhaustionStage: 2,
      finalTurnsRemaining: 1,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('GameOver');
    expect(advanced.turn).toBe(12);
    expect(advanced.activePlayerIndex).toBe(0);
    expect(advanced.finalTurnsRemaining).toBe(0);
  });
});
