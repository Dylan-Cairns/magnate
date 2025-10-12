import { describe, expect, it } from 'vitest';

import { makeGameState, PLAYER_A, PLAYER_B } from '../engine/__tests__/fixtures';
import type { GameState, PlayerId } from '../engine/types';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from './turnReset';

const HUMAN_PLAYER: PlayerId = PLAYER_A;

function makeAnchor(state: GameState, turn = state.turn, playerId: PlayerId = HUMAN_PLAYER): TurnResetAnchor {
  return {
    turn,
    playerId,
    state,
  };
}

describe('shouldCaptureTurnResetAnchor', () => {
  it('captures on human action window before card play when no anchor exists', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
      turn: 3,
    });

    expect(shouldCaptureTurnResetAnchor(state, PLAYER_A, HUMAN_PLAYER, null)).toBe(true);
  });

  it('does not capture outside human pre-card action window', () => {
    const humanPostCard = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: true,
    });
    const botTurn = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 1,
      cardPlayedThisTurn: false,
    });
    const collectIncome = makeGameState({
      phase: 'CollectIncome',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
    });

    expect(shouldCaptureTurnResetAnchor(humanPostCard, PLAYER_A, HUMAN_PLAYER, null)).toBe(false);
    expect(shouldCaptureTurnResetAnchor(botTurn, PLAYER_B, HUMAN_PLAYER, null)).toBe(false);
    expect(shouldCaptureTurnResetAnchor(collectIncome, PLAYER_A, HUMAN_PLAYER, null)).toBe(false);
  });

  it('captures again when a new human turn starts', () => {
    const priorAnchorState = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
      turn: 3,
    });
    const anchor = makeAnchor(priorAnchorState, 3, PLAYER_A);
    const newTurnState = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
      turn: 4,
    });

    expect(shouldCaptureTurnResetAnchor(newTurnState, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(true);
  });

  it('does not recapture when anchor already matches current turn/player', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
      turn: 5,
    });
    const anchor = makeAnchor(state, 5, PLAYER_A);

    expect(shouldCaptureTurnResetAnchor(state, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(false);
  });
});

describe('canUseTurnReset', () => {
  it('is false when no anchor exists', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      turn: 2,
    });

    expect(canUseTurnReset(state, PLAYER_A, HUMAN_PLAYER, null)).toBe(false);
  });

  it('is true for human action window when state diverged from anchor', () => {
    const anchorState = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      cardPlayedThisTurn: false,
      turn: 6,
    });
    const current = {
      ...anchorState,
      cardPlayedThisTurn: true,
    };
    const anchor = makeAnchor(anchorState, 6, PLAYER_A);

    expect(canUseTurnReset(current, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(true);
  });

  it('is false for bot turns and phase/turn mismatch', () => {
    const anchorState = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      turn: 4,
    });
    const anchor = makeAnchor(anchorState, 4, PLAYER_A);
    const botTurnState = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 1,
      turn: 4,
    });
    const collectIncomeState = {
      ...anchorState,
      phase: 'CollectIncome' as const,
    };
    const differentTurnState = {
      ...anchorState,
      turn: 5,
    };

    expect(canUseTurnReset(botTurnState, PLAYER_B, HUMAN_PLAYER, anchor)).toBe(false);
    expect(canUseTurnReset(collectIncomeState, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(false);
    expect(canUseTurnReset(differentTurnState, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(false);
  });

  it('is false when current state is already the anchor snapshot', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 0,
      turn: 7,
    });
    const anchor = makeAnchor(state, 7, PLAYER_A);

    expect(canUseTurnReset(state, PLAYER_A, HUMAN_PLAYER, anchor)).toBe(false);
  });
});
