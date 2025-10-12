import { describe, expect, it } from 'vitest';

import { CARD_BY_ID } from './cards';
import { initialSetup } from './deck';
import { newGame } from './game';

function resourceTotal(resources: Record<string, number>): number {
  return Object.values(resources).reduce((sum, count) => sum + count, 0);
}

describe('newGame', () => {
  it('is deterministic for a fixed seed', () => {
    const first = newGame('new-game-seed');
    const second = newGame('new-game-seed');
    expect(first).toEqual(second);
  });

  it('builds district suit masks from setup marker cards', () => {
    const seed = 'district-marker-seed';
    const setup = initialSetup(seed);
    const state = newGame(seed);

    setup.districts.forEach((markerId, index) => {
      const marker = CARD_BY_ID[markerId];
      const expected = marker.kind === 'Pawn' ? [...marker.suits] : [];
      expect(state.districts[index].markerSuitMask).toEqual(expected);
    });
  });

  it('initializes both players with 3 crowns, 3 hand cards, and 3 starting resources', () => {
    const state = newGame('player-setup-seed');

    state.players.forEach((player) => {
      expect(player.crowns).toHaveLength(3);
      expect(player.hand).toHaveLength(3);
      expect(resourceTotal(player.resources)).toBe(3);
    });
  });

  it('supports selecting which player takes the first turn', () => {
    const firstA = newGame('first-player-seed', { firstPlayer: 'PlayerA' });
    const firstB = newGame('first-player-seed', { firstPlayer: 'PlayerB' });

    expect(firstA.players[firstA.activePlayerIndex].id).toBe('PlayerA');
    expect(firstB.players[firstB.activePlayerIndex].id).toBe('PlayerB');
  });

  it('starts with turn-state flags reset for a fresh turn', () => {
    const state = newGame('turn-state-seed');

    expect(state.phase).toBe('StartTurn');
    expect(state.turn).toBe(1);
    expect(state.cardPlayedThisTurn).toBe(false);
    expect(state.pendingIncomeChoices).toBeUndefined();
    expect(state.incomeChoiceReturnPlayerId).toBeUndefined();
  });
});
