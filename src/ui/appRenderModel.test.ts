import { describe, expect, it } from 'vitest';

import {
  makeGameState,
  makePlayer,
  PLAYER_A,
} from '../engine/__tests__/fixtures';
import {
  buildDeckMapDimming,
  isVisibleIncomeChoicePhase,
} from './appRenderModel';

describe('app render model', () => {
  it('uses the visible state, not a canonical pending phase, for income-choice display', () => {
    const visibleActionWindow = makeGameState({ phase: 'ActionWindow' });
    const visibleIncomeChoice = makeGameState({
      phase: 'CollectIncome',
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Suns'],
        },
      ],
    });

    expect(isVisibleIncomeChoicePhase(visibleActionWindow)).toBe(false);
    expect(isVisibleIncomeChoicePhase(visibleIncomeChoice)).toBe(true);
  });

  it('dims the deck map from visible circulation only', () => {
    const viewState = makeGameState({
      deck: {
        draw: [],
        discard: [],
        reshuffles: 0,
      },
      players: [
        makePlayer(PLAYER_A, { hand: ['6'] }),
        makePlayer('PlayerB', { hand: [] }),
      ],
    });

    const dimming = buildDeckMapDimming({
      deckMapInteractive: true,
      viewState,
    });

    expect(dimming.dimmedCardIds.has('6')).toBe(false);
    expect(
      buildDeckMapDimming({
        deckMapInteractive: false,
        viewState,
      }).dimmedCardIds.size
    ).toBe(0);
  });
});
