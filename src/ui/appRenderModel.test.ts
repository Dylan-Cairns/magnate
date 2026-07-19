import { describe, expect, it } from 'vitest';

import {
  makeGameState,
  makePlayer,
  PLAYER_A,
} from '../engine/__tests__/fixtures';
import {
  awaitingIncomeChoiceCardIds,
  buildDeckMapDimming,
  isVisibleIncomeChoicePhase,
  shouldHideBotWaitMessageDuringAnimationLock,
} from './appRenderModel';

describe('app render model', () => {
  it('highlights only deed income choices that are still awaiting input', () => {
    const viewState = makeGameState({
      phase: 'CollectIncome',
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Suns'],
        },
        {
          playerId: 'PlayerB',
          districtId: 'D2',
          cardId: '8',
          suits: ['Waves', 'Leaves'],
        },
      ],
      submittedIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
        },
      ],
    });

    expect(awaitingIncomeChoiceCardIds(viewState)).toEqual(['8']);
  });

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

  it('keeps bot thinking visible during locked income-choice resolution', () => {
    expect(
      shouldHideBotWaitMessageDuringAnimationLock({
        isAnimationLock: true,
        isIncomeChoicePhase: true,
        botThinking: true,
      })
    ).toBe(false);
    expect(
      shouldHideBotWaitMessageDuringAnimationLock({
        isAnimationLock: true,
        isIncomeChoicePhase: true,
        botThinking: false,
      })
    ).toBe(true);
    expect(
      shouldHideBotWaitMessageDuringAnimationLock({
        isAnimationLock: true,
        isIncomeChoicePhase: false,
        botThinking: true,
      })
    ).toBe(true);
  });
});
