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

  it('does not dim a card held in a visible hand', () => {
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

    const dimming = buildDeckMapDimming({ viewState });

    expect(dimming.dimmedCardIds.has('6')).toBe(false);
  });

  it('keeps a sold card and its suit undimmed while it flies to the first-shuffle discard', () => {
    const players = [
      makePlayer(PLAYER_A, { hand: [] }),
      makePlayer('PlayerB', { hand: [] }),
    ] as const;
    const viewState = makeGameState({
      deck: {
        draw: [],
        discard: [],
        reshuffles: 0,
      },
      players,
    });
    const canonicalState = makeGameState({
      deck: {
        draw: [],
        discard: ['2', '6'],
        reshuffles: 0,
      },
      players,
    });

    const visibleOnly = buildDeckMapDimming({ viewState });
    expect(visibleOnly.dimmedCardIds.has('6')).toBe(true);
    expect(visibleOnly.dimmedSuits.has('Moons')).toBe(true);

    const withCanonical = buildDeckMapDimming({ viewState, canonicalState });
    expect(withCanonical.dimmedCardIds.has('6')).toBe(false);
    expect(withCanonical.dimmedSuits.has('Moons')).toBe(false);
  });

  it('dims a sold card once the discard stops circulating after the first shuffle', () => {
    const players = [
      makePlayer(PLAYER_A, { hand: [] }),
      makePlayer('PlayerB', { hand: [] }),
    ] as const;
    const viewState = makeGameState({
      deck: {
        draw: [],
        discard: [],
        reshuffles: 1,
      },
      players,
    });
    const canonicalState = makeGameState({
      deck: {
        draw: [],
        discard: ['6'],
        reshuffles: 1,
      },
      players,
    });

    const dimming = buildDeckMapDimming({ viewState, canonicalState });

    expect(dimming.dimmedCardIds.has('6')).toBe(true);
  });

  it('dims a Court only once that Court has left circulation', () => {
    const viewState = makeGameState({
      deck: {
        draw: ['41'],
        discard: [],
        reshuffles: 0,
      },
      players: [
        makePlayer(PLAYER_A, { hand: ['42'] }),
        makePlayer('PlayerB', { hand: [] }),
      ],
    });

    const dimming = buildDeckMapDimming({ viewState });

    expect(dimming.dimmedCardIds.has('41')).toBe(false);
    expect(dimming.dimmedCardIds.has('42')).toBe(false);
    expect(dimming.dimmedCardIds.has('43')).toBe(true);
    expect(dimming.dimmedCardIds.has('44')).toBe(true);
  });

  it('dims a suit icon when its Ace is played, even if other suit cards remain', () => {
    const viewState = makeGameState({
      deck: {
        draw: ['2', '13'],
        discard: [],
        reshuffles: 0,
      },
      players: [
        makePlayer(PLAYER_A, { hand: ['6'] }),
        makePlayer('PlayerB', { hand: [] }),
      ],
    });

    const dimming = buildDeckMapDimming({
      viewState: {
        ...viewState,
        deck: { ...viewState.deck, draw: ['13'] },
        districts: viewState.districts.map((district) =>
          district.id === 'D1'
            ? {
                ...district,
                stacks: {
                  ...district.stacks,
                  [PLAYER_A]: { developed: ['2'] },
                },
              }
            : district
        ),
      },
    });

    expect(dimming.dimmedSuits.has('Moons')).toBe(true);
    expect(dimming.dimmedCardIds.has('6')).toBe(false);
    expect(dimming.dimmedCardIds.has('13')).toBe(false);
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
