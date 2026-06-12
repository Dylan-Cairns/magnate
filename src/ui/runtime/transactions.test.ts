import { describe, expect, it } from 'vitest';

import type { GameState } from '../../engine/types';
import {
  PLAYER_A,
  PLAYER_B,
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
} from '../../engine/__tests__/fixtures';
import {
  buildGameTransaction,
  deriveGamePresentationEvents,
  incomeTokenSourceKey,
} from './transactions';

describe('buildGameTransaction', () => {
  it('wraps an action with previous/next state and semantic presentation events', () => {
    const { previous, next } = makeEndTurnStates();

    const transaction = buildGameTransaction({
      previousState: previous,
      action: { type: 'end-turn' },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-end-turn',
      stepToDecision: () => next,
    });

    expect(transaction.id).toBe('tx-end-turn');
    expect(transaction.previousState).toBe(previous);
    expect(transaction.nextState).toBe(next);
    expect(transaction.events.map((event) => event.type)).toEqual([
      'action-started',
      'draw-card',
      'income-roll',
      'tax-token-lost',
      'tax-token-lost',
      'income-token-gained',
      'income-token-gained',
      'active-player-changed',
      'transaction-settled',
    ]);
  });
});

describe('deriveGamePresentationEvents', () => {
  it('derives turn-cycle draw, roll, tax, and income events from state transition semantics', () => {
    const { previous, next } = makeEndTurnStates();

    const events = deriveGamePresentationEvents(
      previous,
      next,
      { type: 'end-turn' },
      PLAYER_A
    );

    expect(events).toContainEqual({
      type: 'draw-card',
      playerId: PLAYER_A,
      cardId: '7',
    });
    expect(events).toContainEqual({
      type: 'income-roll',
      playerId: PLAYER_B,
      roll: { die1: 7, die2: 4, rollId: 12 },
      incomeRank: 7,
    });
    expect(events.filter((event) => event.type === 'tax-token-lost')).toEqual([
      {
        type: 'tax-token-lost',
        playerId: PLAYER_A,
        suit: 'Moons',
        tokenIndex: 0,
      },
      {
        type: 'tax-token-lost',
        playerId: PLAYER_A,
        suit: 'Moons',
        tokenIndex: 1,
      },
    ]);

    const incomeEvents = events.filter(
      (event) => event.type === 'income-token-gained'
    );
    expect(incomeEvents).toEqual([
      {
        type: 'income-token-gained',
        playerId: PLAYER_B,
        suit: 'Suns',
        source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
      },
      {
        type: 'income-token-gained',
        playerId: PLAYER_B,
        suit: 'Knots',
        source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
      },
    ]);
  });

  it('derives final income-choice reveal gains only when all choices resolve', () => {
    const previous = makeGameState({
      phase: 'CollectIncome',
      activePlayerIndex: 1,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Knots'],
        },
        {
          playerId: PLAYER_B,
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
          suit: 'Knots',
        },
      ],
    });
    const next = makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 1,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Knots: 1 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Leaves: 1 }) }),
      ],
    });

    const events = deriveGamePresentationEvents(
      previous,
      next,
      {
        type: 'choose-income-suit',
        playerId: PLAYER_B,
        districtId: 'D2',
        cardId: '8',
        suit: 'Leaves',
      },
      PLAYER_B
    );

    expect(events).toContainEqual({
      type: 'income-choice-submitted',
      playerId: PLAYER_B,
      districtId: 'D2',
      cardId: '8',
      suit: 'Leaves',
    });
    expect(
      events.filter((event) => event.type === 'income-token-gained')
    ).toEqual([
      {
        type: 'income-token-gained',
        playerId: PLAYER_A,
        suit: 'Knots',
        source: { kind: 'income-choice', districtId: 'D1', cardId: '6' },
      },
      {
        type: 'income-token-gained',
        playerId: PLAYER_B,
        suit: 'Leaves',
        source: { kind: 'income-choice', districtId: 'D2', cardId: '8' },
      },
    ]);
  });

  it('keeps pending income choices as request events when turn-cycle income cannot resolve immediately', () => {
    const previous = makeGameState({
      activePlayerIndex: 0,
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: {
            developed: [],
            deed: {
              cardId: '6',
              progress: 1,
              tokens: { Moons: 1 },
            },
          },
        }),
      ],
    });
    const next = {
      ...makeGameState({
        phase: 'CollectIncome',
        activePlayerIndex: 0,
        districts: [...previous.districts],
        lastIncomeRoll: { die1: 2, die2: 2, rollId: 3 },
        pendingIncomeChoices: [
          {
            playerId: PLAYER_A,
            districtId: 'D1',
            cardId: '6',
            suits: ['Moons', 'Knots'],
          },
        ],
        incomeChoiceReturnPlayerId: PLAYER_A,
      }),
      lastTaxSuit: undefined,
    } satisfies GameState;

    const events = deriveGamePresentationEvents(
      previous,
      next,
      { type: 'end-turn' },
      PLAYER_A
    );

    expect(events).toContainEqual({
      type: 'income-choice-required',
      choices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Knots'],
        },
      ],
    });
  });

  it('creates stable source keys for semantic income-token sources', () => {
    expect(
      incomeTokenSourceKey({
        kind: 'income-choice',
        districtId: 'D2',
        cardId: '8',
      })
    ).toBe('income-choice:D2:8');
  });
});

function makeEndTurnStates(): { previous: GameState; next: GameState } {
  const previous = makeGameState({
    phase: 'ActionWindow',
    activePlayerIndex: 0,
    cardPlayedThisTurn: true,
    players: [
      makePlayer(PLAYER_A, {
        hand: ['6'],
        resources: makeResources({ Moons: 3 }),
      }),
      makePlayer(PLAYER_B, {
        hand: ['8'],
        resources: makeResources({ Suns: 1 }),
      }),
    ],
    districts: [
      makeDistrict('D1', ['Suns'], {
        [PLAYER_B]: { developed: ['21'] },
      }),
    ],
  });
  const next = {
    ...makeGameState({
      phase: 'ActionWindow',
      activePlayerIndex: 1,
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6', '7'],
          resources: makeResources({ Moons: 1 }),
        }),
        makePlayer(PLAYER_B, {
          hand: ['8'],
          resources: makeResources({ Suns: 2, Knots: 1 }),
        }),
      ],
      districts: [...previous.districts],
      lastIncomeRoll: { die1: 7, die2: 4, rollId: 12 },
    }),
    lastTaxSuit: 'Moons',
  } satisfies GameState;
  return { previous, next };
}
