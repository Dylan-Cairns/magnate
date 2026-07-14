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
      'tax-resolved',
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
      turn: 1,
      roll: { die1: 7, die2: 4, rollId: 12 },
      incomeRank: 7,
    });
    expect(events).toContainEqual({
      type: 'tax-resolved',
      suit: 'Moons',
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

  it('derives sold-card presentation events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: [],
          resources: makeResources({ Moons: 2, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
      deck: {
        draw: [],
        discard: ['6'],
        reshuffles: 0,
      },
    });

    expect(
      deriveGamePresentationEvents(
        previous,
        next,
        { type: 'sell-card', cardId: '6' },
        PLAYER_A
      )
    ).toContainEqual({
      type: 'card-sold',
      playerId: PLAYER_A,
      cardId: '6',
    });
    expect(
      deriveGamePresentationEvents(
        previous,
        next,
        { type: 'sell-card', cardId: '6' },
        PLAYER_A
      )
    ).toContainEqual({
      type: 'sell-resource-gained',
      playerId: PLAYER_A,
      cardId: '6',
      suit: 'Moons',
      tokenIndex: 0,
    });
  });

  it('derives buy-deed payment and card placement events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 2, Knots: 2 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: [],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: {
            developed: [],
            deed: { cardId: '6', progress: 0, tokens: {} },
          },
        }),
      ],
    });

    const events = deriveGamePresentationEvents(
      previous,
      next,
      { type: 'buy-deed', cardId: '6', districtId: 'D1' },
      PLAYER_A
    );

    expect(events).toContainEqual({
      type: 'resource-payment-started',
      playerId: PLAYER_A,
      reason: 'buy-deed',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Knots: 1 },
    });
    expect(events).toContainEqual({
      type: 'resource-payment-applied',
      playerId: PLAYER_A,
      reason: 'buy-deed',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Knots: 1 },
    });
    expect(events).toContainEqual({
      type: 'card-played-to-district',
      playerId: PLAYER_A,
      cardId: '6',
      districtId: 'D1',
      placement: 'deed',
    });
  });

  it('derives outright-development payment and developed placement events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 3, Knots: 3 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: [],
          resources: makeResources(),
        }),
        makePlayer(PLAYER_B),
      ],
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: { developed: ['6'] },
        }),
      ],
      cardPlayedThisTurn: true,
    });

    const events = deriveGamePresentationEvents(
      previous,
      next,
      {
        type: 'develop-outright',
        cardId: '6',
        districtId: 'D1',
        payment: { Moons: 3, Knots: 3 },
      },
      PLAYER_A
    );

    expect(events).toContainEqual({
      type: 'resource-payment-started',
      playerId: PLAYER_A,
      reason: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 3, Knots: 3 },
    });
    expect(events).toContainEqual({
      type: 'card-played-to-district',
      playerId: PLAYER_A,
      cardId: '6',
      districtId: 'D1',
      placement: 'developed',
    });
  });

  it('derives deed token, progress, and completion events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: {
            developed: [],
            deed: {
              cardId: '6',
              progress: 0,
              tokens: {},
            },
          },
        }),
      ],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources() }),
        makePlayer(PLAYER_B),
      ],
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: { developed: ['6'] },
        }),
      ],
    });

    const events = deriveGamePresentationEvents(
      previous,
      next,
      {
        type: 'develop-deed',
        cardId: '6',
        districtId: 'D1',
        tokens: { Moons: 1, Knots: 1 },
      },
      PLAYER_A
    );

    expect(events).toContainEqual({
      type: 'deed-token-paid',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '6',
      suit: 'Moons',
      tokenIndex: 0,
    });
    expect(events).toContainEqual({
      type: 'deed-token-paid',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '6',
      suit: 'Knots',
      tokenIndex: 0,
    });
    expect(events).toContainEqual({
      type: 'deed-progress-applied',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '6',
      previousProgress: 0,
      nextProgress: 2,
      targetProgress: 2,
      completed: true,
    });
    expect(events).toContainEqual({
      type: 'deed-completed',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '6',
    });
  });

  it('derives trade resource application events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Suns: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });

    expect(
      deriveGamePresentationEvents(
        previous,
        next,
        { type: 'trade', give: 'Moons', receive: 'Suns' },
        PLAYER_A
      )
    ).toContainEqual({
      type: 'trade-resources-applied',
      playerId: PLAYER_A,
      give: 'Moons',
      receive: 'Suns',
      giveCount: 3,
      receiveCount: 1,
    });
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
      returnPlayerId: PLAYER_A,
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
