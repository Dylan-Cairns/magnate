import { describe, expect, it } from 'vitest';

import type { GameState, PlayerId, Suit } from '../../engine/types';
import {
  PLAYER_A,
  PLAYER_B,
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
} from '../../engine/__tests__/fixtures';
import {
  buildAnimationSequence,
  type AnimationSequence,
} from './animationSequence';
import { buildGameTransaction } from './transactions';
import type { GameTransaction } from './types';
import { derivePresentationSnapshotFromSequence } from './presentation';

describe('derivePresentationSnapshotFromSequence', () => {
  it('keeps income resources hidden until the sequence reaches the income gain step', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const incomeFlights = step(sequence, 'launch-income-token-flights');
    const applyIncome = step(sequence, 'apply-income-gains');

    const duringIncomeFlights = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: incomeFlights.startMs,
    });
    expect(duringIncomeFlights.viewState.lastIncomeRoll).toEqual({
      die1: 7,
      die2: 4,
      rollId: 12,
    });
    expect(duringIncomeFlights.overlays.incomeHighlightCardIds).toEqual(['21']);
    expect(resourceCount(duringIncomeFlights.viewState, PLAYER_B, 'Suns')).toBe(
      1
    );
    expect(
      resourceCount(duringIncomeFlights.viewState, PLAYER_B, 'Knots')
    ).toBe(0);

    const beforeIncomeGain = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyIncome.startMs - 1,
    });
    expect(resourceCount(beforeIncomeGain.viewState, PLAYER_B, 'Suns')).toBe(1);
    expect(resourceCount(beforeIncomeGain.viewState, PLAYER_B, 'Knots')).toBe(
      0
    );

    const afterIncomeGain = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyIncome.startMs,
    });
    expect(resourceCount(afterIncomeGain.viewState, PLAYER_B, 'Suns')).toBe(2);
    expect(resourceCount(afterIncomeGain.viewState, PLAYER_B, 'Knots')).toBe(1);
  });

  it('applies tax losses only at the sequence tax-loss step', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const applyTax = step(sequence, 'apply-tax-losses');

    const beforeTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyTax.startMs - 1,
    });
    const afterTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyTax.startMs,
    });

    expect(resourceCount(beforeTaxLoss.viewState, PLAYER_A, 'Moons')).toBe(3);
    expect(resourceCount(afterTaxLoss.viewState, PLAYER_A, 'Moons')).toBe(1);
  });

  it('derives dice visual state from sequence boundaries without leaking tax early', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const incomeRoll = step(sequence, 'roll-income-dice');
    const incomePulse = step(sequence, 'pulse-income-die');
    const taxRoll = step(sequence, 'roll-tax-die');
    const taxPulse = step(sequence, 'pulse-tax-die');

    const duringIncomeRoll = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: incomeRoll.startMs,
    });
    expect(duringIncomeRoll.overlays.dice).toEqual({
      incomeRoll: { die1: 7, die2: 4, rollId: 12 },
      taxSuit: undefined,
      incomePhase: 'rolling',
      taxPhase: 'hidden',
    });
    expect(duringIncomeRoll.viewState.lastTaxSuit).toBeUndefined();

    const duringIncomePulse = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: incomePulse.startMs,
    });
    expect(duringIncomePulse.overlays.dice).toMatchObject({
      incomePhase: 'pulsing',
      taxSuit: undefined,
      taxPhase: 'hidden',
    });
    expect(duringIncomePulse.viewState.lastTaxSuit).toBeUndefined();

    const duringTaxRoll = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: taxRoll.startMs,
    });
    expect(duringTaxRoll.overlays.dice).toMatchObject({
      incomePhase: 'settled',
      taxSuit: 'Moons',
      taxPhase: 'rolling',
    });
    expect(duringTaxRoll.viewState.lastTaxSuit).toBe('Moons');

    const duringTaxPulse = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: taxPulse.startMs,
    });
    expect(duringTaxPulse.overlays.dice).toMatchObject({
      incomePhase: 'settled',
      taxSuit: 'Moons',
      taxPhase: 'pulsing',
    });
  });

  it('commits to nextState at the sequence commit step', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const commit = step(sequence, 'commit-view-state');

    const beforeCommit = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: commit.startMs - 1,
    });
    const atCommit = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: commit.startMs,
    });

    expect(beforeCommit.viewState).not.toBe(transaction.nextState);
    expect(atCommit.viewState).toBe(transaction.nextState);
    expect(atCommit.overlays.incomeHighlightCardIds).toEqual([]);
  });

  it('stages final income-choice reveal and waits for the sequence gain step', () => {
    const transaction = makeIncomeChoiceTransaction();
    const sequence = buildAnimationSequence(transaction);
    const applyIncome = step(sequence, 'apply-income-gains');

    const beforeLanding = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyIncome.startMs - 1,
    });
    const afterLanding = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyIncome.startMs,
    });

    expect(beforeLanding.viewState.submittedIncomeChoices).toEqual([
      {
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Knots',
      },
      {
        playerId: PLAYER_B,
        districtId: 'D2',
        cardId: '8',
        suit: 'Leaves',
      },
    ]);
    expect(resourceCount(beforeLanding.viewState, PLAYER_A, 'Knots')).toBe(0);
    expect(resourceCount(beforeLanding.viewState, PLAYER_B, 'Leaves')).toBe(0);
    expect(resourceCount(afterLanding.viewState, PLAYER_A, 'Knots')).toBe(1);
    expect(resourceCount(afterLanding.viewState, PLAYER_B, 'Leaves')).toBe(1);
  });

  it('stages sold-card state without leaking resources or discard before their steps', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources(),
        }),
        makePlayer(PLAYER_B),
      ],
      deck: {
        draw: [],
        discard: [],
        reshuffles: 0,
      },
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: [],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ],
      deck: {
        draw: [],
        discard: ['6'],
        reshuffles: 0,
      },
    });
    const transaction = buildGameTransaction({
      previousState: previous,
      action: { type: 'sell-card', cardId: '6' },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-sell-card',
      stepToDecision: () => next,
    });
    const sequence = buildAnimationSequence(transaction);
    const sellGains = step(sequence, 'apply-sell-resource-gains');
    const commit = step(sequence, 'commit-view-state');

    const staged = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: 0,
    });
    expect(staged.viewState.players[0].hand).toEqual([]);
    expect(resourceCount(staged.viewState, PLAYER_A, 'Moons')).toBe(0);
    expect(staged.viewState.deck.discard).toEqual([]);

    const afterGain = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: sellGains.startMs,
    });
    expect(resourceCount(afterGain.viewState, PLAYER_A, 'Moons')).toBe(1);
    expect(resourceCount(afterGain.viewState, PLAYER_A, 'Knots')).toBe(1);
    expect(afterGain.viewState.deck.discard).toEqual([]);

    const committed = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: commit.startMs,
    });
    expect(committed.viewState.deck.discard).toEqual(['6']);
  });

  it('applies card-play payment before card placement and does not leak district placement early', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const applyPayment = step(sequence, 'apply-resource-payment');
    const placeCard = step(sequence, 'place-card-in-district');

    const beforePayment = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyPayment.startMs - 1,
    }).viewState;
    expect(resourceCount(beforePayment, PLAYER_A, 'Moons')).toBe(2);
    expect(cardInHand(beforePayment, PLAYER_A, '6')).toBe(true);
    expect(deedCardInDistrict(beforePayment, PLAYER_A, 'D1')).toBeUndefined();

    const afterPayment = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyPayment.startMs,
    }).viewState;
    expect(resourceCount(afterPayment, PLAYER_A, 'Moons')).toBe(1);
    expect(resourceCount(afterPayment, PLAYER_A, 'Knots')).toBe(1);
    expect(cardInHand(afterPayment, PLAYER_A, '6')).toBe(true);
    expect(deedCardInDistrict(afterPayment, PLAYER_A, 'D1')).toBeUndefined();

    const afterPlacement = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: placeCard.startMs,
    }).viewState;
    expect(cardInHand(afterPlacement, PLAYER_A, '6')).toBe(false);
    expect(deedCardInDistrict(afterPlacement, PLAYER_A, 'D1')).toBe('6');
  });

  it('applies deed payment, token landing, progress, then completion reveal', () => {
    const transaction = makeCompleteDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const launchTokens = step(sequence, 'launch-deed-token-flights');
    const applyTokens = step(sequence, 'apply-deed-tokens');
    const applyProgress = step(sequence, 'apply-deed-progress');
    const revealCompletion = step(sequence, 'reveal-deed-completion');

    const initial = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: 0,
    }).viewState;
    expect(resourceCount(initial, PLAYER_A, 'Moons')).toBe(0);
    expect(resourceCount(initial, PLAYER_A, 'Knots')).toBe(0);
    expect(deedProgress(initial, PLAYER_A, 'D1')).toBe(0);
    expect(deedTokens(initial, PLAYER_A, 'D1')).toEqual({});
    expect(developedCards(initial, PLAYER_A, 'D1')).toEqual([]);

    const duringTokenFlight = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: launchTokens.startMs,
    }).viewState;
    expect(resourceCount(duringTokenFlight, PLAYER_A, 'Moons')).toBe(0);
    expect(resourceCount(duringTokenFlight, PLAYER_A, 'Knots')).toBe(0);
    expect(deedProgress(duringTokenFlight, PLAYER_A, 'D1')).toBe(0);
    expect(deedTokens(duringTokenFlight, PLAYER_A, 'D1')).toEqual({});

    const afterTokenLanding = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyTokens.startMs,
    }).viewState;
    expect(deedProgress(afterTokenLanding, PLAYER_A, 'D1')).toBe(0);
    expect(deedTokens(afterTokenLanding, PLAYER_A, 'D1')).toEqual({
      Moons: 1,
      Knots: 1,
    });
    expect(developedCards(afterTokenLanding, PLAYER_A, 'D1')).toEqual([]);

    const afterProgress = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyProgress.startMs,
    }).viewState;
    expect(deedProgress(afterProgress, PLAYER_A, 'D1')).toBe(2);
    expect(deedTokens(afterProgress, PLAYER_A, 'D1')).toEqual({
      Moons: 1,
      Knots: 1,
    });
    expect(developedCards(afterProgress, PLAYER_A, 'D1')).toEqual([]);

    const afterCompletion = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: revealCompletion.startMs,
    }).viewState;
    expect(deedCardInDistrict(afterCompletion, PLAYER_A, 'D1')).toBeUndefined();
    expect(developedCards(afterCompletion, PLAYER_A, 'D1')).toEqual(['6']);
  });

  it('applies trade resources at the trade sequence step', () => {
    const transaction = makeTradeTransaction();
    const sequence = buildAnimationSequence(transaction);
    const tradeStep = step(sequence, 'apply-trade-resources');

    const afterTrade = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: tradeStep.startMs,
    }).viewState;
    expect(resourceCount(afterTrade, PLAYER_A, 'Moons')).toBe(0);
    expect(resourceCount(afterTrade, PLAYER_A, 'Suns')).toBe(1);
  });
});

function makeEndTurnTransaction(): {
  transaction: GameTransaction;
} {
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
  const transaction = buildGameTransaction({
    previousState: previous,
    action: { type: 'end-turn' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-end-turn',
    stepToDecision: () => next,
  });
  return {
    transaction,
  };
}

function makeIncomeChoiceTransaction(): GameTransaction {
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
  return buildGameTransaction({
    previousState: previous,
    action: {
      type: 'choose-income-suit',
      playerId: PLAYER_B,
      districtId: 'D2',
      cardId: '8',
      suit: 'Leaves',
    },
    actingPlayerId: PLAYER_B,
    transactionId: 'tx-income-choice',
    stepToDecision: () => next,
  });
}

function makeBuyDeedTransaction(): GameTransaction {
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
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'buy-deed', cardId: '6', districtId: 'D1' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-buy-deed',
    stepToDecision: () => next,
  });
}

function makeCompleteDeedTransaction(): GameTransaction {
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
          deed: { cardId: '6', progress: 0, tokens: {} },
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
  return buildGameTransaction({
    previousState: previous,
    action: {
      type: 'develop-deed',
      cardId: '6',
      districtId: 'D1',
      tokens: { Moons: 1, Knots: 1 },
    },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-develop-deed',
    stepToDecision: () => next,
  });
}

function makeTradeTransaction(): GameTransaction {
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
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'trade', give: 'Moons', receive: 'Suns' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-trade',
    stepToDecision: () => next,
  });
}

function resourceCount(
  state: GameState,
  playerId: PlayerId,
  suit: Suit
): number {
  return state.players.find((player) => player.id === playerId)?.resources[
    suit
  ] as number;
}

function cardInHand(
  state: GameState,
  playerId: PlayerId,
  cardId: string
): boolean {
  return (
    state.players.find((player) => player.id === playerId)?.hand.includes(
      cardId
    ) ?? false
  );
}

function stackFor(state: GameState, playerId: PlayerId, districtId: string) {
  return state.districts.find((district) => district.id === districtId)?.stacks[
    playerId
  ];
}

function deedCardInDistrict(
  state: GameState,
  playerId: PlayerId,
  districtId: string
): string | undefined {
  return stackFor(state, playerId, districtId)?.deed?.cardId;
}

function deedProgress(
  state: GameState,
  playerId: PlayerId,
  districtId: string
): number | undefined {
  return stackFor(state, playerId, districtId)?.deed?.progress;
}

function deedTokens(
  state: GameState,
  playerId: PlayerId,
  districtId: string
): Partial<Record<Suit, number>> | undefined {
  return stackFor(state, playerId, districtId)?.deed?.tokens;
}

function developedCards(
  state: GameState,
  playerId: PlayerId,
  districtId: string
): readonly string[] {
  return stackFor(state, playerId, districtId)?.developed ?? [];
}

function step<TType extends AnimationSequence['steps'][number]['type']>(
  sequence: AnimationSequence,
  type: TType
): Extract<AnimationSequence['steps'][number], { type: TType }> {
  const found = sequence.steps.find((entry) => entry.type === type);
  if (!found) {
    throw new Error(`Missing step ${type}`);
  }
  return found as Extract<AnimationSequence['steps'][number], { type: TType }>;
}
