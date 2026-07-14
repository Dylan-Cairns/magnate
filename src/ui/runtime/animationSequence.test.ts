import { describe, expect, it } from 'vitest';

import {
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../../engine/__tests__/fixtures';
import type { GameState } from '../../engine/types';
import { buildGameTransaction } from './transactions';
import {
  buildAnimationSequence,
  DEFAULT_ANIMATION_DURATIONS,
  type AnimationSequence,
} from './animationSequence';

describe('buildAnimationSequence', () => {
  it('builds a single ordered turn-cycle sequence where income flights wait for dice roll and pulse', () => {
    const transaction = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'draw-card-flight',
      'roll-income-dice',
      'pulse-income-die',
      'roll-tax-die',
      'pulse-tax-die',
      'hold-before-tax-flights',
      'launch-tax-token-flights',
      'apply-tax-losses',
      'stage-gap',
      'hold-before-income-flights',
      'highlight-income-sources',
      'launch-income-token-flights',
      'apply-income-gains',
      'post-income-hold',
      'commit-view-state',
    ]);

    const draw = step(sequence, 'draw-card-flight');
    expect(draw.startMs).toBe(0);
    expect(draw.durationMs).toBe(
      DEFAULT_ANIMATION_DURATIONS.cardFlightMs +
        DEFAULT_ANIMATION_DURATIONS.commitBufferMs
    );

    const incomeRoll = step(sequence, 'roll-income-dice');
    const incomePulse = step(sequence, 'pulse-income-die');
    const incomeFlights = step(sequence, 'launch-income-token-flights');
    expect(incomePulse.startMs).toBe(incomeRoll.endMs);
    expect(incomeFlights.startMs).toBeGreaterThanOrEqual(incomePulse.endMs);
  });

  it('uses the step duration as the single source for staggered income flight length', () => {
    const sequence = buildAnimationSequence(makeEndTurnTransaction());
    const incomeFlights = step(sequence, 'launch-income-token-flights');

    expect(incomeFlights.durationMs).toBe(
      DEFAULT_ANIMATION_DURATIONS.incomeFlightMs +
        DEFAULT_ANIMATION_DURATIONS.incomeFlightStaggerMs
    );
    expect(step(sequence, 'apply-income-gains').startMs).toBe(
      incomeFlights.endMs
    );
    expect(sequence.commitMs).toBe(step(sequence, 'commit-view-state').startMs);
    expect(sequence.inputUnlockMs).toBe(
      step(sequence, 'commit-view-state').startMs
    );
  });

  it('builds final income-choice reveal as submission, hold, income flights, gains, and commit', () => {
    const sequence = buildAnimationSequence(makeIncomeChoiceTransaction());

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'reveal-income-choice-submission',
      'hold-before-income-flights',
      'highlight-income-sources',
      'launch-income-token-flights',
      'apply-income-gains',
      'post-income-hold',
      'commit-view-state',
    ]);
    expect(step(sequence, 'launch-income-token-flights').startMs).toBe(
      DEFAULT_ANIMATION_DURATIONS.incomePreFlightHoldMs
    );
  });

  it('sequences card placement before card play payment', () => {
    const sequence = buildAnimationSequence(makeBuyDeedTransaction());

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'launch-card-to-district-flight',
      'place-card-in-district',
      'launch-payment-token-flights',
      'apply-resource-payment',
      'commit-view-state',
    ]);

    const payment = step(sequence, 'launch-payment-token-flights');
    const applyPayment = step(sequence, 'apply-resource-payment');
    const cardFlight = step(sequence, 'launch-card-to-district-flight');
    const placement = step(sequence, 'place-card-in-district');
    expect(payment.durationMs).toBe(
      DEFAULT_ANIMATION_DURATIONS.actionResourceFlightMs +
        DEFAULT_ANIMATION_DURATIONS.actionResourceFlightStaggerMs
    );
    expect(cardFlight.startMs).toBe(0);
    expect(placement.startMs).toBe(cardFlight.endMs);
    expect(payment.startMs).toBe(placement.endMs);
    expect(applyPayment.startMs).toBe(payment.endMs);
  });

  it('sequences deed progress and completion after deed-token flights', () => {
    const sequence = buildAnimationSequence(makeCompleteDeedTransaction());

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'apply-resource-payment',
      'launch-deed-token-flights',
      'apply-deed-tokens',
      'apply-deed-progress',
      'reveal-deed-completion',
      'commit-view-state',
    ]);

    const payment = step(sequence, 'apply-resource-payment');
    const tokenFlights = step(sequence, 'launch-deed-token-flights');
    const applyTokens = step(sequence, 'apply-deed-tokens');
    const progress = step(sequence, 'apply-deed-progress');
    const completion = step(sequence, 'reveal-deed-completion');
    expect(payment.startMs).toBe(0);
    expect(tokenFlights.startMs).toBe(payment.endMs);
    expect(applyTokens.startMs).toBe(tokenFlights.endMs);
    expect(progress.startMs).toBe(applyTokens.endMs);
    expect(completion.startMs).toBe(progress.endMs);
    expect(progress.event.completed).toBe(true);
  });

  it('adds explicit sell gain and trade resource steps', () => {
    const sellSequence = buildAnimationSequence(makeSellCardTransaction());
    expect(stepTypes(sellSequence)).toEqual([
      'hold-previous-state',
      'stage-sold-card',
      'apply-sell-resource-gains',
      'commit-view-state',
    ]);
    expect(step(sellSequence, 'apply-sell-resource-gains').startMs).toBe(
      step(sellSequence, 'stage-sold-card').endMs
    );

    const tradeSequence = buildAnimationSequence(makeTradeTransaction());
    expect(stepTypes(tradeSequence)).toEqual([
      'hold-previous-state',
      'apply-trade-resources',
      'commit-view-state',
    ]);
  });

  it('keeps no-loss tax animation on the turn-cycle transaction that resolved it', () => {
    const sequence = buildAnimationSequence(makeNoLossTaxEndTurnTransaction());

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'draw-card-flight',
      'roll-income-dice',
      'pulse-income-die',
      'roll-tax-die',
      'pulse-tax-die',
      'commit-view-state',
    ]);
  });

  it('does not animate a stale previous tax suit on a later sell-card transaction', () => {
    const sequence = buildAnimationSequence(
      makeSellCardTransactionWithStickyLastTaxSuit()
    );

    expect(stepTypes(sequence)).toEqual([
      'hold-previous-state',
      'stage-sold-card',
      'apply-sell-resource-gains',
      'commit-view-state',
    ]);
    expect(sequence.commitMs).toBe(
      step(sequence, 'apply-sell-resource-gains').endMs
    );
  });
});

function makeEndTurnTransaction() {
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
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'end-turn' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-end-turn',
    stepToDecision: () => next,
  });
}

function makeIncomeChoiceTransaction() {
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

function makeBuyDeedTransaction() {
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

function makeCompleteDeedTransaction() {
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

function makeSellCardTransaction() {
  const previous = makeGameState({
    players: [
      makePlayer(PLAYER_A, {
        hand: ['6'],
        resources: makeResources(),
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
    deck: {
      draw: [],
      discard: ['6'],
      reshuffles: 0,
    },
  });
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'sell-card', cardId: '6' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-sell-card',
    stepToDecision: () => next,
  });
}

function makeSellCardTransactionWithStickyLastTaxSuit() {
  const previous = {
    ...makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources(),
        }),
        makePlayer(PLAYER_B),
      ],
    }),
    lastTaxSuit: 'Moons',
  } satisfies GameState;
  const next = {
    ...makeGameState({
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
    }),
    lastTaxSuit: 'Moons',
  } satisfies GameState;
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'sell-card', cardId: '6' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-sell-card-after-tax',
    stepToDecision: () => next,
  });
}

function makeNoLossTaxEndTurnTransaction() {
  const previous = makeGameState({
    phase: 'ActionWindow',
    activePlayerIndex: 0,
    cardPlayedThisTurn: true,
    players: [
      makePlayer(PLAYER_A, {
        hand: ['6'],
        resources: makeResources({ Moons: 1 }),
      }),
      makePlayer(PLAYER_B, {
        hand: ['8'],
        resources: makeResources({ Moons: 1 }),
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
          resources: makeResources({ Moons: 1 }),
        }),
      ],
      lastIncomeRoll: { die1: 1, die2: 7, rollId: 12 },
    }),
    lastTaxSuit: 'Moons',
  } satisfies GameState;
  return buildGameTransaction({
    previousState: previous,
    action: { type: 'end-turn' },
    actingPlayerId: PLAYER_A,
    transactionId: 'tx-no-loss-tax-end-turn',
    stepToDecision: () => next,
  });
}

function makeTradeTransaction() {
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

function stepTypes(sequence: AnimationSequence): string[] {
  return sequence.steps.map((entry) => entry.type);
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
