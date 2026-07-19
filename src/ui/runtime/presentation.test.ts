import { describe, expect, it } from 'vitest';

import type { GameState, PlayerId, Suit } from '../../engine/types';
import { scoreLive } from '../../engine/scoring';
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

  it('applies each tax loss when its staggered token flight launches', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const applyTaxSteps = sequence.steps.filter(
      (candidate) => candidate.type === 'apply-tax-token-loss'
    );
    const firstTaxLoss = applyTaxSteps[0];
    const secondTaxLoss = applyTaxSteps[1];
    if (!firstTaxLoss || !secondTaxLoss) {
      throw new Error('Expected two staggered tax loss steps.');
    }

    const beforeFirstTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: firstTaxLoss.startMs - 1,
    });
    const afterFirstTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: firstTaxLoss.startMs,
    });
    const beforeSecondTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: secondTaxLoss.startMs - 1,
    });
    const afterSecondTaxLoss = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: secondTaxLoss.startMs,
    });

    expect(resourceCount(beforeFirstTaxLoss.viewState, PLAYER_A, 'Moons')).toBe(
      3
    );
    expect(resourceCount(afterFirstTaxLoss.viewState, PLAYER_A, 'Moons')).toBe(
      2
    );
    expect(
      resourceCount(beforeSecondTaxLoss.viewState, PLAYER_A, 'Moons')
    ).toBe(2);
    expect(resourceCount(afterSecondTaxLoss.viewState, PLAYER_A, 'Moons')).toBe(
      1
    );
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

  it('holds the visible active player through the draw and changes it when income rolling begins', () => {
    const { transaction } = makeEndTurnTransaction();
    const sequence = buildAnimationSequence(transaction);
    const incomeRoll = step(sequence, 'roll-income-dice');

    const duringDraw = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: incomeRoll.startMs - 1,
    });
    expect(duringDraw.viewState.activePlayerIndex).toBe(0);
    expect(duringDraw.overlays.activePlayerHighlightOverride).toBe(PLAYER_A);

    const duringIncomeRoll = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: incomeRoll.startMs,
    });
    expect(duringIncomeRoll.viewState.activePlayerIndex).toBe(1);
    expect(duringIncomeRoll.overlays.activePlayerHighlightOverride).toBeNull();
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

  it('keeps terminal state hidden until the presentation commit step', () => {
    const previous = makeGameState({
      players: [makePlayer(PLAYER_A, { hand: ['6'] }), makePlayer(PLAYER_B)],
    });
    const next = makeGameState({
      phase: 'GameOver',
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)],
      deck: { draw: [], discard: ['6'], reshuffles: 0 },
    });
    const transaction = buildGameTransaction({
      previousState: previous,
      action: { type: 'sell-card', cardId: '6' },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-terminal-sell',
      stepToDecision: () => next,
    });
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

    expect(beforeCommit.viewState.phase).toBe('ActionWindow');
    expect(atCommit.viewState.phase).toBe('GameOver');
    expect(atCommit.viewState).toBe(next);
  });

  it('stages final income-choice reveal and waits for the sequence gain step', () => {
    const transaction = makeIncomeChoiceTransaction();
    const sequence = buildAnimationSequence(transaction);
    const revealSubmission = step(
      sequence,
      'reveal-income-choice-submission'
    );
    const launchIncome = step(sequence, 'launch-income-token-flights');
    const applyIncome = step(sequence, 'apply-income-gains');

    const beforeLaunch = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: launchIncome.startMs - 1,
    });
    const atLaunch = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: launchIncome.startMs,
    });

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

    expect(revealSubmission.startMs).toBe(launchIncome.startMs);
    expect(beforeLaunch.viewState.submittedIncomeChoices).toEqual([
      {
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Knots',
      },
    ]);
    expect(atLaunch.viewState.submittedIncomeChoices).toEqual([
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
    expect(atLaunch.overlays.incomeHighlightCardIds).toEqual([]);
    expect(beforeLanding.overlays.incomeHighlightCardIds).toEqual([]);
    expect(resourceCount(beforeLanding.viewState, PLAYER_A, 'Knots')).toBe(0);
    expect(resourceCount(beforeLanding.viewState, PLAYER_B, 'Leaves')).toBe(0);
    expect(resourceCount(afterLanding.viewState, PLAYER_A, 'Knots')).toBe(1);
    expect(resourceCount(afterLanding.viewState, PLAYER_B, 'Leaves')).toBe(1);
  });

  it('stages sold-card state without leaking resources or discard before their steps', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6', '7'],
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
          // Deliberately not the legal sell result. The staging step should
          // remove only the sold card from the current view, not copy this hand.
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
    expect(staged.viewState.players[0].hand).toEqual(['7']);
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
    expect(committed.viewState.players[0].hand).toEqual([]);
  });

  it('places card-play cards before applying payment without leaking either step early', () => {
    const transaction = makeBuyDeedTransaction();
    const sequence = buildAnimationSequence(transaction);
    const applyPayment = step(sequence, 'apply-resource-payment');
    const placeCard = step(sequence, 'place-card-in-district');

    const beforePlacement = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: placeCard.startMs - 1,
    }).viewState;
    expect(resourceCount(beforePlacement, PLAYER_A, 'Moons')).toBe(2);
    expect(cardInHand(beforePlacement, PLAYER_A, '6')).toBe(true);
    expect(deedCardInDistrict(beforePlacement, PLAYER_A, 'D1')).toBeUndefined();

    const afterPlacement = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: placeCard.startMs,
    }).viewState;
    expect(resourceCount(afterPlacement, PLAYER_A, 'Moons')).toBe(2);
    expect(resourceCount(afterPlacement, PLAYER_A, 'Knots')).toBe(2);
    expect(cardInHand(afterPlacement, PLAYER_A, '6')).toBe(false);
    expect(deedCardInDistrict(afterPlacement, PLAYER_A, 'D1')).toBe('6');

    const beforePayment = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyPayment.startMs - 1,
    }).viewState;
    expect(resourceCount(beforePayment, PLAYER_A, 'Moons')).toBe(2);
    expect(resourceCount(beforePayment, PLAYER_A, 'Knots')).toBe(2);
    expect(cardInHand(beforePayment, PLAYER_A, '6')).toBe(false);
    expect(deedCardInDistrict(beforePayment, PLAYER_A, 'D1')).toBe('6');

    const afterPayment = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: applyPayment.startMs,
    }).viewState;
    expect(resourceCount(afterPayment, PLAYER_A, 'Moons')).toBe(1);
    expect(resourceCount(afterPayment, PLAYER_A, 'Knots')).toBe(1);
    expect(cardInHand(afterPayment, PLAYER_A, '6')).toBe(false);
    expect(deedCardInDistrict(afterPayment, PLAYER_A, 'D1')).toBe('6');
  });

  it('places developed cards from the placement event without copying unrelated next-state stack changes', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 2, Knots: 2 }),
        }),
        makePlayer(PLAYER_B),
      ],
      districts: [
        makeDistrict('D1', ['Moons'], {
          [PLAYER_A]: { developed: ['8'] },
        }),
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
          // Deliberately includes an unrelated extra card; the placement step
          // should not copy it before commit.
          [PLAYER_A]: { developed: ['8', '6', '7'] },
        }),
      ],
    });
    const transaction = buildGameTransaction({
      previousState: previous,
      action: {
        type: 'develop-outright',
        cardId: '6',
        districtId: 'D1',
        payment: { Moons: 1, Knots: 1 },
      },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-place-developed-card',
      stepToDecision: () => next,
    });
    const sequence = delayCommitStep(buildAnimationSequence(transaction));
    const placeCard = step(sequence, 'place-card-in-district');

    const beforePlacement = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: placeCard.startMs - 1,
    }).viewState;

    const placed = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: placeCard.startMs,
    }).viewState;

    expect(developedCards(placed, PLAYER_A, 'D1')).toEqual(['8', '6']);
    expect(scoreLive(placed).rankTotals[PLAYER_A]).toBeGreaterThan(
      scoreLive(beforePlacement).rankTotals[PLAYER_A]
    );
    expect(scoreLive(placed).rankTotals[PLAYER_A]).toBeLessThan(
      scoreLive(next).rankTotals[PLAYER_A]
    );
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

  it('reveals deed completion from the completion event without copying unrelated next-state stack changes', () => {
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
            developed: ['8'],
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
          // Deliberately includes an unrelated extra card; completion reveal
          // should only append the completed deed card.
          [PLAYER_A]: { developed: ['8', '6', '7'] },
        }),
      ],
    });
    const transaction = buildGameTransaction({
      previousState: previous,
      action: {
        type: 'develop-deed',
        cardId: '6',
        districtId: 'D1',
        tokens: { Moons: 1, Knots: 1 },
      },
      actingPlayerId: PLAYER_A,
      transactionId: 'tx-complete-deed-no-stack-leak',
      stepToDecision: () => next,
    });
    const sequence = delayCommitStep(buildAnimationSequence(transaction));
    const revealCompletion = step(sequence, 'reveal-deed-completion');

    const afterCompletion = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: revealCompletion.startMs,
    }).viewState;

    expect(deedCardInDistrict(afterCompletion, PLAYER_A, 'D1')).toBeUndefined();
    expect(developedCards(afterCompletion, PLAYER_A, 'D1')).toEqual(['8', '6']);
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
    state.players
      .find((player) => player.id === playerId)
      ?.hand.includes(cardId) ?? false
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

function delayCommitStep(
  sequence: AnimationSequence,
  delayMs = 1
): AnimationSequence {
  return {
    ...sequence,
    durationMs: sequence.durationMs + delayMs,
    commitMs: sequence.commitMs + delayMs,
    inputUnlockMs: sequence.inputUnlockMs + delayMs,
    steps: sequence.steps.map((entry) =>
      entry.type === 'commit-view-state'
        ? {
            ...entry,
            startMs: entry.startMs + delayMs,
            endMs: entry.endMs + delayMs,
          }
        : entry
    ),
  };
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
