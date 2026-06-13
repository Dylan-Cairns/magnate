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

  it('stages sold-card state without leaking the discard pile before commit', () => {
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
    const commit = step(sequence, 'commit-view-state');

    const staged = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: 0,
    });
    expect(staged.viewState.players[0].hand).toEqual([]);
    expect(resourceCount(staged.viewState, PLAYER_A, 'Moons')).toBe(1);
    expect(staged.viewState.deck.discard).toEqual([]);

    const committed = derivePresentationSnapshotFromSequence({
      transaction,
      sequence,
      elapsedMs: commit.startMs,
    });
    expect(committed.viewState.deck.discard).toEqual(['6']);
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

function resourceCount(
  state: GameState,
  playerId: PlayerId,
  suit: Suit
): number {
  return state.players.find((player) => player.id === playerId)?.resources[
    suit
  ] as number;
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
