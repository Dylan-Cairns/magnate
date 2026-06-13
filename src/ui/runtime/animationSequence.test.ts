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
