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
import {
  buildAnimationSequence,
  type AnimationSequence,
} from './animationSequence';
import { deriveAnimationVisualCommands } from './animationVisualCommands';
import { buildGameTransaction } from './transactions';

describe('deriveAnimationVisualCommands', () => {
  it('derives tax pulses and resource flights from sequence step boundaries', () => {
    const sequence = buildAnimationSequence(makeEndTurnTransaction());
    const commands = deriveAnimationVisualCommands(sequence);
    const drawFlight = commands.find(
      (command) => command.type === 'launch-draw-card-flight'
    );
    const taxPulse = commands.find(
      (command) => command.type === 'pulse-tax-resources'
    );
    const taxFlights = commands.find(
      (command) => command.type === 'launch-tax-token-flights'
    );

    expect(drawFlight).toEqual({
      type: 'launch-draw-card-flight',
      atMs: step(sequence, 'draw-card-flight').startMs,
      playerId: PLAYER_A,
      cardId: '7',
    });
    expect(taxPulse).toEqual({
      type: 'pulse-tax-resources',
      startMs: step(sequence, 'pulse-tax-die').startMs,
      endMs: step(sequence, 'pulse-tax-die').endMs,
      targets: [{ playerId: PLAYER_A, suit: 'Moons' }],
    });
    expect(taxFlights).toMatchObject({
      type: 'launch-tax-token-flights',
      atMs: step(sequence, 'launch-tax-token-flights').startMs,
      durationMs: step(sequence, 'launch-tax-token-flights').durationMs,
    });
  });

  it('schedules income flights after the income dice roll and pulse complete', () => {
    const sequence = buildAnimationSequence(makeEndTurnTransaction());
    const incomeFlights = deriveAnimationVisualCommands(sequence).find(
      (command) => command.type === 'launch-income-token-flights'
    );

    expect(incomeFlights).toMatchObject({
      type: 'launch-income-token-flights',
      atMs: step(sequence, 'launch-income-token-flights').startMs,
      durationMs: step(sequence, 'launch-income-token-flights').durationMs,
    });
    expect(incomeFlights?.atMs).toBeGreaterThanOrEqual(
      step(sequence, 'pulse-income-die').endMs
    );
  });

  it('preserves final income-choice sources for sequence-launched flights', () => {
    const sequence = buildAnimationSequence(makeIncomeChoiceTransaction());
    const incomeFlights = deriveAnimationVisualCommands(sequence).find(
      (command) => command.type === 'launch-income-token-flights'
    );

    expect(incomeFlights).toMatchObject({
      type: 'launch-income-token-flights',
      atMs: step(sequence, 'launch-income-token-flights').startMs,
      gains: [
        {
          playerId: PLAYER_A,
          suit: 'Knots',
          source: {
            kind: 'income-choice',
            districtId: 'D1',
            cardId: '6',
          },
        },
        {
          playerId: PLAYER_B,
          suit: 'Leaves',
          source: {
            kind: 'income-choice',
            districtId: 'D2',
            cardId: '8',
          },
        },
      ],
    });
  });

  it('derives action card and token flights from action sequence steps', () => {
    const buySequence = buildAnimationSequence(makeBuyDeedTransaction());
    const buyCommands = deriveAnimationVisualCommands(buySequence);

    expect(buyCommands).toContainEqual({
      type: 'launch-payment-token-flights',
      atMs: step(buySequence, 'launch-payment-token-flights').startMs,
      durationMs: step(buySequence, 'launch-payment-token-flights').durationMs,
      event: step(buySequence, 'launch-payment-token-flights').event,
    });
    expect(buyCommands).toContainEqual({
      type: 'launch-card-to-district-flight',
      atMs: step(buySequence, 'launch-card-to-district-flight').startMs,
      event: step(buySequence, 'launch-card-to-district-flight').event,
    });

    const deedSequence = buildAnimationSequence(makeCompleteDeedTransaction());
    expect(deriveAnimationVisualCommands(deedSequence)).toContainEqual({
      type: 'launch-deed-token-flights',
      atMs: step(deedSequence, 'launch-deed-token-flights').startMs,
      durationMs: step(deedSequence, 'launch-deed-token-flights').durationMs,
      tokens: step(deedSequence, 'launch-deed-token-flights').tokens,
    });
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
