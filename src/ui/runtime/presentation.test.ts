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
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  CARD_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
} from '../animations/timing';
import { buildPresentationTimeline } from './timeline';
import { buildGameTransaction } from './transactions';
import type { GameTransaction } from './types';
import { derivePresentationSnapshot } from './presentation';

describe('derivePresentationSnapshot', () => {
  it('holds previous resources before delayed turn-cycle reveal even after canonical nextState has income', () => {
    const { transaction, timeline } = makeEndTurnTransaction();
    const snapshot = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: 0,
    });

    expect(resourceCount(snapshot.viewState, PLAYER_A, 'Moons')).toBe(3);
    expect(resourceCount(snapshot.viewState, PLAYER_B, 'Suns')).toBe(1);
    expect(snapshot.viewState.lastIncomeRoll).toBeUndefined();
    expect(snapshot.viewState.activePlayerIndex).toBe(0);
    expect(snapshot.viewState.players[0].hand).toEqual(['6']);
    expect(snapshot.overlays.activePlayerHighlightOverride).toBe(PLAYER_A);
  });

  it('reveals the drawn card, active player, and income roll without applying future resources', () => {
    const { transaction, timeline } = makeEndTurnTransaction();
    const drawRevealMs =
      CARD_FLIGHT_DURATION_MS + ACTION_FLIGHT_COMMIT_BUFFER_MS;
    const snapshot = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: drawRevealMs,
    });

    expect(snapshot.viewState.players[0].hand).toEqual(['6', '7']);
    expect(snapshot.viewState.activePlayerIndex).toBe(1);
    expect(snapshot.viewState.lastIncomeRoll).toEqual({
      die1: 7,
      die2: 4,
      rollId: 12,
    });
    expect(snapshot.viewState.lastTaxSuit).toBe('Moons');
    expect(resourceCount(snapshot.viewState, PLAYER_A, 'Moons')).toBe(3);
    expect(resourceCount(snapshot.viewState, PLAYER_B, 'Suns')).toBe(1);
    expect(snapshot.overlays.activePlayerHighlightOverride).toBeNull();
  });

  it('applies tax and income only at their presentation timeline checkpoints', () => {
    const { transaction, timeline } = makeEndTurnTransaction();

    expect(
      resourceCount(
        derivePresentationSnapshot({
          transaction,
          timeline,
          elapsedMs: 2849,
        }).viewState,
        PLAYER_A,
        'Moons'
      )
    ).toBe(3);
    expect(
      resourceCount(
        derivePresentationSnapshot({
          transaction,
          timeline,
          elapsedMs: 2850,
        }).viewState,
        PLAYER_A,
        'Moons'
      )
    ).toBe(2);
    expect(
      resourceCount(
        derivePresentationSnapshot({
          transaction,
          timeline,
          elapsedMs: 3350,
        }).viewState,
        PLAYER_A,
        'Moons'
      )
    ).toBe(1);

    const afterIncomeLaunch = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: 4870,
    });
    expect(resourceCount(afterIncomeLaunch.viewState, PLAYER_B, 'Suns')).toBe(
      1
    );
    expect(afterIncomeLaunch.overlays.incomeHighlightCardIds).toEqual(['21']);

    expect(
      resourceCount(
        derivePresentationSnapshot({
          transaction,
          timeline,
          elapsedMs: 4870 + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
        }).viewState,
        PLAYER_B,
        'Suns'
      )
    ).toBe(2);
  });

  it('commits to nextState at the final presentation checkpoint', () => {
    const { transaction, timeline } = makeEndTurnTransaction();
    const snapshot = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: timeline.durationMs,
    });

    expect(snapshot.viewState).toBe(transaction.nextState);
    expect(snapshot.overlays.incomeHighlightCardIds).toEqual([]);
  });

  it('stages final income-choice reveal without applying resources before token landing', () => {
    const transaction = makeIncomeChoiceTransaction();
    const timeline = buildPresentationTimeline(transaction);
    const beforeLanding = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS - 1,
    });
    const afterFirstLanding = derivePresentationSnapshot({
      transaction,
      timeline,
      elapsedMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
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
    expect(resourceCount(afterFirstLanding.viewState, PLAYER_A, 'Knots')).toBe(
      1
    );
    expect(resourceCount(afterFirstLanding.viewState, PLAYER_B, 'Leaves')).toBe(
      0
    );
  });
});

function makeEndTurnTransaction(): {
  transaction: GameTransaction;
  timeline: ReturnType<typeof buildPresentationTimeline>;
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
    timeline: buildPresentationTimeline(transaction),
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
