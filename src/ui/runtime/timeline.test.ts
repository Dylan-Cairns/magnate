import { describe, expect, it } from 'vitest';

import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  CARD_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
} from '../animations/timing';
import {
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
} from '../../engine/__tests__/fixtures';
import type { GameState } from '../../engine/types';
import type { GameTransaction } from './types';
import { buildPresentationTimeline } from './timeline';

describe('buildPresentationTimeline', () => {
  it('delays turn-cycle reveal events until after draw-card presentation settles', () => {
    const timeline = buildPresentationTimeline({
      id: 'tx-delayed-income',
      previousState: stateForTimeline(),
      nextState: {
        ...stateForTimeline(),
        activePlayerIndex: 1,
        phase: 'ActionWindow',
        lastIncomeRoll: { die1: 7, die2: 4, rollId: 12 },
        lastTaxSuit: 'Moons',
      },
      action: { type: 'end-turn' },
      actingPlayerId: 'PlayerA',
      events: [
        {
          type: 'action-started',
          action: { type: 'end-turn' },
          actingPlayerId: 'PlayerA',
        },
        { type: 'draw-card', playerId: 'PlayerA', cardId: '7' },
        {
          type: 'income-roll',
          playerId: 'PlayerB',
          roll: { die1: 7, die2: 4, rollId: 12 },
          incomeRank: 7,
        },
        {
          type: 'tax-token-lost',
          playerId: 'PlayerA',
          suit: 'Moons',
          tokenIndex: 0,
        },
        {
          type: 'income-token-gained',
          playerId: 'PlayerB',
          suit: 'Suns',
          source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
        },
        { type: 'transaction-settled' },
      ],
    });

    const drawRevealMs =
      CARD_FLIGHT_DURATION_MS + ACTION_FLIGHT_COMMIT_BUFFER_MS;
    expect(timeline.events[0]).toEqual({
      atMs: 0,
      type: 'hold-previous-state',
    });
    expect(timeline.events).toContainEqual({
      atMs: drawRevealMs,
      type: 'reveal-drawn-card',
      event: { type: 'draw-card', playerId: 'PlayerA', cardId: '7' },
    });
    expect(timeline.events).toContainEqual({
      atMs: drawRevealMs,
      type: 'show-income-roll',
      event: {
        type: 'income-roll',
        playerId: 'PlayerB',
        roll: { die1: 7, die2: 4, rollId: 12 },
        incomeRank: 7,
      },
    });
  });

  it('uses turn-cycle plan timing for tax losses, income flights, and final commit', () => {
    const timeline = buildPresentationTimeline({
      id: 'tx-turn-cycle',
      previousState: stateForTimeline(),
      nextState: {
        ...stateForTimeline(),
        phase: 'ActionWindow',
        activePlayerIndex: 1,
        lastIncomeRoll: { die1: 7, die2: 4, rollId: 12 },
        lastTaxSuit: 'Moons',
      },
      action: { type: 'end-turn' },
      actingPlayerId: 'PlayerA',
      events: [
        {
          type: 'action-started',
          action: { type: 'end-turn' },
          actingPlayerId: 'PlayerA',
        },
        {
          type: 'income-roll',
          playerId: 'PlayerB',
          roll: { die1: 7, die2: 4, rollId: 12 },
          incomeRank: 7,
        },
        {
          type: 'tax-token-lost',
          playerId: 'PlayerA',
          suit: 'Moons',
          tokenIndex: 0,
        },
        {
          type: 'income-token-gained',
          playerId: 'PlayerB',
          suit: 'Suns',
          source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
        },
        { type: 'transaction-settled' },
      ],
    });

    expect(timeline.events).toContainEqual({
      atMs: 2550,
      type: 'apply-tax-token-loss',
      event: {
        type: 'tax-token-lost',
        playerId: 'PlayerA',
        suit: 'Moons',
        tokenIndex: 0,
      },
    });
    expect(timeline.events).toContainEqual({
      atMs: 4570,
      type: 'show-income-highlights',
      cardIds: ['21'],
      crowns: [],
    });
    expect(timeline.events).toContainEqual({
      atMs: 5530,
      type: 'clear-income-highlights',
    });
    expect(timeline.events).toContainEqual({
      atMs: 4570,
      type: 'launch-income-token-flight',
      event: {
        type: 'income-token-gained',
        playerId: 'PlayerB',
        suit: 'Suns',
        source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
      },
    });
    expect(timeline.events).toContainEqual({
      atMs: 4570 + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
      type: 'apply-income-token-gain',
      event: {
        type: 'income-token-gained',
        playerId: 'PlayerB',
        suit: 'Suns',
        source: { kind: 'district-card', districtId: 'D1', cardId: '21' },
      },
    });
    expect(timeline.events.at(-1)).toEqual({
      atMs: 5770,
      type: 'commit-view-to-next-state',
    });
    expect(timeline.durationMs).toBe(5770);
  });

  it('builds a final income-choice reveal timeline without a turn-cycle plan', () => {
    const transaction: GameTransaction = {
      id: 'tx-income-choice',
      previousState: stateForTimeline(),
      nextState: stateForTimeline(),
      action: {
        type: 'choose-income-suit',
        playerId: 'PlayerB',
        districtId: 'D2',
        cardId: '8',
        suit: 'Leaves',
      },
      actingPlayerId: 'PlayerB',
      events: [
        {
          type: 'action-started',
          action: {
            type: 'choose-income-suit',
            playerId: 'PlayerB',
            districtId: 'D2',
            cardId: '8',
            suit: 'Leaves',
          },
          actingPlayerId: 'PlayerB',
        },
        {
          type: 'income-choice-submitted',
          playerId: 'PlayerB',
          districtId: 'D2',
          cardId: '8',
          suit: 'Leaves',
        },
        {
          type: 'income-token-gained',
          playerId: 'PlayerA',
          suit: 'Knots',
          source: { kind: 'income-choice', districtId: 'D1', cardId: '6' },
        },
        {
          type: 'income-token-gained',
          playerId: 'PlayerB',
          suit: 'Leaves',
          source: { kind: 'income-choice', districtId: 'D2', cardId: '8' },
        },
        { type: 'transaction-settled' },
      ],
    };

    const timeline = buildPresentationTimeline(transaction);

    expect(timeline.events).toContainEqual({
      atMs: 0,
      type: 'show-income-highlights',
      cardIds: ['6', '8'],
      crowns: [],
    });
    expect(
      timeline.events.filter(
        (event) => event.type === 'launch-income-token-flight'
      )
    ).toMatchObject([
      { atMs: 0 },
      { atMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS },
    ]);
    expect(timeline.events.at(-1)).toEqual({
      atMs:
        TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS +
        TURN_CYCLE_INCOME_FLIGHT_DURATION_MS +
        ACTION_FLIGHT_COMMIT_BUFFER_MS,
      type: 'commit-view-to-next-state',
    });
  });
});

function stateForTimeline(): GameState {
  return makeGameState({
    seed: 'timeline-seed',
    phase: 'ActionWindow',
    activePlayerIndex: 0,
    cardPlayedThisTurn: true,
    deck: {
      draw: ['7', '8'],
      discard: [],
      reshuffles: 0,
    },
    players: [
      makePlayer('PlayerA', {
        hand: ['6'],
        crowns: ['30', '31', '32'],
        resources: makeResources({ Moons: 3 }),
      }),
      makePlayer('PlayerB', {
        hand: ['8'],
        crowns: ['33', '34', '35'],
        resources: makeResources({ Suns: 1 }),
      }),
    ],
    districts: [
      makeDistrict('D1', ['Suns'], {
        PlayerB: { developed: ['21'] },
      }),
    ],
  });
}
