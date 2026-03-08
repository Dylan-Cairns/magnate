import { describe, expect, it } from 'vitest';

import type { GameAction, GameState } from '../engine/types';
import {
  PLAYER_A,
  PLAYER_B,
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
} from '../engine/__tests__/fixtures';
import { deriveTurnCycleEvents } from './turnCycleEvents';

function withEndTurnIncome(
  previousState: GameState,
  nextState: GameState
) {
  const action: GameAction = { type: 'end-turn' };
  return deriveTurnCycleEvents(previousState, nextState, action);
}

describe('deriveTurnCycleEvents', () => {
  it('returns null for non end-turn actions', () => {
    const previous = makeGameState();
    const next = makeGameState({
      lastIncomeRoll: { die1: 6, die2: 3 },
    });

    const cycle = deriveTurnCycleEvents(previous, next, {
      type: 'trade',
      give: 'Moons',
      receive: 'Suns',
    });

    expect(cycle).toBeNull();
  });

  it('derives tax, deterministic income tokens, and highlight cards from properties/deeds', () => {
    const districts = [
      makeDistrict('D1', ['Moons'], {
        [PLAYER_A]: { developed: ['29'] },
      }),
      makeDistrict('D2', ['Suns'], {
        [PLAYER_B]: {
          developed: [],
          deed: { cardId: '28', progress: 2, tokens: { Leaves: 1, Knots: 1 } },
        },
      }),
      makeDistrict('D3', ['Waves'], {
        [PLAYER_B]: { developed: ['27'] },
      }),
      makeDistrict('D4', ['Leaves']),
      makeDistrict('D5', []),
    ];

    const previous = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Suns: 1 }),
        }),
        makePlayer(PLAYER_B, {
          resources: makeResources({ Moons: 2, Waves: 1 }),
        }),
      ] as const,
      districts,
    });

    const next = {
      ...makeGameState({
        turn: 2,
        phase: 'CollectIncome',
        activePlayerIndex: 0,
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 2, Suns: 2 }),
          }),
          makePlayer(PLAYER_B, {
            resources: makeResources({ Moons: 1, Waves: 2, Wyrms: 1 }),
          }),
        ] as const,
        districts,
        lastIncomeRoll: { die1: 1, die2: 9 },
        pendingIncomeChoices: [
          {
            playerId: PLAYER_B,
            districtId: 'D2',
            cardId: '28',
            suits: ['Leaves', 'Knots'],
          },
        ],
        incomeChoiceReturnPlayerId: PLAYER_B,
      }),
      lastTaxSuit: 'Moons' as const,
    };

    const cycle = withEndTurnIncome(previous, next);
    expect(cycle).not.toBeNull();
    if (!cycle) {
      return;
    }

    expect(cycle.cycleOwner).toBe(PLAYER_B);
    expect(cycle.incomeRank).toBe(9);
    expect(cycle.tax).toEqual({
      suit: 'Moons',
      lossesByPlayer: [
        { playerId: PLAYER_A, count: 2 },
        { playerId: PLAYER_B, count: 1 },
      ],
    });
    expect(cycle.incomeTokens).toEqual([
      {
        playerId: PLAYER_A,
        suit: 'Moons',
        source: { kind: 'district-card', cardId: '29', districtId: 'D1' },
      },
      {
        playerId: PLAYER_A,
        suit: 'Suns',
        source: { kind: 'district-card', cardId: '29', districtId: 'D1' },
      },
      {
        playerId: PLAYER_B,
        suit: 'Waves',
        source: { kind: 'district-card', cardId: '27', districtId: 'D3' },
      },
      {
        playerId: PLAYER_B,
        suit: 'Wyrms',
        source: { kind: 'district-card', cardId: '27', districtId: 'D3' },
      },
    ]);
    expect(cycle.incomeHighlights).toEqual([
      { playerId: PLAYER_A, districtId: 'D1', cardId: '29' },
      { playerId: PLAYER_B, districtId: 'D2', cardId: '28' },
      { playerId: PLAYER_B, districtId: 'D3', cardId: '27' },
    ]);
    expect(cycle.pendingChoices).toHaveLength(1);
  });

  it('uses crown cards for rank-10 income token events', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, { crowns: ['30', '31', '32'] }),
        makePlayer(PLAYER_B, { crowns: ['33', '34', '35'] }),
      ] as const,
    });
    const next = makeGameState({
      turn: 2,
      activePlayerIndex: 1,
      players: [
        makePlayer(PLAYER_A, { crowns: ['30', '31', '32'] }),
        makePlayer(PLAYER_B, { crowns: ['33', '34', '35'] }),
      ] as const,
      lastIncomeRoll: { die1: 10, die2: 4 },
    });

    const cycle = withEndTurnIncome(previous, next);
    expect(cycle).not.toBeNull();
    if (!cycle) {
      return;
    }

    expect(cycle.cycleOwner).toBe(PLAYER_B);
    expect(cycle.tax).toBeNull();
    expect(cycle.incomeHighlights).toEqual([]);
    expect(cycle.incomeTokens).toEqual([
      {
        playerId: PLAYER_A,
        suit: 'Knots',
        source: { kind: 'crown', cardId: '30' },
      },
      {
        playerId: PLAYER_A,
        suit: 'Leaves',
        source: { kind: 'crown', cardId: '31' },
      },
      {
        playerId: PLAYER_A,
        suit: 'Moons',
        source: { kind: 'crown', cardId: '32' },
      },
      {
        playerId: PLAYER_B,
        suit: 'Suns',
        source: { kind: 'crown', cardId: '33' },
      },
      {
        playerId: PLAYER_B,
        suit: 'Waves',
        source: { kind: 'crown', cardId: '34' },
      },
      {
        playerId: PLAYER_B,
        suit: 'Wyrms',
        source: { kind: 'crown', cardId: '35' },
      },
    ]);
  });
});
