import { describe, expect, it } from 'vitest';

import type { GameAction, GameLogEntry, GameState } from '../engine/types';
import {
  PLAYER_A,
  PLAYER_B,
  makeGameState,
  makePlayer,
  makeResources,
} from '../engine/__tests__/fixtures';
import { transitionLogEntries, transitionLogUpdate } from './logTimeline';

function withLog(state: GameState, log: readonly GameLogEntry[]): GameState {
  return {
    ...state,
    log: [...log],
  };
}

describe('transitionLogEntries', () => {
  it('returns only new engine entries for non end-turn actions', () => {
    const previous = withLog(makeGameState(), []);
    const next = withLog(makeGameState(), [
      {
        turn: 1,
        player: PLAYER_A,
        phase: 'ActionWindow',
        summary: 'develop 6',
      },
    ]);

    const action: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Knots: 1, Suns: 1, Waves: 1, Leaves: 1, Wyrms: 1 },
    };

    expect(transitionLogEntries(previous, next, action, PLAYER_A)).toEqual(
      next.log
    );
  });

  it('adds roll, tax, and income summaries after end-turn transitions', () => {
    const previous = withLog(
      makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 1, Suns: 0 }),
          }),
          makePlayer(PLAYER_B, {
            resources: makeResources({ Moons: 0, Suns: 0 }),
          }),
        ] as const,
      }),
      [
        {
          turn: 1,
          player: PLAYER_A,
          phase: 'ActionWindow',
          summary: 'develop 6',
        },
      ]
    );
    const next = withLog(
      {
        ...makeGameState({
          turn: 2,
          phase: 'ActionWindow',
          activePlayerIndex: 1,
          players: [
            makePlayer(PLAYER_A, {
              resources: makeResources({ Moons: 2, Suns: 0 }),
            }),
            makePlayer(PLAYER_B, {
              resources: makeResources({ Moons: 0, Suns: 1 }),
            }),
          ] as const,
          lastIncomeRoll: { die1: 7, die2: 4 },
        }),
        lastTaxSuit: undefined,
      },
      [
        ...previous.log,
        {
          turn: 1,
          player: PLAYER_A,
          phase: 'DrawCard',
          summary: 'end turn',
        },
      ]
    );

    const entries = transitionLogEntries(
      previous,
      next,
      { type: 'end-turn' },
      PLAYER_A
    );

    expect(entries.map((entry) => entry.summary)).toEqual([
      'end turn',
      'Roll d10 7/4 (income 7)',
      'Income You +1 Moons',
      'Income Bot +1 Suns',
    ]);
    expect(entries[1]?.player).toBe(PLAYER_B);
  });

  it('summarizes tax losses but defers income summaries while partial income is pending', () => {
    const previous = withLog(
      makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 3, Suns: 0, Leaves: 1 }),
          }),
          makePlayer(PLAYER_B, {
            resources: makeResources({ Moons: 2, Suns: 1, Leaves: 0 }),
          }),
        ] as const,
      }),
      []
    );
    const next = withLog(
      {
        ...makeGameState({
          turn: 2,
          phase: 'CollectIncome',
          activePlayerIndex: 0,
          players: [
            makePlayer(PLAYER_A, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 1 }),
            }),
            makePlayer(PLAYER_B, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 0 }),
            }),
          ] as const,
          lastIncomeRoll: { die1: 1, die2: 8 },
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D2',
              cardId: '8',
              suits: ['Suns', 'Leaves'],
            },
          ],
          incomeChoiceReturnPlayerId: PLAYER_B,
        }),
        lastTaxSuit: 'Moons',
      },
      [
        {
          turn: 1,
          player: PLAYER_A,
          phase: 'DrawCard',
          summary: 'end turn',
        },
      ]
    );

    const update = transitionLogUpdate(
      previous,
      next,
      { type: 'end-turn' },
      PLAYER_A,
      null
    );

    expect(update.entries.map((entry) => entry.summary)).toEqual([
      'end turn',
      'Roll d10 1/8 (income 8)',
      'Tax Moons (You -2, Bot -1)',
    ]);
    expect(update.entries[2]?.player).toBe(PLAYER_B);
    expect(update.deferredIncomeLogContext).not.toBeNull();
  });

  it('adds one complete income summary after final partial income choice resolves', () => {
    const previous = withLog(
      makeGameState({
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 3, Suns: 0, Leaves: 1 }),
          }),
          makePlayer(PLAYER_B, {
            resources: makeResources({ Moons: 2, Suns: 1, Leaves: 0 }),
          }),
        ] as const,
      }),
      []
    );
    const pending = withLog(
      {
        ...makeGameState({
          turn: 2,
          phase: 'CollectIncome',
          activePlayerIndex: 0,
          players: [
            makePlayer(PLAYER_A, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 1 }),
            }),
            makePlayer(PLAYER_B, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 0 }),
            }),
          ] as const,
          lastIncomeRoll: { die1: 1, die2: 8 },
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D2',
              cardId: '8',
              suits: ['Suns', 'Leaves'],
            },
          ],
          incomeChoiceReturnPlayerId: PLAYER_B,
        }),
        lastTaxSuit: 'Moons',
      },
      [
        {
          turn: 1,
          player: PLAYER_A,
          phase: 'DrawCard',
          summary: 'end turn',
        },
      ]
    );
    const initialUpdate = transitionLogUpdate(
      previous,
      pending,
      { type: 'end-turn' },
      PLAYER_A,
      null
    );
    const resolved = withLog(
      {
        ...makeGameState({
          turn: 2,
          phase: 'ActionWindow',
          activePlayerIndex: 1,
          players: [
            makePlayer(PLAYER_A, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 2 }),
            }),
            makePlayer(PLAYER_B, {
              resources: makeResources({ Moons: 1, Suns: 1, Leaves: 0 }),
            }),
          ] as const,
          lastIncomeRoll: { die1: 1, die2: 8 },
        }),
        lastTaxSuit: 'Moons',
      },
      [
        ...pending.log,
        {
          turn: 2,
          player: PLAYER_A,
          phase: 'CollectIncome',
          summary: 'income choice 8:Leaves',
        },
      ]
    );

    const finalUpdate = transitionLogUpdate(
      pending,
      resolved,
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D2',
        cardId: '8',
        suit: 'Leaves',
      },
      PLAYER_A,
      initialUpdate.deferredIncomeLogContext
    );

    expect(finalUpdate.entries.map((entry) => entry.summary)).toEqual([
      'income choice 8:Leaves',
      'Income You +1 Suns, +1 Leaves',
      'Income Bot none',
    ]);
    expect(finalUpdate.deferredIncomeLogContext).toBeNull();
  });
});
