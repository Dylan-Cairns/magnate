import { describe, expect, it } from 'vitest';

import { makeGameState, makePlayer, makeResources, PLAYER_A, PLAYER_B } from './__tests__/fixtures';
import type { GameAction } from './types';
import { actionStableKey, legalActionsCanonical, paymentSignature, toKeyedActions } from './actionSurface';

describe('action surface', () => {
  it('paymentSignature is stable regardless of object insertion order', () => {
    expect(paymentSignature({ Moons: 1, Knots: 2 })).toBe(
      paymentSignature({ Knots: 2, Moons: 1 })
    );
  });

  it('actionStableKey is stable for equivalent payloads', () => {
    const first: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D2',
      payment: { Moons: 1, Knots: 1 },
    };
    const second: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D2',
      payment: { Knots: 1, Moons: 1 },
    };

    expect(actionStableKey(first)).toBe(actionStableKey(second));
  });

  it('toKeyedActions sorts actions by stable key', () => {
    const unsorted: GameAction[] = [
      { type: 'trade', give: 'Suns', receive: 'Moons' },
      { type: 'end-turn' },
      { type: 'sell-card', cardId: '6' },
    ];

    const sorted = toKeyedActions(unsorted);
    expect(sorted.map((entry) => entry.actionKey)).toEqual([
      'end-turn',
      'sell-card:6',
      'trade:Suns:Moons',
    ]);
  });

  it('legalActionsCanonical is deterministic for the same state', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const first = legalActionsCanonical(state);
    const second = legalActionsCanonical(state);
    expect(first).toEqual(second);
  });
});

