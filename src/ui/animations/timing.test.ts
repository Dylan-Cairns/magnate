import { describe, expect, it } from 'vitest';

import {
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../../engine/__tests__/fixtures';
import type { GameAction } from '../../engine/types';
import {
  applySingleTaxLossToPreview,
  buildResourcePreviewByPlayer,
  cardFlightSettleMs,
  resourceFlightSettleMs,
  shouldAllowHumanActionsDuringAnimationSettle,
  shouldCommitBeforeAnimationSettle,
} from './timing';
import type { CardFlight, ResourceFlight } from './types';

describe('animation settle timing', () => {
  it('uses the latest resource flight end plus the commit buffer', () => {
    const flights: ResourceFlight[] = [
      makeResourceFlight({ delayMs: 75 }),
      makeResourceFlight({ delayMs: 10, durationMs: 600 }),
    ];

    expect(resourceFlightSettleMs([])).toBe(0);
    expect(resourceFlightSettleMs(flights)).toBe(630);
  });

  it('uses the latest card flight end plus the commit buffer', () => {
    const flights: CardFlight[] = [
      makeCardFlight({ delayMs: 280 }),
      makeCardFlight({ delayMs: 10, durationMs: 700 }),
    ];

    expect(cardFlightSettleMs([])).toBe(0);
    expect(cardFlightSettleMs(flights)).toBe(730);
  });
});

describe('animation commit semantics', () => {
  const actions: GameAction[] = [
    { type: 'buy-deed', cardId: '6', districtId: 'D1' },
    {
      type: 'develop-deed',
      cardId: '6',
      districtId: 'D1',
      tokens: { Moons: 1 },
    },
    {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Knots: 1 },
    },
    { type: 'trade', give: 'Moons', receive: 'Suns' },
    { type: 'sell-card', cardId: '6' },
    { type: 'end-turn' },
    {
      type: 'choose-income-suit',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '6',
      suit: 'Moons',
    },
  ];

  it('commits only sell and end-turn actions before animations settle', () => {
    expect(
      actions
        .filter(shouldCommitBeforeAnimationSettle)
        .map((action) => action.type)
    ).toEqual(['sell-card', 'end-turn']);
  });

  it('allows follow-up human input only during end-turn and income-choice settle', () => {
    expect(
      actions
        .filter(shouldAllowHumanActionsDuringAnimationSettle)
        .map((action) => action.type)
    ).toEqual(['end-turn', 'choose-income-suit']);
  });
});

describe('tax resource preview', () => {
  it('clones canonical resources and applies tax losses without mutating prior previews', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Moons: 1 }) }),
      ] as const,
    });
    const initial = buildResourcePreviewByPlayer(state);
    const afterFirst = applySingleTaxLossToPreview(initial, {
      playerId: PLAYER_A,
      suit: 'Moons',
    });
    const afterSecond = applySingleTaxLossToPreview(afterFirst, {
      playerId: PLAYER_A,
      suit: 'Moons',
    });

    expect(initial.PlayerA?.Moons).toBe(3);
    expect(afterFirst?.PlayerA?.Moons).toBe(2);
    expect(afterSecond?.PlayerA?.Moons).toBe(1);
    expect(state.players[0].resources.Moons).toBe(3);
  });

  it('leaves null and exhausted previews unchanged', () => {
    expect(
      applySingleTaxLossToPreview(null, {
        playerId: PLAYER_A,
        suit: 'Moons',
      })
    ).toBeNull();

    const preview = {
      [PLAYER_A]: makeResources({ Moons: 0 }),
    };
    expect(
      applySingleTaxLossToPreview(preview, {
        playerId: PLAYER_A,
        suit: 'Moons',
      })
    ).toBe(preview);
  });
});

function makeResourceFlight(
  overrides: Partial<ResourceFlight> = {}
): ResourceFlight {
  return {
    id: 'resource-flight',
    suit: 'Moons',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    delayMs: 0,
    ...overrides,
  };
}

function makeCardFlight(overrides: Partial<CardFlight> = {}): CardFlight {
  return {
    id: 'card-flight',
    variant: 'play',
    visual: 'face',
    cardId: '6',
    isDeed: false,
    perspective: 'human',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    startWidth: 100,
    startHeight: 150,
    endWidth: 100,
    endHeight: 150,
    delayMs: 0,
    ...overrides,
  };
}
