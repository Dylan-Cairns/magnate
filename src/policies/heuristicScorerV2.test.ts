import { describe, expect, it } from 'vitest';

import type { CardId } from '../engine/cards';
import type {
  DeedState,
  DistrictState,
  GameAction,
  GameState,
  ResourcePool,
  Suit,
} from '../engine/types';
import {
  earningPotentialValueForPlayerV2,
  scoreHeuristicV2Action,
} from './heuristicScorerV2';

describe('heuristic scorer v2', () => {
  it('values high-rank off-suit generator deeds as future earning access', () => {
    const state = heuristicV2FixtureState({
      turn: 4,
      resources: fixtureResources({ Waves: 1, Wyrms: 1 }),
      hand: ['27'],
    });
    const buyRank9OffSuit: GameAction = {
      type: 'buy-deed',
      cardId: '27',
      districtId: 'D0',
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicV2Action(buyRank9OffSuit, { state })).toBeGreaterThan(
      scoreHeuristicV2Action(endTurn, { state })
    );
  });

  it('applies diminishing returns to extra access in an already strong suit', () => {
    const weakMoons = heuristicV2FixtureState({ turn: 4 });
    const strongMoons = heuristicV2FixtureState({
      turn: 4,
      crowns: ['32', '32', '32'],
    });
    const developAceMoons: GameAction = {
      type: 'develop-outright',
      cardId: '2',
      districtId: 'D0',
      payment: { Moons: 3 },
    };

    expect(
      scoreHeuristicV2Action(developAceMoons, { state: strongMoons })
    ).toBeLessThan(scoreHeuristicV2Action(developAceMoons, { state: weakMoons }));
  });

  it('saturates scoring delta so overinvesting in a far-ahead district is weak', () => {
    const state = heuristicV2FixtureState({
      turn: 34,
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['29', '25'],
        }),
        fixtureDistrict({
          id: 'D1',
          playerBDeveloped: ['9'],
        }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const overinvest: GameAction = {
      type: 'develop-outright',
      cardId: '13',
      districtId: 'D0',
      payment: { Moons: 2, Suns: 2 },
    };
    const flipCloseDistrict: GameAction = {
      type: 'develop-outright',
      cardId: '13',
      districtId: 'D1',
      payment: { Moons: 2, Suns: 2 },
    };

    expect(scoreHeuristicV2Action(flipCloseDistrict, { state })).toBeGreaterThan(
      scoreHeuristicV2Action(overinvest, { state })
    );
  });

  it('values near-complete deeds more than new deeds for scoring potential', () => {
    const newDeed = heuristicV2FixtureState({
      turn: 30,
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 0, tokens: {} },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const nearCompleteDeed = heuristicV2FixtureState({
      turn: 30,
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 8, tokens: { Moons: 8 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const developDeed: GameAction = {
      type: 'develop-deed',
      cardId: '29',
      districtId: 'D0',
      tokens: { Suns: 1 },
    };

    expect(
      scoreHeuristicV2Action(developDeed, { state: nearCompleteDeed })
    ).toBeGreaterThan(scoreHeuristicV2Action(developDeed, { state: newDeed }));
  });

  it('does not count current loose resources in earning potential', () => {
    const noLooseResources = heuristicV2FixtureState({
      resources: fixtureResources({}),
    });
    const manyLooseResources = heuristicV2FixtureState({
      resources: fixtureResources({
        Moons: 9,
        Suns: 9,
        Waves: 9,
        Leaves: 9,
        Wyrms: 9,
        Knots: 9,
      }),
    });

    expect(earningPotentialValueForPlayerV2(manyLooseResources, 'PlayerA')).toBe(
      earningPotentialValueForPlayerV2(noLooseResources, 'PlayerA')
    );
  });

  it('prefers developing a deed from surplus tokens over spending a scarce last token', () => {
    const scarce = heuristicV2FixtureState({
      resources: fixtureResources({ Wyrms: 1 }),
      hand: [],
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '24', progress: 0, tokens: {} },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const surplus = heuristicV2FixtureState({
      resources: fixtureResources({ Wyrms: 4 }),
      hand: [],
      districts: [...scarce.districts],
    });
    const developWyrms: GameAction = {
      type: 'develop-deed',
      cardId: '24',
      districtId: 'D0',
      tokens: { Wyrms: 1 },
    };

    expect(scoreHeuristicV2Action(developWyrms, { state: surplus })).toBeGreaterThan(
      scoreHeuristicV2Action(developWyrms, { state: scarce })
    );
  });

  it('prefers selling for suits that match stronger remaining demand', () => {
    const state = heuristicV2FixtureState({
      resources: fixtureResources({}),
      hand: ['12', '13', '24'],
    });
    const sellWyrmsKnots: GameAction = { type: 'sell-card', cardId: '12' };
    const sellMoonsSuns: GameAction = { type: 'sell-card', cardId: '13' };

    expect(scoreHeuristicV2Action(sellWyrmsKnots, { state })).toBeGreaterThan(
      scoreHeuristicV2Action(sellMoonsSuns, { state })
    );
  });

  it('prefers trade liquidity into a missing high-demand suit', () => {
    const state = heuristicV2FixtureState({
      resources: fixtureResources({ Moons: 3 }),
      hand: ['24'],
    });
    const tradeForDemand: GameAction = {
      type: 'trade',
      give: 'Moons',
      receive: 'Wyrms',
    };
    const tradeForLowDemand: GameAction = {
      type: 'trade',
      give: 'Moons',
      receive: 'Suns',
    };

    expect(scoreHeuristicV2Action(tradeForDemand, { state })).toBeGreaterThan(
      scoreHeuristicV2Action(tradeForLowDemand, { state })
    );
  });
});

function heuristicV2FixtureState({
  resources = fixtureResources(),
  districts = [
    fixtureDistrict({ id: 'D0' }),
    fixtureDistrict({ id: 'D1' }),
    fixtureDistrict({ id: 'D2' }),
    fixtureDistrict({ id: 'D3' }),
    fixtureDistrict({ id: 'D4' }),
  ],
  hand = ['6', '9', '13', '29'],
  crowns = [],
  turn = 20,
}: {
  resources?: ResourcePool;
  districts?: DistrictState[];
  hand?: CardId[];
  crowns?: CardId[];
  turn?: number;
} = {}): GameState {
  return {
    schemaVersion: 1,
    seed: 'heuristic-v2-fixture',
    rngCursor: 0,
    deck: {
      draw: ['6', '7', '8', '9', '10', '11', '12', '13'],
      discard: [],
      reshuffles: 0,
    },
    players: [
      {
        id: 'PlayerA',
        hand,
        crowns,
        resources,
      },
      {
        id: 'PlayerB',
        hand: [],
        crowns: [],
        resources: fixtureResources({}),
      },
    ],
    activePlayerIndex: 0,
    turn,
    phase: 'ActionWindow',
    districts,
    cardPlayedThisTurn: false,
    log: [],
  };
}

function fixtureDistrict({
  id,
  playerADeveloped = [],
  playerBDeveloped = [],
  playerADeed,
  playerBDeed,
}: {
  id: string;
  playerADeveloped?: CardId[];
  playerBDeveloped?: CardId[];
  playerADeed?: DeedState;
  playerBDeed?: DeedState;
}): DistrictState {
  return {
    id,
    markerSuitMask: [],
    stacks: {
      PlayerA: {
        developed: playerADeveloped,
        deed: playerADeed,
      },
      PlayerB: {
        developed: playerBDeveloped,
        deed: playerBDeed,
      },
    },
  };
}

function fixtureResources(
  overrides: Partial<Record<Suit, number>> = {
    Moons: 5,
    Suns: 5,
    Waves: 5,
    Leaves: 5,
    Wyrms: 5,
    Knots: 5,
  }
): ResourcePool {
  return {
    Moons: overrides.Moons ?? 0,
    Suns: overrides.Suns ?? 0,
    Waves: overrides.Waves ?? 0,
    Leaves: overrides.Leaves ?? 0,
    Wyrms: overrides.Wyrms ?? 0,
    Knots: overrides.Knots ?? 0,
  };
}
