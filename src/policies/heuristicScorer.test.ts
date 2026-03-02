import { describe, expect, it } from 'vitest';

import type { CardId } from '../engine/cards';
import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import type {
  DeedState,
  DistrictState,
  GameAction,
  GameState,
  ResourcePool,
  Suit,
} from '../engine/types';
import { toPlayerView } from '../engine/view';
import { heuristicPolicy } from './heuristicPolicy';
import {
  heuristicPriorsByKey,
  rankHeuristicActions,
  scoreHeuristicAction,
  scoreHeuristicActions,
} from './heuristicScorer';

describe('heuristic scorer', () => {
  it('preserves the established action-type and rank preferences', () => {
    const develop: GameAction = {
      type: 'develop-outright',
      cardId: '29',
      districtId: 'D0',
      payment: {},
    };
    const sell: GameAction = {
      type: 'sell-card',
      cardId: '6',
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(develop)).toBeGreaterThan(
      scoreHeuristicAction(sell)
    );
    expect(scoreHeuristicAction(sell)).toBeGreaterThan(
      scoreHeuristicAction(endTurn)
    );
  });

  it('ranks equal-score actions by stable action key', () => {
    const actions: GameAction[] = [
      { type: 'sell-card', cardId: '7' },
      { type: 'sell-card', cardId: '6' },
    ];

    const ranked = rankHeuristicActions(actions);

    expect(ranked.map((entry) => entry.actionKey)).toEqual([
      'sell-card:6',
      'sell-card:7',
    ]);
  });

  it('returns finite normalized priors for legal actions', () => {
    const actions: GameAction[] = [
      { type: 'sell-card', cardId: '6' },
      { type: 'sell-card', cardId: '7' },
      { type: 'end-turn' },
    ];

    const priors = heuristicPriorsByKey(actions);
    const total = [...priors.values()].reduce((sum, value) => sum + value, 0);

    expect(priors.size).toBe(actions.length);
    expect(total).toBeCloseTo(1);
    for (const value of priors.values()) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
    }
  });

  it('scores actions with rank and prior metadata', () => {
    const actions: GameAction[] = [
      { type: 'end-turn' },
      { type: 'sell-card', cardId: '6' },
    ];

    const scored = scoreHeuristicActions(actions);

    expect(scored).toHaveLength(2);
    expect(scored[0].rank).toBe(0);
    expect(scored[0].score).toBeGreaterThan(scored[1].score);
    expect(scored[0].prior).toBeGreaterThan(scored[1].prior);
  });

  it('strongly rejects rank 2 deed purchases', () => {
    const state = heuristicFixtureState();
    const buyRank2: GameAction = {
      type: 'buy-deed',
      cardId: '6',
      districtId: 'D0',
    };
    const sellRank2: GameAction = {
      type: 'sell-card',
      cardId: '6',
    };

    expect(scoreHeuristicAction(buyRank2, { state })).toBeLessThan(
      scoreHeuristicAction(sellRank2, { state })
    );
  });

  it('does not reject ace deed purchases as rank 2 deeds', () => {
    const state = heuristicFixtureState({
      resources: fixtureResources({ Knots: 1 }),
      hand: ['0'],
    });
    const buyAce: GameAction = {
      type: 'buy-deed',
      cardId: '0',
      districtId: 'D0',
    };
    const sellAce: GameAction = {
      type: 'sell-card',
      cardId: '0',
    };

    expect(scoreHeuristicAction(buyAce, { state })).toBeGreaterThan(
      scoreHeuristicAction(sellAce, { state })
    );
  });

  it('avoids high-rank deed purchases in the final turns', () => {
    const state = heuristicFixtureState({
      finalTurnsRemaining: 2,
    });
    const buyHighRank: GameAction = {
      type: 'buy-deed',
      cardId: '29',
      districtId: 'D0',
    };
    const sellHighRank: GameAction = {
      type: 'sell-card',
      cardId: '29',
    };

    expect(scoreHeuristicAction(buyHighRank, { state })).toBeLessThan(
      scoreHeuristicAction(sellHighRank, { state })
    );
  });

  it('prioritizes a district-flipping development over reinforcing a district already controlled', () => {
    const state = heuristicFixtureState({
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['13'],
          playerBDeveloped: ['24'],
        }),
        fixtureDistrict({
          id: 'D1',
          playerADeveloped: ['13'],
        }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const flipDistrict: GameAction = {
      type: 'develop-outright',
      cardId: '29',
      districtId: 'D0',
      payment: { Moons: 5, Suns: 4 },
    };
    const reinforceControlled: GameAction = {
      type: 'develop-outright',
      cardId: '29',
      districtId: 'D1',
      payment: { Moons: 5, Suns: 4 },
    };

    const ranked = rankHeuristicActions(
      [reinforceControlled, flipDistrict],
      { state }
    );

    expect(ranked[0].action).toEqual(flipDistrict);
  });

  it('deprioritizes investing outside the main districts when control is unrealistic', () => {
    const state = heuristicFixtureState({
      districts: [
        fixtureDistrict({ id: 'D0' }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({
          id: 'D4',
          playerBDeveloped: ['29', '24'],
        }),
      ],
    });
    const mainDistrict: GameAction = {
      type: 'develop-outright',
      cardId: '9',
      districtId: 'D0',
      payment: { Moons: 2, Waves: 1 },
    };
    const unrealisticDistrict: GameAction = {
      type: 'develop-outright',
      cardId: '9',
      districtId: 'D4',
      payment: { Moons: 2, Waves: 1 },
    };

    const ranked = rankHeuristicActions(
      [unrealisticDistrict, mainDistrict],
      { state }
    );

    expect(ranked[0].action).toEqual(mainDistrict);
  });

  it('prefers developing deeds with surplus tokens over spending a last token in a suit', () => {
    const state = heuristicFixtureState({
      resources: fixtureResources({ Moons: 1, Suns: 3 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '13', progress: 0, tokens: {} },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const spendLastMoon: GameAction = {
      type: 'develop-deed',
      cardId: '13',
      districtId: 'D0',
      tokens: { Moons: 1 },
    };
    const shelterSurplusSun: GameAction = {
      type: 'develop-deed',
      cardId: '13',
      districtId: 'D0',
      tokens: { Suns: 1 },
    };

    const ranked = rankHeuristicActions(
      [spendLastMoon, shelterSurplusSun],
      { state }
    );

    expect(ranked[0].action).toEqual(shelterSurplusSun);
  });

  it('prefers ending the turn over non-completing deed progress that spends a last suit token', () => {
    const state = heuristicFixtureState({
      cardPlayedThisTurn: true,
      resources: fixtureResources({ Moons: 1 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 6, tokens: { Suns: 6 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const spendLastMoon: GameAction = {
      type: 'develop-deed',
      cardId: '29',
      districtId: 'D0',
      tokens: { Moons: 1 },
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(endTurn, { state })).toBeGreaterThan(
      scoreHeuristicAction(spendLastMoon, { state })
    );
  });

  it('still completes a deed with the last token when completion is available', () => {
    const state = heuristicFixtureState({
      cardPlayedThisTurn: true,
      resources: fixtureResources({ Moons: 1 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 8, tokens: { Suns: 8 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const completeDeed: GameAction = {
      type: 'develop-deed',
      cardId: '29',
      districtId: 'D0',
      tokens: { Moons: 1 },
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(completeDeed, { state })).toBeGreaterThan(
      scoreHeuristicAction(endTurn, { state })
    );
  });

  it('allows non-completing deed progress with surplus tokens in a useful district', () => {
    const state = heuristicFixtureState({
      cardPlayedThisTurn: true,
      resources: fixtureResources({ Moons: 2 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 6, tokens: { Suns: 6 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const shelterSurplusMoon: GameAction = {
      type: 'develop-deed',
      cardId: '29',
      districtId: 'D0',
      tokens: { Moons: 1 },
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(shelterSurplusMoon, { state })).toBeGreaterThan(
      scoreHeuristicAction(endTurn, { state })
    );
  });

  it('penalizes trades that do not unlock a high-value move', () => {
    const state = heuristicFixtureState({
      resources: fixtureResources({ Moons: 3 }),
      hand: ['13'],
    });
    const trade: GameAction = {
      type: 'trade',
      give: 'Moons',
      receive: 'Suns',
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(trade, { state })).toBeLessThan(
      scoreHeuristicAction(endTurn, { state })
    );
  });

  it('allows trades that immediately unlock a district-flipping development', () => {
    const state = heuristicFixtureState({
      resources: fixtureResources({ Moons: 4, Knots: 3 }),
      hand: ['13'],
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerBDeveloped: ['9'],
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const trade: GameAction = {
      type: 'trade',
      give: 'Knots',
      receive: 'Suns',
    };
    const endTurn: GameAction = { type: 'end-turn' };

    expect(scoreHeuristicAction(trade, { state })).toBeGreaterThan(
      scoreHeuristicAction(endTurn, { state })
    );
  });

  it('penalizes unsupported rank 9 deed starts more than supported rank 9 starts', () => {
    const unsupported = heuristicFixtureState({
      resources: fixtureResources({ Moons: 1, Suns: 1 }),
      hand: ['29'],
      crowns: [],
    });
    const supported = heuristicFixtureState({
      resources: fixtureResources({ Moons: 1, Suns: 1 }),
      hand: ['29'],
      crowns: ['32', '33'],
    });
    const buyRank9: GameAction = {
      type: 'buy-deed',
      cardId: '29',
      districtId: 'D0',
    };

    expect(scoreHeuristicAction(buyRank9, { state: unsupported })).toBeLessThan(
      scoreHeuristicAction(buyRank9, { state: supported })
    );
  });

  it('prioritizes deed progress when completion is close', () => {
    const farFromCompletion = heuristicFixtureState({
      resources: fixtureResources({ Moons: 2, Suns: 2 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '25', progress: 0, tokens: {} },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const closeToCompletion = heuristicFixtureState({
      resources: fixtureResources({ Moons: 2, Suns: 2 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeed: { cardId: '25', progress: 6, tokens: { Moons: 6 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const developDeed: GameAction = {
      type: 'develop-deed',
      cardId: '25',
      districtId: 'D0',
      tokens: { Suns: 1 },
    };

    expect(scoreHeuristicAction(developDeed, { state: closeToCompletion })).toBeGreaterThan(
      scoreHeuristicAction(developDeed, { state: farFromCompletion })
    );
  });

  it('defends fragile controlled districts before overinvesting in safely controlled districts', () => {
    const state = heuristicFixtureState({
      resources: fixtureResources({ Moons: 5, Suns: 5 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['13'],
          playerBDeveloped: ['9'],
        }),
        fixtureDistrict({
          id: 'D1',
          playerADeveloped: ['29'],
        }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const defendFragile: GameAction = {
      type: 'develop-outright',
      cardId: '25',
      districtId: 'D0',
      payment: { Moons: 4, Suns: 4 },
    };
    const overinvest: GameAction = {
      type: 'develop-outright',
      cardId: '25',
      districtId: 'D1',
      payment: { Moons: 4, Suns: 4 },
    };

    const ranked = rankHeuristicActions([overinvest, defendFragile], { state });

    expect(ranked[0].action).toEqual(defendFragile);
  });

  it('prefers a credible third-district deed start over a safe controlled district', () => {
    const state = heuristicFixtureState({
      hand: ['23'],
      resources: fixtureResources({ Moons: 1, Leaves: 1 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['19'],
        }),
        fixtureDistrict({
          id: 'D1',
          playerADeveloped: ['29'],
        }),
        fixtureDistrict({
          id: 'D2',
          playerADeveloped: ['25'],
          playerBDeveloped: ['29'],
        }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const safeControlledDistrict: GameAction = {
      type: 'buy-deed',
      cardId: '23',
      districtId: 'D0',
    };
    const credibleThirdDistrict: GameAction = {
      type: 'buy-deed',
      cardId: '23',
      districtId: 'D2',
    };

    const ranked = rankHeuristicActions(
      [safeControlledDistrict, credibleThirdDistrict],
      { state }
    );

    expect(ranked[0].action).toEqual(credibleThirdDistrict);
  });

  it('prefers selling over dumping a card into a safe district when expansion is unrealistic', () => {
    const state = heuristicFixtureState({
      hand: ['15'],
      resources: fixtureResources({ Suns: 2, Waves: 3 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['19'],
        }),
        fixtureDistrict({
          id: 'D1',
          playerADeveloped: ['29'],
        }),
        fixtureDistrict({
          id: 'D2',
          playerBDeveloped: ['29', '25'],
        }),
        fixtureDistrict({
          id: 'D3',
          playerBDeveloped: ['27', '17'],
        }),
        fixtureDistrict({
          id: 'D4',
          playerBDeveloped: ['28', '24'],
        }),
      ],
    });
    const overinvest: GameAction = {
      type: 'develop-outright',
      cardId: '15',
      districtId: 'D0',
      payment: { Suns: 2, Waves: 3 },
    };
    const sell: GameAction = {
      type: 'sell-card',
      cardId: '15',
    };

    expect(scoreHeuristicAction(sell, { state })).toBeGreaterThan(
      scoreHeuristicAction(overinvest, { state })
    );
  });

  it('treats far-from-complete opponent deeds as less urgent than close deeds', () => {
    const farThreat = heuristicFixtureState({
      hand: ['25'],
      resources: fixtureResources({ Moons: 4, Suns: 4 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['18'],
          playerBDeed: { cardId: '27', progress: 0, tokens: {} },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const closeThreat = heuristicFixtureState({
      hand: ['25'],
      resources: fixtureResources({ Moons: 4, Suns: 4 }),
      districts: [
        fixtureDistrict({
          id: 'D0',
          playerADeveloped: ['18'],
          playerBDeed: { cardId: '27', progress: 7, tokens: { Waves: 7 } },
        }),
        fixtureDistrict({ id: 'D1' }),
        fixtureDistrict({ id: 'D2' }),
        fixtureDistrict({ id: 'D3' }),
        fixtureDistrict({ id: 'D4' }),
      ],
    });
    const defendDistrict: GameAction = {
      type: 'develop-outright',
      cardId: '25',
      districtId: 'D0',
      payment: { Moons: 4, Suns: 4 },
    };

    expect(scoreHeuristicAction(defendDistrict, { state: closeThreat })).toBeGreaterThan(
      scoreHeuristicAction(defendDistrict, { state: farThreat })
    );
  });
});

describe('heuristic policy', () => {
  it('selects a legal action in a real session', async () => {
    const state = createSession('heuristic-policy-test', 'PlayerA');
    const view = toPlayerView(state, 'PlayerA');
    const actions = legalActions(state);
    const selected = await Promise.resolve(
      heuristicPolicy.selectAction({
        state,
        view,
        legalActions: actions,
        random: rngFromSeed('heuristic-policy-test-rng'),
      })
    );

    expect(selected).toBeDefined();
    const legalKeys = new Set(actions.map((action) => actionStableKey(action)));
    expect(legalKeys.has(actionStableKey(selected!))).toBe(true);
  });
});

function heuristicFixtureState({
  resources = fixtureResources(),
  districts = [
    fixtureDistrict({ id: 'D0' }),
    fixtureDistrict({ id: 'D1' }),
    fixtureDistrict({ id: 'D2' }),
    fixtureDistrict({ id: 'D3' }),
    fixtureDistrict({ id: 'D4' }),
  ],
  finalTurnsRemaining,
  hand = ['6', '9', '13', '29'],
  crowns = [],
  cardPlayedThisTurn = false,
}: {
  resources?: ResourcePool;
  districts?: DistrictState[];
  finalTurnsRemaining?: number;
  hand?: CardId[];
  crowns?: CardId[];
  cardPlayedThisTurn?: boolean;
} = {}): GameState {
  return {
    schemaVersion: 1,
    seed: 'heuristic-fixture',
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
    turn: 20,
    phase: 'ActionWindow',
    districts,
    cardPlayedThisTurn,
    finalTurnsRemaining,
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
