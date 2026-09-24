import { describe, expect, it } from 'vitest';

import type { CardId } from '../engine/cards';
import { findDevelopableCard } from '../engine/stateHelpers';
import type {
  DeedState,
  DistrictState,
  GameAction,
  GameState,
  ResourcePool,
  Suit,
} from '../engine/types';
import {
  courtActionBreakdown,
  courtPotentialValueForPlayerV2,
  isCourtCard,
} from './courtPotentialV2';
import { createHeuristicV2PositionContext } from './heuristicV2PositionContext';

const COURT_ID: CardId = '41';

describe('court valuation v2', () => {
  it('ignores non-court actions and standard-ruleset states', () => {
    const state = courtState({ hand: ['29', COURT_ID] });
    const context = createHeuristicV2PositionContext(state, 'PlayerA');
    const standardState = { ...state, ruleset: 'standard' as const };
    const standardContext = createHeuristicV2PositionContext(
      standardState,
      'PlayerA'
    );

    expect(
      courtActionBreakdown(
        { type: 'buy-deed', cardId: '29', districtId: 'D0' },
        state,
        'PlayerA',
        context,
        1
      )
    ).toBeUndefined();
    expect(
      courtActionBreakdown(
        {
          type: 'develop-outright',
          cardId: COURT_ID,
          districtId: 'D0',
          payment: { Moons: 4, Waves: 3, Knots: 3 },
        },
        state,
        'PlayerA',
        context,
        1
      )
    ).toBeUndefined();
    expect(
      courtActionBreakdown(
        { type: 'sell-card', cardId: COURT_ID },
        state,
        'PlayerA',
        context,
        1
      )
    ).toBeUndefined();
    expect(
      courtActionBreakdown(
        { type: 'buy-deed', cardId: COURT_ID, districtId: 'D0' },
        standardState,
        'PlayerA',
        standardContext,
        1
      )
    ).toBeUndefined();
  });

  it('ignores develop-deed actions on non-court deeds', () => {
    const state = courtState({
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: '29', progress: 4, tokens: { Moons: 4 } },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const context = createHeuristicV2PositionContext(state, 'PlayerA');

    expect(
      courtActionBreakdown(
        {
          type: 'develop-deed',
          cardId: '29',
          districtId: 'D0',
          tokens: { Moons: 1 },
        },
        state,
        'PlayerA',
        context,
        1
      )
    ).toBeUndefined();
  });

  it('gates a fresh court purchase on resources and income flow', () => {
    const buyCourt: GameAction = {
      type: 'buy-deed',
      cardId: COURT_ID,
      districtId: 'D0',
    };
    const broke = courtState({ resources: fixtureResources({}) });
    const funded = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
    });
    const flowing = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({ id: 'D0' }),
        district({ id: 'D1', playerADeveloped: ['27', '29', '24'] }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });

    const brokeBreakdown = courtActionBreakdown(
      buyCourt,
      broke,
      'PlayerA',
      createHeuristicV2PositionContext(broke, 'PlayerA'),
      1
    );
    const fundedBreakdown = courtActionBreakdown(
      buyCourt,
      funded,
      'PlayerA',
      createHeuristicV2PositionContext(funded, 'PlayerA'),
      1
    );
    const flowingBreakdown = courtActionBreakdown(
      buyCourt,
      flowing,
      'PlayerA',
      createHeuristicV2PositionContext(flowing, 'PlayerA'),
      1
    );

    expect(brokeBreakdown?.delta).toBe(0);
    expect(brokeBreakdown?.feasibility).toBe(0);
    expect(fundedBreakdown?.delta).toBeGreaterThan(0);
    expect(flowingBreakdown?.delta).toBeGreaterThan(fundedBreakdown?.delta ?? 0);
  });

  it('keeps buy, progress, and completion incentives coherent', () => {
    const buyCourt: GameAction = {
      type: 'buy-deed',
      cardId: COURT_ID,
      districtId: 'D0',
    };
    const fresh = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
    });
    const buyBreakdown = courtActionBreakdown(
      buyCourt,
      fresh,
      'PlayerA',
      createHeuristicV2PositionContext(fresh, 'PlayerA'),
      1
    );
    expect(buyBreakdown?.delta).toBeGreaterThan(0);

    for (const progress of [0, 4]) {
      const state = courtState({
        resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
        districts: [
          district({
            id: 'D0',
            playerADeed: {
              cardId: COURT_ID,
              progress,
              tokens: { Moons: progress },
            },
          }),
          district({ id: 'D1' }),
          district({ id: 'D2' }),
          district({ id: 'D3' }),
          district({ id: 'D4' }),
        ],
      });
      const breakdown = courtActionBreakdown(
        {
          type: 'develop-deed',
          cardId: COURT_ID,
          districtId: 'D0',
          tokens: { Waves: 1 },
        },
        state,
        'PlayerA',
        createHeuristicV2PositionContext(state, 'PlayerA'),
        1
      );
      expect(breakdown?.delta).toBeGreaterThan(0);
    }

    const completing = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: COURT_ID, progress: 9, tokens: { Moons: 9 } },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const completionBreakdown = courtActionBreakdown(
      {
        type: 'develop-deed',
        cardId: COURT_ID,
        districtId: 'D0',
        tokens: { Waves: 1 },
      },
      completing,
      'PlayerA',
      createHeuristicV2PositionContext(completing, 'PlayerA'),
      1
    );
    expect(completionBreakdown?.completes).toBe(true);
  });

  it('scales linearly with courtValueScale', () => {
    const state = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: COURT_ID, progress: 4, tokens: { Moons: 4 } },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const context = createHeuristicV2PositionContext(state, 'PlayerA');
    const action: GameAction = {
      type: 'develop-deed',
      cardId: COURT_ID,
      districtId: 'D0',
      tokens: { Waves: 1 },
    };

    const single = courtActionBreakdown(action, state, 'PlayerA', context, 1);
    const doubled = courtActionBreakdown(action, state, 'PlayerA', context, 2);
    const disabled = courtActionBreakdown(action, state, 'PlayerA', context, 0);

    expect(single?.delta).toBeGreaterThan(0);
    expect(doubled?.delta).toBeCloseTo((single?.delta ?? 0) * 2, 10);
    expect(disabled?.delta).toBe(0);
  });

  it('discounts courts in already-decided districts', () => {
    const contested = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: COURT_ID, progress: 4, tokens: { Moons: 4 } },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const decided = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({
          id: 'D0',
          playerADeveloped: ['29', '25'],
          playerADeed: { cardId: COURT_ID, progress: 4, tokens: { Moons: 4 } },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const action: GameAction = {
      type: 'develop-deed',
      cardId: COURT_ID,
      districtId: 'D0',
      tokens: { Waves: 1 },
    };

    const contestedBreakdown = courtActionBreakdown(
      action,
      contested,
      'PlayerA',
      createHeuristicV2PositionContext(contested, 'PlayerA'),
      1
    );
    const decidedBreakdown = courtActionBreakdown(
      action,
      decided,
      'PlayerA',
      createHeuristicV2PositionContext(decided, 'PlayerA'),
      1
    );

    expect(contestedBreakdown?.swing).toBeGreaterThan(0.9);
    expect(decidedBreakdown?.swing).toBeLessThan(
      (contestedBreakdown?.swing ?? 0) * 0.05
    );
  });

  it('shrinks feasibility as the game runs out of turns', () => {
    const buyCourt: GameAction = {
      type: 'buy-deed',
      cardId: COURT_ID,
      districtId: 'D0',
    };
    const early = courtState({
      turn: 10,
      resources: fixtureResources({ Moons: 2, Waves: 2, Knots: 2 }),
      districts: [
        district({ id: 'D0' }),
        district({ id: 'D1', playerADeveloped: ['27', '29', '24'] }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const late = {
      ...early,
      turn: 40,
      finalTurnsRemaining: 1,
    };

    const earlyBreakdown = courtActionBreakdown(
      buyCourt,
      early,
      'PlayerA',
      createHeuristicV2PositionContext(early, 'PlayerA'),
      1
    );
    const lateBreakdown = courtActionBreakdown(
      buyCourt,
      late,
      'PlayerA',
      createHeuristicV2PositionContext(late, 'PlayerA'),
      1
    );

    expect(earlyBreakdown?.feasibility).toBeGreaterThan(0);
    expect(lateBreakdown?.feasibility ?? 0).toBeLessThan(
      earlyBreakdown?.feasibility ?? 0
    );
  });

  it('identifies court cards for generic-potential exclusion', () => {
    expect(isCourtCard(undefined)).toBe(false);
    expect(isCourtCard(findDevelopableCard('29'))).toBe(false);
    expect(isCourtCard(findDevelopableCard(COURT_ID))).toBe(true);
  });

  it('values incomplete courts from state for the leaf evaluator', () => {
    const state = courtState({
      resources: fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: COURT_ID, progress: 4, tokens: {} },
        }),
        district({ id: 'D1', playerBDeveloped: ['29'] }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const context = createHeuristicV2PositionContext(state, 'PlayerA');
    const courtDistrict = state.districts[0];

    const value = courtPotentialValueForPlayerV2(
      state,
      'PlayerA',
      courtDistrict,
      context,
      1
    );
    expect(value).toBeGreaterThan(0);
    expect(
      courtPotentialValueForPlayerV2(
        state,
        'PlayerA',
        courtDistrict,
        context,
        2
      )
    ).toBeCloseTo((value ?? 0) * 2, 10);

    const standardState = { ...state, ruleset: 'standard' as const };
    expect(
      courtPotentialValueForPlayerV2(
        standardState,
        'PlayerA',
        standardState.districts[0],
        createHeuristicV2PositionContext(standardState, 'PlayerA'),
        1
      )
    ).toBeUndefined();
    expect(
      courtPotentialValueForPlayerV2(
        state,
        'PlayerA',
        state.districts[2],
        context,
        1
      )
    ).toBeUndefined();
    expect(
      courtPotentialValueForPlayerV2(state, 'PlayerB', courtDistrict, context, 1)
    ).toBeUndefined();
  });

  it('grows the state value convexly from court progress', () => {
    const base = courtState({
      resources: fixtureResources({ Moons: 1, Waves: 1, Knots: 1 }),
      districts: [
        district({
          id: 'D0',
          playerADeed: { cardId: COURT_ID, progress: 0, tokens: {} },
        }),
        district({ id: 'D1' }),
        district({ id: 'D2' }),
        district({ id: 'D3' }),
        district({ id: 'D4' }),
      ],
    });
    const atProgress = (progress: number): GameState => ({
      ...base,
      districts: base.districts.map((entry) => {
        if (entry.id !== 'D0') {
          return entry;
        }
        const deed = entry.stacks.PlayerA.deed;
        if (!deed) {
          return entry;
        }
        return {
          ...entry,
          stacks: {
            ...entry.stacks,
            PlayerA: {
              ...entry.stacks.PlayerA,
              deed: { ...deed, progress },
            },
          },
        };
      }),
    });
    const valueAt = (progress: number): number => {
      const state = atProgress(progress);
      return (
        courtPotentialValueForPlayerV2(
          state,
          'PlayerA',
          state.districts[0],
          createHeuristicV2PositionContext(state, 'PlayerA'),
          1
        ) ?? 0
      );
    };

    const fresh = valueAt(0);
    const mid = valueAt(4);
    const late = valueAt(8);

    expect(fresh).toBe(0);
    expect(mid).toBeGreaterThan(fresh);
    expect(late - mid).toBeGreaterThan(mid - fresh);
  });
});

function courtState({
  resources = fixtureResources({ Moons: 4, Waves: 4, Knots: 4 }),
  hand = [COURT_ID],
  crowns = [],
  districts = [
    district({ id: 'D0' }),
    district({ id: 'D1' }),
    district({ id: 'D2' }),
    district({ id: 'D3' }),
    district({ id: 'D4' }),
  ],
  turn = 20,
}: {
  resources?: ResourcePool;
  hand?: CardId[];
  crowns?: CardId[];
  districts?: DistrictState[];
  turn?: number;
} = {}): GameState {
  return {
    schemaVersion: 1,
    seed: 'court-valuation-fixture',
    rngCursor: 0,
    ruleset: 'extended',
    deck: {
      draw: ['42', '43', '44', '6', '7', '8', '9', '10'],
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

function district({
  id,
  playerADeveloped = [],
  playerBDeveloped = [],
  playerADeed,
}: {
  id: string;
  playerADeveloped?: CardId[];
  playerBDeveloped?: CardId[];
  playerADeed?: DeedState;
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
        deed: undefined,
      },
    },
  };
}

function fixtureResources(
  overrides: Partial<Record<Suit, number>> = {}
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
