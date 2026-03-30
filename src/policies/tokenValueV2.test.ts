import { describe, expect, it } from 'vitest';

import {
  asCardId,
  makeDefaultDistricts,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../engine/__tests__/fixtures';
import type { DistrictState, Suit } from '../engine/types';
import {
  contextualSuitTokenValuesV2,
  directResourceValueForSuit,
  resourceBankValueBreakdownV2,
  resourceDeltaForActionV2,
  tokenDeltaForActionV2,
  tradeLiquidityValueForSuitV2,
  type SuitTokenValueV2,
  type SuitValueMap,
} from './tokenValueV2';

describe('token value v2', () => {
  it('values scarce suits with current hand demand more than unrelated suits', () => {
    const state = makeTokenValueState({
      handA: ['24'],
      crownsA: [],
    });

    const values = contextualSuitTokenValuesV2(state, PLAYER_A);

    expect(values.Wyrms.value).toBeGreaterThan(values.Suns.value);
    expect(values.Knots.value).toBeGreaterThan(values.Suns.value);
  });

  it('discounts demanded suits when the player has strong replacement access', () => {
    const scarceWyrms = makeTokenValueState({
      handA: ['24'],
      crownsA: [],
    });
    const accessibleWyrms = makeTokenValueState({
      handA: ['24'],
      crownsA: ['35', '35', '35'],
    });

    expect(
      contextualSuitTokenValuesV2(accessibleWyrms, PLAYER_A).Wyrms.value
    ).toBeLessThan(
      contextualSuitTokenValuesV2(scarceWyrms, PLAYER_A).Wyrms.value
    );
  });

  it('uses a concave direct marginal token curve within a suit', () => {
    const one = directResourceValueForSuit(1, 10);
    const two = directResourceValueForSuit(2, 10);
    const three = directResourceValueForSuit(3, 10);
    const four = directResourceValueForSuit(4, 10);

    expect(two - one).toBeLessThan(one);
    expect(three - two).toBeLessThan(two - one);
    expect(four - three).toBeLessThan(three - two);
  });

  it('adds modulo-3 trade liquidity for completed and near-completed trade sets', () => {
    const values = fixedSuitValues({
      Moons: 1,
      Wyrms: 10,
    });

    const liquidity2 = tradeLiquidityValueForSuitV2(
      'Moons',
      makeResources({ Moons: 2 }),
      values
    );
    const liquidity3 = tradeLiquidityValueForSuitV2(
      'Moons',
      makeResources({ Moons: 3 }),
      values
    );
    const liquidity4 = tradeLiquidityValueForSuitV2(
      'Moons',
      makeResources({ Moons: 4 }),
      values
    );
    const liquidity5 = tradeLiquidityValueForSuitV2(
      'Moons',
      makeResources({ Moons: 5 }),
      values
    );
    const liquidity6 = tradeLiquidityValueForSuitV2(
      'Moons',
      makeResources({ Moons: 6 }),
      values
    );

    expect(liquidity2).toBeGreaterThan(0);
    expect(liquidity3).toBeGreaterThan(liquidity2);
    expect(liquidity4).toBeGreaterThan(liquidity3);
    expect(liquidity5).toBeGreaterThan(liquidity4);
    expect(liquidity6).toBeGreaterThan(liquidity5);
  });

  it('keeps trade liquidity separate from direct bank value', () => {
    const state = makeTokenValueState({
      handA: ['24'],
      crownsA: [],
      resourcesA: makeResources({ Moons: 3 }),
    });

    const breakdown = resourceBankValueBreakdownV2(state, PLAYER_A);

    expect(breakdown.directValue).toBeGreaterThan(0);
    expect(breakdown.tradeLiquidityBySuit.Moons).toBeGreaterThan(0);
    expect(breakdown.totalValue).toBe(
      breakdown.directValue + breakdown.tradeLiquidityValue
    );
  });

  it('projects action resource deltas for local token scoring', () => {
    expect(
      resourceDeltaForActionV2({
        type: 'trade',
        give: 'Moons',
        receive: 'Wyrms',
      })
    ).toEqual({ Moons: -3, Wyrms: 1 });
    expect(
      resourceDeltaForActionV2({
        type: 'sell-card',
        cardId: asCardId('0'),
      })
    ).toEqual({ Knots: 2 });
  });

  it('recognizes a trade from surplus low-demand tokens into missing high-demand tokens', () => {
    const state = makeTokenValueState({
      handA: ['24'],
      crownsA: [],
      resourcesA: makeResources({ Moons: 3 }),
    });

    expect(
      tokenDeltaForActionV2(
        { type: 'trade', give: 'Moons', receive: 'Wyrms' },
        state,
        PLAYER_A
      )
    ).toBeGreaterThan(0);
  });
});

function makeTokenValueState({
  handA = [],
  crownsA = [],
  resourcesA = makeResources(),
  districts = makeDefaultDistricts(),
}: {
  handA?: string[];
  crownsA?: string[];
  resourcesA?: ReturnType<typeof makeResources>;
  districts?: DistrictState[];
} = {}) {
  return makeGameState({
    turn: 8,
    districts,
    players: [
      makePlayer(PLAYER_A, {
        hand: handA.map(asCardId),
        crowns: crownsA.map(asCardId),
        resources: resourcesA,
      }),
      makePlayer(PLAYER_B, {
        hand: [],
        crowns: [],
        resources: makeResources(),
      }),
    ],
  });
}

function fixedSuitValues(
  overrides: Partial<Record<Suit, number>>
): SuitValueMap<SuitTokenValueV2> {
  const make = (suit: Suit): SuitTokenValueV2 => {
    const value = overrides[suit] ?? 0.5;
    return {
      suit,
      earningDemand: value,
      scoringDemand: 0,
      rawDemand: value,
      access: 0,
      replaceabilityMultiplier: 1,
      value,
    };
  };
  return {
    Moons: make('Moons'),
    Suns: make('Suns'),
    Waves: make('Waves'),
    Leaves: make('Leaves'),
    Wyrms: make('Wyrms'),
    Knots: make('Knots'),
  };
}
