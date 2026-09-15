import { describe, expect, it } from 'vitest';

import { legalActions } from './actionBuilders';
import { CARD_BY_ID, COURT_CARDS, PROPERTY_CARDS } from './cards';
import { initialSetup } from './deck';
import { newGame } from './game';
import { incomeForResult } from './income';
import { applyKnownLegalAction } from './reducer';
import { districtScore, scoreGame } from './scoring';
import {
  deedCost,
  developmentCost,
  enumerateOutrightPayments,
} from './stateHelpers';
import {
  PLAYER_A,
  findLegalActionByType,
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
  withDeed,
} from './__tests__/fixtures';
import type { DevelopableCard, GameState } from './types';

// The Consul is the extended-deck Court with suits Moons, Waves, and Knots.
const CONSUL_ID = '41';
const CONSUL = CARD_BY_ID[CONSUL_ID] as DevelopableCard;

const COURT_IDS = COURT_CARDS.map((card) => card.id);

function extendedSetup(seed: string) {
  return initialSetup(seed, 'extended');
}

describe('extended rules: setup', () => {
  it('deals the Courts only in the extended ruleset', () => {
    const extended = extendedSetup('extended-seed');
    const standard = initialSetup('extended-seed');

    const extendedPool = [
      ...extended.deck.draw,
      ...extended.handsByPlayer.PlayerA,
      ...extended.handsByPlayer.PlayerB,
    ];
    const standardPool = [
      ...standard.deck.draw,
      ...standard.handsByPlayer.PlayerA,
      ...standard.handsByPlayer.PlayerB,
    ];

    expect(new Set(extendedPool).size).toBe(
      PROPERTY_CARDS.length + COURT_CARDS.length
    );
    expect(extended.deck.draw).toHaveLength(
      PROPERTY_CARDS.length + COURT_CARDS.length - 6
    );
    for (const courtId of COURT_IDS) {
      expect(extendedPool).toContain(courtId);
    }

    expect(standard.deck.draw).toHaveLength(PROPERTY_CARDS.length - 6);
    for (const courtId of COURT_IDS) {
      expect(standardPool).not.toContain(courtId);
    }
  });

  it('records the ruleset on the game state', () => {
    expect(newGame('standard-seed').ruleset).toBe('standard');
    expect(newGame('standard-seed', { ruleset: 'extended' }).ruleset).toBe(
      'extended'
    );
  });
});

describe('extended rules: costs', () => {
  it('deeds a Court for three tokens, one per suit', () => {
    expect(deedCost(CONSUL)).toEqual({
      Moons: 1,
      Waves: 1,
      Knots: 1,
    });
  });

  it('develops a Court for ten tokens', () => {
    expect(developmentCost(CONSUL)).toBe(10);
  });

  it('requires at least one token of each Court suit to develop outright', () => {
    const payments = enumerateOutrightPayments(
      CONSUL,
      makeResources({ Moons: 8, Waves: 1, Knots: 1 })
    );
    expect(payments.length).toBeGreaterThan(0);
    for (const payment of payments) {
      expect(
        (payment.Moons ?? 0) + (payment.Waves ?? 0) + (payment.Knots ?? 0)
      ).toBe(10);
      expect(payment.Moons ?? 0).toBeGreaterThanOrEqual(1);
      expect(payment.Waves ?? 0).toBeGreaterThanOrEqual(1);
      expect(payment.Knots ?? 0).toBeGreaterThanOrEqual(1);
    }
    expect(
      enumerateOutrightPayments(CONSUL, makeResources({ Moons: 9, Waves: 1 }))
    ).toEqual([]);
  });
});

describe('extended rules: playing a Court', () => {
  it('sells a Court for three tokens, one per suit', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, { hand: [CONSUL_ID] }),
        makePlayer('PlayerB'),
      ] as const,
      phase: 'ActionWindow',
    });

    const sell = findLegalActionByType(state, 'sell-card');
    const next = applyKnownLegalAction(state, sell);

    expect(next.players[0].hand).toEqual([]);
    expect(next.players[0].resources.Moons).toBe(1);
    expect(next.players[0].resources.Waves).toBe(1);
    expect(next.players[0].resources.Knots).toBe(1);
  });

  it('buys a Court deed for three tokens, one per suit', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: [CONSUL_ID],
          resources: makeResources({ Moons: 1, Waves: 1, Knots: 1 }),
        }),
        makePlayer('PlayerB'),
      ] as const,
      phase: 'ActionWindow',
    });

    const buy = findLegalActionByType(state, 'buy-deed');
    const next = applyKnownLegalAction(state, buy);

    expect(next.players[0].resources.Moons).toBe(0);
    expect(next.players[0].resources.Waves).toBe(0);
    expect(next.players[0].resources.Knots).toBe(0);
    const district = next.districts.find((item) => item.id === buy.districtId);
    expect(district?.stacks[PLAYER_A].deed).toEqual({
      cardId: CONSUL_ID,
      progress: 0,
      tokens: {},
    });
  });

  it('develops a Court outright for ten tokens', () => {
    const state = makeGameState({
      districts: [
        makeDistrict('D1', ['Moons']),
        makeDistrict('D2', ['Suns']),
        makeDistrict('D3', ['Waves']),
        makeDistrict('D4', ['Leaves']),
        makeDistrict('D5', []),
      ],
      players: [
        makePlayer(PLAYER_A, {
          hand: [CONSUL_ID],
          resources: makeResources({ Moons: 8, Waves: 1, Knots: 1 }),
        }),
        makePlayer('PlayerB'),
      ] as const,
      phase: 'ActionWindow',
    });

    const develop = legalActions(state).find(
      (action) =>
        action.type === 'develop-outright' &&
        action.cardId === CONSUL_ID &&
        action.districtId === 'D1'
    );
    expect(develop).toBeDefined();
    const next = applyKnownLegalAction(state, develop!);

    const district = next.districts.find((item) => item.id === 'D1');
    expect(district?.stacks[PLAYER_A].developed).toEqual([CONSUL_ID]);
    expect(next.players[0].resources).toMatchObject({
      Moons: 0,
      Waves: 0,
      Knots: 0,
    });
  });

  it('completes a Court deed at ten developed tokens', () => {
    const base = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 1 }) }),
        makePlayer('PlayerB'),
      ] as const,
      phase: 'ActionWindow',
    });
    const state = withDeed(base, 'D1', PLAYER_A, {
      cardId: CONSUL_ID,
      progress: 9,
      tokens: { Moons: 5, Waves: 2, Knots: 2 },
    });

    const develop = findLegalActionByType(state, 'develop-deed');
    const next = applyKnownLegalAction(state, develop);

    const district = next.districts.find((item) => item.id === 'D1');
    expect(district?.stacks[PLAYER_A].developed).toEqual([CONSUL_ID]);
    expect(district?.stacks[PLAYER_A].deed).toBeUndefined();
  });
});

describe('extended rules: income and victory', () => {
  function stateWithDevelopedCourt(): GameState {
    return makeGameState({
      districts: [
        makeDistrict('D1', ['Moons'], {
          PlayerA: { developed: [CONSUL_ID] },
        }),
        makeDistrict('D2', ['Suns']),
        makeDistrict('D3', ['Waves']),
        makeDistrict('D4', ['Leaves']),
        makeDistrict('D5', []),
      ],
      players: [makePlayer(PLAYER_A), makePlayer('PlayerB')] as const,
      phase: 'ActionWindow',
    });
  }

  it('never provides rank or Ace income', () => {
    const state = stateWithDevelopedCourt();
    for (let result = 1; result <= 9; result += 1) {
      const income = incomeForResult(
        state,
        PLAYER_A,
        result as Parameters<typeof incomeForResult>[2]
      );
      expect(income.fixedDelta).toEqual({});
      expect(income.pendingChoices).toEqual([]);
    }
  });

  it('still collects Crown income on a ten', () => {
    const state = stateWithDevelopedCourt();
    const crowns = state.players[0].crowns;
    const expected: Record<string, number> = {};
    for (const crownId of crowns) {
      const crown = CARD_BY_ID[crownId];
      if (crown.kind === 'Crown') {
        expected[crown.suits[0]] = (expected[crown.suits[0]] ?? 0) + 1;
      }
    }
    expect(incomeForResult(state, PLAYER_A, 10).fixedDelta).toEqual(expected);
  });

  it('counts a developed Court as rank 10 for scoring', () => {
    const state = stateWithDevelopedCourt();
    const district = state.districts.find((item) => item.id === 'D1')!;
    // Consul (10) + the default Crowns are not properties, so score is 10.
    expect(districtScore(district.stacks[PLAYER_A])).toBe(10);
    expect(scoreGame(state).rankTotals[PLAYER_A]).toBe(10);
  });

  it('counts a Court toward the Ace suit bonus', () => {
    const state = makeGameState({
      districts: [
        makeDistrict('D1', ['Moons'], {
          PlayerA: { developed: ['2', CONSUL_ID] },
        }),
        makeDistrict('D2', ['Suns']),
        makeDistrict('D3', ['Waves']),
        makeDistrict('D4', ['Leaves']),
        makeDistrict('D5', []),
      ],
      phase: 'ActionWindow',
    });
    const district = state.districts.find((item) => item.id === 'D1')!;
    // Ace of Moons (1) + Court (10) + 1 Ace bonus for the Court's Moons suit.
    expect(districtScore(district.stacks[PLAYER_A])).toBe(12);
  });
});

describe('extended rules: determinism', () => {
  it('replays the same extended game from the same seed', () => {
    const first = newGame('extended-determinism', { ruleset: 'extended' });
    const second = newGame('extended-determinism', { ruleset: 'extended' });
    expect(first).toEqual(second);
  });

  it('keeps the standard deck composition independent of Courts', () => {
    const state = newGame('standard-composition');
    expect(state.deck.draw).toHaveLength(PROPERTY_CARDS.length - 6);
    for (const cardId of state.deck.draw) {
      expect(COURT_CARDS.some((court) => court.id === cardId)).toBe(false);
    }
  });
});
