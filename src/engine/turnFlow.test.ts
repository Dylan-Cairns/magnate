import { describe, expect, it } from 'vitest';

import {
  makeDefaultDistricts,
  PLAYER_A,
  PLAYER_B,
  makeGameState,
  makePlayer,
  makeResources,
} from './__tests__/fixtures';
import { advanceToDecision } from './turnFlow';

describe('advanceToDecision', () => {
  it('returns the same state when already at a decision phase', () => {
    const state = makeGameState({ phase: 'PlayCard' });
    expect(advanceToDecision(state)).toBe(state);
  });

  it('auto-advances StartTurn through CollectIncome into OptionalTrade', () => {
    const state = makeGameState({
      phase: 'StartTurn',
      turn: 5,
      activePlayerIndex: 1,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(5);
    expect(advanced.activePlayerIndex).toBe(1);
  });

  it('resolves DrawCard by drawing to active player hand and handing off turn', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 1,
      activePlayerIndex: 0,
      players: [
        makePlayer(PLAYER_A, { hand: ['6'] }),
        makePlayer(PLAYER_B, { hand: ['7'] }),
      ],
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(2);
    expect(advanced.activePlayerIndex).toBe(1);
    expect(advanced.players[0].hand).toEqual(['6', '6']);
    expect(advanced.deck.draw).toEqual(['7', '8']);
  });

  it('on first second-exhaustion draw, enters final-turn countdown and continues', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 10,
      activePlayerIndex: 0,
      deck: { draw: [], discard: [], reshuffles: 1 },
      exhaustionStage: 1,
      finalTurnsRemaining: undefined,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('OptionalTrade');
    expect(advanced.turn).toBe(11);
    expect(advanced.activePlayerIndex).toBe(1);
    expect(advanced.exhaustionStage).toBe(2);
    expect(advanced.finalTurnsRemaining).toBe(1);
  });

  it('ends the game when final-turn countdown reaches zero on DrawCard', () => {
    const state = makeGameState({
      phase: 'DrawCard',
      turn: 11,
      activePlayerIndex: 1,
      deck: { draw: [], discard: [], reshuffles: 2 },
      exhaustionStage: 2,
      finalTurnsRemaining: 1,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.phase).toBe('GameOver');
    expect(advanced.turn).toBe(12);
    expect(advanced.activePlayerIndex).toBe(0);
    expect(advanced.finalTurnsRemaining).toBe(0);
  });
});

describe('tax and income resolution', () => {
  it('TaxCheck records lastIncomeRoll and advances rng cursor deterministically', () => {
    const startCursor = 4;
    const state = makeGameState({
      phase: 'TaxCheck',
      seed: 'tax-roll-seed',
      rngCursor: startCursor,
    });

    const advanced = advanceToDecision(state);
    expect(advanced.lastIncomeRoll).toBeDefined();
    const roll = advanced.lastIncomeRoll;
    if (!roll) {
      throw new Error('Missing expected income roll.');
    }
    const taxTriggered = roll.die1 === 1 || roll.die2 === 1;
    expect(advanced.rngCursor).toBe(startCursor + (taxTriggered ? 3 : 2));
  });

  it('TaxCheck taxation caps taxed suit to one token for both players', () => {
    const seed = findSeedWithTaxTrigger();
    const baseResources = makeResources({
      Moons: 5,
      Suns: 5,
      Waves: 5,
      Leaves: 5,
      Wyrms: 5,
      Knots: 5,
    });

    const state = makeGameState({
      phase: 'TaxCheck',
      seed,
      rngCursor: 0,
      players: [
        makePlayer(PLAYER_A, { resources: baseResources, crowns: [] }),
        makePlayer(PLAYER_B, { resources: baseResources, crowns: [] }),
      ],
      districts: makeDefaultDistricts(),
    });

    const advanced = advanceToDecision(state);
    expect(advanced.lastIncomeRoll?.die1 === 1 || advanced.lastIncomeRoll?.die2 === 1).toBe(
      true
    );

    advanced.players.forEach((player) => {
      const values = Object.values(player.resources);
      expect(values.filter((value) => value === 1)).toHaveLength(1);
      expect(values.filter((value) => value === 5)).toHaveLength(5);
    });
  });

  it('CollectIncome on 10 pays one token per crown suit', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      lastIncomeRoll: { die1: 10, die2: 7 },
      players: [
        makePlayer(PLAYER_A, { resources: makeResources(), crowns: ['30', '31', '32'] }),
        makePlayer(PLAYER_B, { resources: makeResources(), crowns: ['30', '31', '32'] }),
      ],
    });

    const advanced = advanceToDecision(state);
    const playerA = advanced.players.find((player) => player.id === PLAYER_A);
    const playerB = advanced.players.find((player) => player.id === PLAYER_B);
    if (!playerA || !playerB) {
      throw new Error('Missing expected players.');
    }

    [playerA, playerB].forEach((player) => {
      expect(player.resources.Knots).toBe(1);
      expect(player.resources.Leaves).toBe(1);
      expect(player.resources.Moons).toBe(1);
    });
  });

  it('CollectIncome on 2-9 pays developed suits and one deterministic deed suit', () => {
    const districts = makeDefaultDistricts().map((district) => {
      if (district.id !== 'D1') {
        return district;
      }
      return {
        ...district,
        stacks: {
          ...district.stacks,
          [PLAYER_A]: {
            developed: ['6'],
            deed: { cardId: '7', progress: 0, tokens: {} },
          },
        },
      };
    });

    const state = makeGameState({
      phase: 'CollectIncome',
      lastIncomeRoll: { die1: 2, die2: 2 },
      districts,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources(), crowns: [] }),
        makePlayer(PLAYER_B, { resources: makeResources(), crowns: [] }),
      ],
    });

    const advanced = advanceToDecision(state);
    const playerA = advanced.players.find((player) => player.id === PLAYER_A);
    if (!playerA) {
      throw new Error('Missing PlayerA.');
    }

    expect(playerA.resources.Moons).toBe(1);
    expect(playerA.resources.Knots).toBe(1);
    expect(playerA.resources.Suns).toBe(1);
    expect(playerA.resources.Wyrms).toBe(0);
  });

  it('CollectIncome on double ones pays ace income for ace properties in play', () => {
    const districts = makeDefaultDistricts().map((district) => {
      if (district.id !== 'D1') {
        return district;
      }
      return {
        ...district,
        stacks: {
          ...district.stacks,
          [PLAYER_A]: {
            developed: ['0'],
            deed: { cardId: '2', progress: 0, tokens: {} },
          },
        },
      };
    });

    const state = makeGameState({
      phase: 'CollectIncome',
      lastIncomeRoll: { die1: 1, die2: 1 },
      districts,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources(), crowns: [] }),
        makePlayer(PLAYER_B, { resources: makeResources(), crowns: [] }),
      ],
    });

    const advanced = advanceToDecision(state);
    const playerA = advanced.players.find((player) => player.id === PLAYER_A);
    if (!playerA) {
      throw new Error('Missing PlayerA.');
    }

    expect(playerA.resources.Knots).toBe(1);
    expect(playerA.resources.Moons).toBe(1);
  });

  it('CollectIncome throws when lastIncomeRoll is missing', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
    });
    expect(() => advanceToDecision(state)).toThrow(
      'CollectIncome phase requires lastIncomeRoll to be present.'
    );
  });
});

function findSeedWithTaxTrigger(): string {
  for (let i = 0; i < 5000; i += 1) {
    const seed = `tax-trigger-${i}`;
    const state = makeGameState({
      phase: 'TaxCheck',
      seed,
      rngCursor: 0,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources(), crowns: [] }),
        makePlayer(PLAYER_B, { resources: makeResources(), crowns: [] }),
      ],
    });

    const advanced = advanceToDecision(state);
    if (advanced.lastIncomeRoll?.die1 === 1 || advanced.lastIncomeRoll?.die2 === 1) {
      return seed;
    }
  }

  throw new Error('Failed to find a deterministic seed that triggers taxation.');
}
