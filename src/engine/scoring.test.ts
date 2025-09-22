import { describe, expect, it } from 'vitest';

import {
  asCardId,
  makeDefaultDistricts,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from './__tests__/fixtures';
import { isTerminal, scoreGame } from './scoring';

describe('isTerminal', () => {
  it('returns true only for GameOver phase', () => {
    expect(isTerminal(makeGameState({ phase: 'GameOver' }))).toBe(true);
    expect(isTerminal(makeGameState({ phase: 'PlayCard' }))).toBe(false);
  });
});

describe('scoreGame', () => {
  it('scores district points with ace district bonus and district winner', () => {
    const districts = withStacks('D1', {
      [PLAYER_A]: { developed: ['0', '6'] },
      [PLAYER_B]: { developed: ['7'] },
    });
    const state = makeGameState({
      phase: 'GameOver',
      districts,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });

    const score = scoreGame(state);
    expect(score.districtPoints).toEqual({ PlayerA: 1, PlayerB: 0 });
    expect(score.winner).toBe('PlayerA');
    expect(score.decidedBy).toBe('districts');
  });

  it('uses rank-total tiebreaker when district points tie', () => {
    const districts = withStacks('D1', {
      [PLAYER_A]: { developed: ['0', '6'] },
      [PLAYER_B]: { developed: ['15'] },
    });
    const state = makeGameState({
      phase: 'GameOver',
      districts,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });

    const score = scoreGame(state);
    expect(score.districtPoints).toEqual({ PlayerA: 0, PlayerB: 0 });
    expect(score.rankTotals).toEqual({ PlayerA: 3, PlayerB: 5 });
    expect(score.winner).toBe('PlayerB');
    expect(score.decidedBy).toBe('rank-total');
  });

  it('uses resource tiebreaker when districts and rank totals tie', () => {
    const districts = withStacks('D1', {
      [PLAYER_A]: { developed: ['6'] },
      [PLAYER_B]: { developed: ['6'] },
    });
    const state = makeGameState({
      phase: 'GameOver',
      districts,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Moons: 1 }) }),
      ] as const,
    });

    const score = scoreGame(state);
    expect(score.districtPoints).toEqual({ PlayerA: 0, PlayerB: 0 });
    expect(score.rankTotals).toEqual({ PlayerA: 2, PlayerB: 2 });
    expect(score.winner).toBe('PlayerA');
    expect(score.decidedBy).toBe('resources');
  });

  it('returns draw when all tiebreakers are equal', () => {
    const districts = withStacks('D1', {
      [PLAYER_A]: { developed: ['6'] },
      [PLAYER_B]: { developed: ['6'] },
    });
    const state = makeGameState({
      phase: 'GameOver',
      districts,
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 1 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Moons: 1 }) }),
      ] as const,
    });

    const score = scoreGame(state);
    expect(score.winner).toBe('Draw');
    expect(score.decidedBy).toBe('draw');
  });

  it('ignores incomplete deeds and deed tokens for final scoring', () => {
    const districts = withStacks('D1', {
      [PLAYER_A]: {
        developed: ['6'],
        deed: { cardId: '29', progress: 0, tokens: {} },
      },
      [PLAYER_B]: { developed: ['7'] },
    });
    const state = makeGameState({
      phase: 'GameOver',
      districts,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });

    const score = scoreGame(state);
    expect(score.rankTotals).toEqual({ PlayerA: 2, PlayerB: 2 });
    expect(score.districtPoints).toEqual({ PlayerA: 0, PlayerB: 0 });
  });
});

function withStacks(
  districtId: string,
  stacks: {
    PlayerA: {
      developed: string[];
      deed?: { cardId: string; progress: number; tokens: Record<string, number> };
    };
    PlayerB: {
      developed: string[];
      deed?: { cardId: string; progress: number; tokens: Record<string, number> };
    };
  }
) {
  return makeDefaultDistricts().map((district) => {
    if (district.id !== districtId) {
      return district;
    }
    return {
      ...district,
      stacks: {
        ...district.stacks,
        [PLAYER_A]: {
          developed: stacks.PlayerA.developed.map(asCardId),
          deed: stacks.PlayerA.deed
            ? {
                cardId: asCardId(stacks.PlayerA.deed.cardId),
                progress: stacks.PlayerA.deed.progress,
                tokens: stacks.PlayerA.deed.tokens,
              }
            : undefined,
        },
        [PLAYER_B]: {
          developed: stacks.PlayerB.developed.map(asCardId),
          deed: stacks.PlayerB.deed
            ? {
                cardId: asCardId(stacks.PlayerB.deed.cardId),
                progress: stacks.PlayerB.deed.progress,
                tokens: stacks.PlayerB.deed.tokens,
              }
            : undefined,
        },
      },
    };
  });
}
