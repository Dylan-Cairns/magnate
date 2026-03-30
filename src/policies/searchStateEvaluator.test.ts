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
import { scoreGame } from '../engine/scoring';
import type {
  DeckState,
  DistrictStack,
  DistrictState,
  GameState,
  PlayerId,
  ResourcePool,
} from '../engine/types';
import {
  evaluateSearchLeafState,
  evaluateSearchTerminalState,
} from './searchStateEvaluator';

describe('search state evaluator', () => {
  it('uses ace bonus when evaluating district control', () => {
    const withAceBonus = makeEvalState({
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['0', '6'] }),
          [PLAYER_B]: stack({ developed: ['10'] }),
        },
      }),
    });
    const withoutAceBonus = makeEvalState({
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['0'] }),
          [PLAYER_B]: stack({ developed: ['10'] }),
        },
      }),
    });

    expect(evaluateSearchLeafState(withAceBonus, PLAYER_A)).toBeGreaterThan(0);
    expect(evaluateSearchLeafState(withAceBonus, PLAYER_A)).toBeGreaterThan(
      evaluateSearchLeafState(withoutAceBonus, PLAYER_A)
    );
  });

  it('values district control more than resource surplus alone', () => {
    const districtControl = makeEvalState({
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['6'] }),
          [PLAYER_B]: stack(),
        },
      }),
    });
    const resourceSurplus = makeEvalState({
      resourcesA: makeResources({
        Moons: 6,
        Suns: 6,
        Waves: 6,
        Leaves: 6,
        Wyrms: 6,
        Knots: 6,
      }),
    });

    expect(evaluateSearchLeafState(districtControl, PLAYER_A)).toBeGreaterThan(
      evaluateSearchLeafState(resourceSurplus, PLAYER_A)
    );
  });

  it('treats close opponent deed threats as more dangerous than distant threats', () => {
    const farThreat = makeEvalState({
      resourcesB: makeResources({ Suns: 1 }),
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['15'] }),
          [PLAYER_B]: stack({
            developed: ['6'],
            deed: { cardId: '21', progress: 1 },
          }),
        },
      }),
    });
    const closeThreat = makeEvalState({
      resourcesB: makeResources({ Suns: 1 }),
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['15'] }),
          [PLAYER_B]: stack({
            developed: ['6'],
            deed: { cardId: '21', progress: 6 },
          }),
        },
      }),
    });

    expect(evaluateSearchLeafState(closeThreat, PLAYER_A)).toBeLessThan(
      evaluateSearchLeafState(farThreat, PLAYER_A)
    );
  });

  it('prefers completed late-game control over speculative incomplete deed potential', () => {
    const completedControl = makeEvalState({
      finalTurnsRemaining: 1,
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({ developed: ['15'] }),
          [PLAYER_B]: stack({ developed: ['6'] }),
        },
      }),
    });
    const speculativeDeed = makeEvalState({
      finalTurnsRemaining: 1,
      districts: withDistrictStacks({
        D1: {
          [PLAYER_A]: stack({
            deed: { cardId: '29', progress: 8 },
          }),
          [PLAYER_B]: stack({ developed: ['6'] }),
        },
      }),
    });

    expect(evaluateSearchLeafState(completedControl, PLAYER_A)).toBeGreaterThan(
      evaluateSearchLeafState(speculativeDeed, PLAYER_A)
    );
  });

  it('prefers suit coverage over equivalent resources concentrated in one suit', () => {
    const state = makeEvalState({
      resourcesA: makeResources({
        Moons: 1,
        Suns: 1,
        Waves: 1,
        Leaves: 1,
        Wyrms: 1,
        Knots: 1,
      }),
      resourcesB: makeResources({ Moons: 6 }),
    });

    expect(evaluateSearchLeafState(state, PLAYER_A)).toBeGreaterThan(0);
  });

  it('values resource banks that match contextual suit demand', () => {
    const state = makeEvalState({
      handA: ['24'],
      handB: ['24'],
      resourcesA: makeResources({ Wyrms: 1, Knots: 1 }),
      resourcesB: makeResources({ Suns: 1, Leaves: 1 }),
    });

    expect(evaluateSearchLeafState(state, PLAYER_A)).toBeGreaterThan(0);
  });

  it('does not use hand size as a leaf-value signal', () => {
    const rootLargeHand = makeEvalState({
      handA: ['6', '7', '8'],
      handB: [],
    });
    const opponentLargeHand = makeEvalState({
      handA: [],
      handB: ['6', '7', '8'],
    });

    expect(evaluateSearchLeafState(rootLargeHand, PLAYER_A)).toBe(
      evaluateSearchLeafState(opponentLargeHand, PLAYER_A)
    );
  });

  it('keeps terminal wins above draws and losses', () => {
    const narrowWin = terminalState(
      makeEvalState({
        districts: withDistrictStacks({
          D1: {
            [PLAYER_A]: stack({ developed: ['0', '6'] }),
            [PLAYER_B]: stack({ developed: ['10'] }),
          },
        }),
      })
    );
    const draw = terminalState(makeEvalState());
    const loss = terminalState(
      makeEvalState({
        districts: withDistrictStacks({
          D1: {
            [PLAYER_A]: stack(),
            [PLAYER_B]: stack({ developed: ['29'] }),
          },
        }),
      })
    );

    expect(evaluateSearchTerminalState(narrowWin, PLAYER_A)).toBeGreaterThan(
      0
    );
    expect(evaluateSearchTerminalState(draw, PLAYER_A)).toBe(0);
    expect(evaluateSearchTerminalState(loss, PLAYER_A)).toBeLessThan(0);
  });

  it('rewards wider terminal margins without changing the outcome sign', () => {
    const narrowWin = terminalState(
      makeEvalState({
        districts: withDistrictStacks({
          D1: {
            [PLAYER_A]: stack({ developed: ['0', '6'] }),
            [PLAYER_B]: stack({ developed: ['10'] }),
          },
        }),
      })
    );
    const wideWin = terminalState(
      makeEvalState({
        districts: withDistrictStacks({
          D1: { [PLAYER_A]: stack({ developed: ['29'] }) },
          D2: { [PLAYER_A]: stack({ developed: ['28'] }) },
          D3: { [PLAYER_A]: stack({ developed: ['27'] }) },
        }),
      })
    );

    expect(evaluateSearchTerminalState(wideWin, PLAYER_A)).toBeGreaterThan(
      evaluateSearchTerminalState(narrowWin, PLAYER_A)
    );
    expect(evaluateSearchTerminalState(narrowWin, PLAYER_B)).toBeLessThan(0);
  });

  it('uses final-score tiebreak margins for terminal bonuses', () => {
    const rankTiebreakWin = terminalState(
      makeEvalState({
        districts: withDistrictStacks({
          D1: {
            [PLAYER_A]: stack({ developed: ['15'] }),
            [PLAYER_B]: stack(),
          },
          D2: {
            [PLAYER_A]: stack(),
            [PLAYER_B]: stack({ developed: ['6'] }),
          },
        }),
      })
    );
    const resourceTiebreakWin = terminalState(
      makeEvalState({
        resourcesA: makeResources({ Moons: 3 }),
        resourcesB: makeResources({ Moons: 1 }),
      })
    );

    expect(rankTiebreakWin.finalScore?.decidedBy).toBe('rank-total');
    expect(resourceTiebreakWin.finalScore?.decidedBy).toBe('resources');
    expect(
      evaluateSearchTerminalState(rankTiebreakWin, PLAYER_A)
    ).toBeGreaterThan(0.72);
    expect(
      evaluateSearchTerminalState(resourceTiebreakWin, PLAYER_A)
    ).toBeGreaterThan(0.72);
  });
});

function makeEvalState({
  districts = makeDefaultDistricts(),
  resourcesA = makeResources(),
  resourcesB = makeResources(),
  handA = [],
  handB = [],
  finalTurnsRemaining,
  deck,
}: {
  districts?: DistrictState[];
  resourcesA?: ResourcePool;
  resourcesB?: ResourcePool;
  handA?: string[];
  handB?: string[];
  finalTurnsRemaining?: number;
  deck?: DeckState;
} = {}): GameState {
  return makeGameState({
    districts,
    deck,
    finalTurnsRemaining,
    players: [
      makePlayer(PLAYER_A, {
        crowns: [],
        hand: handA.map(asCardId),
        resources: resourcesA,
      }),
      makePlayer(PLAYER_B, {
        crowns: [],
        hand: handB.map(asCardId),
        resources: resourcesB,
      }),
    ] as const,
  });
}

function withDistrictStacks(
  overrides: Partial<Record<string, Partial<Record<PlayerId, DistrictStack>>>>
): DistrictState[] {
  return makeDefaultDistricts().map((district) => ({
    ...district,
    stacks: {
      ...district.stacks,
      ...(overrides[district.id] ?? {}),
    },
  }));
}

function stack({
  developed = [],
  deed,
}: {
  developed?: string[];
  deed?: { cardId: string; progress: number };
} = {}): DistrictStack {
  return {
    developed: developed.map(asCardId),
    deed: deed
      ? {
          cardId: asCardId(deed.cardId),
          progress: deed.progress,
          tokens: {},
        }
      : undefined,
  };
}

function terminalState(state: GameState): GameState {
  const terminal = {
    ...state,
    phase: 'GameOver' as const,
  };
  return {
    ...terminal,
    finalScore: scoreGame(terminal),
  };
}
