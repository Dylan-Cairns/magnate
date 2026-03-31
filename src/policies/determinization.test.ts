import { describe, expect, it } from 'vitest';

import { rngFromSeed } from '../engine/rng';
import { toPlayerView } from '../engine/view';
import {
  asCardId,
  makeDefaultDistricts,
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../engine/__tests__/fixtures';
import type { CardId } from '../engine/cards';
import type { GameState } from '../engine/types';
import { sampleHiddenWorldStates } from './determinization';

const ROOT_HAND = ['24', '25', '26'].map(asCardId);
const ALL_PROPERTY_IDS = Array.from({ length: 30 }, (_, index) =>
  asCardId(String(index))
);
const HIDDEN_POOL = ALL_PROPERTY_IDS.filter(
  (cardId) => !ROOT_HAND.includes(cardId)
);

describe('determinization', () => {
  it('samples from the public unknown pool instead of true hidden assignment', () => {
    const firstHiddenAssignment = hiddenAssignmentState({
      opponentHand: HIDDEN_POOL.slice(0, 2),
      draw: HIDDEN_POOL.slice(2),
    });
    const secondHiddenAssignment = hiddenAssignmentState({
      opponentHand: HIDDEN_POOL.slice(2, 4),
      draw: [...HIDDEN_POOL.slice(0, 2), ...HIDDEN_POOL.slice(4)],
    });

    const worldsA = sampleHiddenWorldStates({
      state: firstHiddenAssignment,
      view: toPlayerView(firstHiddenAssignment, PLAYER_A),
      rootPlayer: PLAYER_A,
      worldCount: 4,
      random: rngFromSeed('determinization-hidden-assignment'),
      errorPrefix: 'test',
    });
    const worldsB = sampleHiddenWorldStates({
      state: secondHiddenAssignment,
      view: toPlayerView(secondHiddenAssignment, PLAYER_A),
      rootPlayer: PLAYER_A,
      worldCount: 4,
      random: rngFromSeed('determinization-hidden-assignment'),
      errorPrefix: 'test',
    });

    expect(hiddenAssignments(worldsB)).toEqual(hiddenAssignments(worldsA));
  });
});

function hiddenAssignmentState({
  opponentHand,
  draw,
}: {
  opponentHand: readonly CardId[];
  draw: readonly CardId[];
}): GameState {
  return makeGameState({
    turn: 8,
    phase: 'ActionWindow',
    districts: makeDefaultDistricts(),
    deck: {
      draw: [...draw],
      discard: [],
      reshuffles: 0,
    },
    players: [
      makePlayer(PLAYER_A, {
        hand: ROOT_HAND,
        crowns: [],
        resources: makeResources({
          Moons: 4,
          Suns: 4,
          Waves: 4,
          Leaves: 4,
          Wyrms: 4,
          Knots: 4,
        }),
      }),
      makePlayer(PLAYER_B, {
        hand: [...opponentHand],
        crowns: [],
        resources: makeResources(),
      }),
    ] as const,
  });
}

function hiddenAssignments(worlds: readonly GameState[]): {
  opponentHand: readonly CardId[];
  draw: readonly CardId[];
}[] {
  return worlds.map((world) => ({
    opponentHand:
      world.players.find((player) => player.id === PLAYER_B)?.hand ?? [],
    draw: world.deck.draw,
  }));
}
