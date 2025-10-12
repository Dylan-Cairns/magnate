import { CARD_BY_ID } from './cards';
import type { CardId } from './cards';
import { initialSetup } from './deck';
import type { DistrictState, GameState, PlayerId, PlayerState, ResourcePool } from './types';

const PLAYER_IDS: readonly [PlayerId, PlayerId] = ['PlayerA', 'PlayerB'];

interface NewGameOptions {
  firstPlayer?: PlayerId;
}

export function newGame(seed: string, options: NewGameOptions = {}): GameState {
  const setup = initialSetup(seed);
  const firstPlayer = options.firstPlayer ?? 'PlayerA';

  const players: readonly [PlayerState, PlayerState] = [
    createPlayerState('PlayerA', setup.handsByPlayer.PlayerA, setup.crownsByPlayer.PlayerA, setup.startingResourcesByPlayer.PlayerA),
    createPlayerState('PlayerB', setup.handsByPlayer.PlayerB, setup.crownsByPlayer.PlayerB, setup.startingResourcesByPlayer.PlayerB),
  ];

  const districts = setup.districts.map((markerCardId, index) =>
    districtFromMarker(markerCardId, `D${index + 1}`)
  );

  return {
    schemaVersion: 1,
    seed,
    rngCursor: 0,
    deck: {
      draw: [...setup.deck.draw],
      discard: [...setup.deck.discard],
      reshuffles: setup.deck.reshuffles,
    },
    players,
    activePlayerIndex: playerIndexFor(firstPlayer),
    turn: 1,
    phase: 'StartTurn',
    districts,
    cardPlayedThisTurn: false,
    exhaustionStage: setup.deck.reshuffles,
    finalTurnsRemaining: undefined,
    lastIncomeRoll: undefined,
    lastTaxSuit: undefined,
    pendingIncomeChoices: undefined,
    incomeChoiceReturnPlayerIndex: undefined,
    finalScore: undefined,
    log: [],
  };
}

function createPlayerState(
  id: PlayerId,
  hand: readonly [CardId, CardId, CardId],
  crowns: readonly [CardId, CardId, CardId],
  resources: ResourcePool
): PlayerState {
  return {
    id,
    hand: [...hand],
    crowns: [...crowns],
    resources: cloneResources(resources),
  };
}

function cloneResources(resources: ResourcePool): ResourcePool {
  return {
    Moons: resources.Moons,
    Suns: resources.Suns,
    Waves: resources.Waves,
    Leaves: resources.Leaves,
    Wyrms: resources.Wyrms,
    Knots: resources.Knots,
  };
}

function districtFromMarker(markerCardId: CardId, districtId: string): DistrictState {
  const marker = CARD_BY_ID[markerCardId];
  if (marker.kind === 'Pawn') {
    return {
      id: districtId,
      markerSuitMask: [...marker.suits],
      stacks: {
        PlayerA: { developed: [] },
        PlayerB: { developed: [] },
      },
    };
  }
  if (marker.kind === 'Excuse') {
    return {
      id: districtId,
      markerSuitMask: [],
      stacks: {
        PlayerA: { developed: [] },
        PlayerB: { developed: [] },
      },
    };
  }
  throw new Error(`Invalid district marker card kind: ${marker.kind}`);
}

function playerIndexFor(playerId: PlayerId): number {
  return PLAYER_IDS.indexOf(playerId);
}
