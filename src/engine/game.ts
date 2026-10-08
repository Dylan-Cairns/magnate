import { GamePhase, Suit, CardKind } from './values';
import { CARD_BY_ID } from './cards';
import type { CardId } from './cards';
import { initialSetup } from './deck';
import {
  type DistrictState,
  type GameState,
  PlayerId,
  type PlayerState,
  type ResourcePool,
  Ruleset,
} from './types';

import { PLAYER_IDS } from './values';

export interface NewGameOptions {
  firstPlayer?: PlayerId;
  ruleset?: Ruleset;
}

export function newGame(seed: string, options: NewGameOptions = {}): GameState {
  const ruleset = options.ruleset ?? Ruleset.Standard;
  const setup = initialSetup(seed, ruleset);
  const firstPlayer = options.firstPlayer ?? PlayerId.PlayerA;

  const players: readonly [PlayerState, PlayerState] = [
    createPlayerState(
      PlayerId.PlayerA,
      setup.handsByPlayer[PlayerId.PlayerA],
      setup.crownsByPlayer[PlayerId.PlayerA],
      setup.startingResourcesByPlayer[PlayerId.PlayerA]
    ),
    createPlayerState(
      PlayerId.PlayerB,
      setup.handsByPlayer[PlayerId.PlayerB],
      setup.crownsByPlayer[PlayerId.PlayerB],
      setup.startingResourcesByPlayer[PlayerId.PlayerB]
    ),
  ];

  const districts = setup.districts.map((markerCardId, index) =>
    districtFromMarker(markerCardId, `D${index + 1}`)
  );

  return {
    schemaVersion: 1,
    seed,
    rngCursor: 0,
    ruleset,
    deck: {
      draw: [...setup.deck.draw],
      discard: [...setup.deck.discard],
      reshuffles: setup.deck.reshuffles,
    },
    players,
    activePlayerIndex: playerIndexFor(firstPlayer),
    turn: 1,
    phase: GamePhase.StartTurn,
    districts,
    cardPlayedThisTurn: false,
    finalTurnsRemaining: undefined,
    lastIncomeRoll: undefined,
    lastTaxSuit: undefined,
    pendingIncomeChoices: undefined,
    submittedIncomeChoices: undefined,
    incomeChoiceReturnPlayerId: undefined,
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
    [Suit.Moons]: resources[Suit.Moons],
    [Suit.Suns]: resources[Suit.Suns],
    [Suit.Waves]: resources[Suit.Waves],
    [Suit.Leaves]: resources[Suit.Leaves],
    [Suit.Wyrms]: resources[Suit.Wyrms],
    [Suit.Knots]: resources[Suit.Knots],
  };
}

function districtFromMarker(
  markerCardId: CardId,
  districtId: string
): DistrictState {
  const marker = CARD_BY_ID[markerCardId];
  if (marker.kind === CardKind.Pawn) {
    return {
      id: districtId,
      markerSuitMask: [...marker.suits],
      stacks: {
        [PlayerId.PlayerA]: { developed: [] },
        [PlayerId.PlayerB]: { developed: [] },
      },
    };
  }
  if (marker.kind === CardKind.Excuse) {
    return {
      id: districtId,
      markerSuitMask: [],
      stacks: {
        [PlayerId.PlayerA]: { developed: [] },
        [PlayerId.PlayerB]: { developed: [] },
      },
    };
  }
  throw new Error(`Invalid district marker card kind: ${marker.kind}`);
}

function playerIndexFor(playerId: PlayerId): 0 | 1 {
  const index = PLAYER_IDS.indexOf(playerId);
  if (index < 0) {
    throw new Error(`Unknown first player: ${playerId}`);
  }
  return index as 0 | 1;
}
