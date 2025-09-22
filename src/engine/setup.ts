import type { CardId } from './cards';
import {
  CARD_BY_ID,
  CROWN_CARDS,
  EXCUSE_CARD,
  PAWN_CARDS,
  PROPERTY_CARDS,
} from './cards';
import { createDeck } from './deckCore';
import { rngFromSeed, shuffleInPlace } from './rng';
import type { DeckState, PlayerId, ResourcePool } from './types';

export interface SetupResult {
  deck: DeckState;
  crownsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]>;
  handsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]>;
  startingResourcesByPlayer: Record<PlayerId, ResourcePool>;
  districts: CardId[]; // marker card IDs: four Pawns + the Excuse
}

export function initialSetup(seed: string): SetupResult {
  const rand = rngFromSeed(seed);

  const draw = PROPERTY_CARDS.map((card) => card.id);
  shuffleInPlace(draw, rand);

  const crowns = CROWN_CARDS.map((card) => card.id);
  shuffleInPlace(crowns, rand);

  const playerA: PlayerId = 'PlayerA';
  const playerB: PlayerId = 'PlayerB';

  const dealMany = (pile: CardId[], count: number): CardId[] => {
    if (pile.length < count) {
      throw new Error(`Not enough cards to deal ${count}.`);
    }
    return pile.splice(0, count);
  };

  const toTriple = (cards: CardId[]): readonly [CardId, CardId, CardId] => {
    if (cards.length !== 3) {
      throw new Error('Expected exactly 3 cards.');
    }
    return [cards[0], cards[1], cards[2]];
  };

  const crownsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]> = {
    [playerA]: toTriple(dealMany(crowns, 3)),
    [playerB]: toTriple(dealMany(crowns, 3)),
  };

  const handsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]> = {
    [playerA]: toTriple(dealMany(draw, 3)),
    [playerB]: toTriple(dealMany(draw, 3)),
  };

  const createEmptyPool = (): ResourcePool => ({
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  });

  const resourcesFromCrowns = (
    crownIds: readonly [CardId, CardId, CardId]
  ): ResourcePool => {
    const pool = createEmptyPool();
    crownIds.forEach((cardId) => {
      const card = CARD_BY_ID[cardId];
      if (card.kind !== 'Crown') {
        throw new Error(`Expected crown card during setup, got ${card.kind}.`);
      }
      const suit = card.suits[0];
      pool[suit] += 1;
    });
    return pool;
  };

  const startingResourcesByPlayer: Record<PlayerId, ResourcePool> = {
    [playerA]: resourcesFromCrowns(crownsByPlayer[playerA]),
    [playerB]: resourcesFromCrowns(crownsByPlayer[playerB]),
  };

  const deck = createDeck(draw);

  const shuffledPawns = PAWN_CARDS.map((pawn) => pawn.id);
  shuffleInPlace(shuffledPawns, rand);
  const districts: CardId[] = [
    ...shuffledPawns.slice(0, 2),
    EXCUSE_CARD.id,
    ...shuffledPawns.slice(2),
  ];

  return {
    deck,
    crownsByPlayer,
    handsByPlayer,
    startingResourcesByPlayer,
    districts,
  };
}
