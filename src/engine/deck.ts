import type { CardId } from './cards';
import {
  PROPERTY_CARDS,
  CROWN_CARDS,
  PAWN_CARDS,
  EXCUSE_CARD,
  CARD_BY_ID,
} from './cards';
import type { DeckState, PlayerId, ResourcePool } from './types';

function xmur3(str: string) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    return (h ^= h >>> 16) >>> 0;
  };
}

function sfc32(a: number, b: number, c: number, d: number) {
  return function () {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

function rngFromSeed(seed: string) {
  const h = xmur3(seed);
  return sfc32(h(), h(), h(), h());
}

function shuffleInPlace<T>(arr: T[], rand: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

export interface SetupResult {
  deck: DeckState;
  crownsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]>;
  handsByPlayer: Record<PlayerId, readonly [CardId, CardId, CardId]>;
  startingResourcesByPlayer: Record<PlayerId, ResourcePool>;
  districts: CardId[]; // marker card IDs: four Pawns + the Excuse
}

export function initialSetup(seed: string): SetupResult {
  const rand = rngFromSeed(seed);

  const draw = PROPERTY_CARDS.map((c) => c.id);
  shuffleInPlace(draw, rand);

  const crowns = CROWN_CARDS.map((c) => c.id);
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

  const deck: DeckState = { draw, discard: [], reshuffles: 0 };

  const districts: CardId[] = [
    ...PAWN_CARDS.map((p) => p.id),
    EXCUSE_CARD.id,
  ];

  return {
    deck,
    crownsByPlayer,
    handsByPlayer,
    startingResourcesByPlayer,
    districts,
  };
}

export interface DrawContext {
  deck: DeckState;
  seed: string;
  rngCursor: number;
  exhaustionStage: 0 | 1 | 2;
  finalTurnsRemaining?: number;
}

export interface DrawResult {
  cardId?: CardId;
  deck: DeckState;
  rngCursor: number;
  exhaustionStage: 0 | 1 | 2;
  finalTurnsRemaining?: number;
}

export function drawOne(context: DrawContext): DrawResult {
  const { deck, seed, rngCursor, exhaustionStage, finalTurnsRemaining } =
    context;

  if (deck.draw.length > 0) {
    const [cardId, ...rest] = deck.draw;
    return {
      cardId,
      deck: { ...deck, draw: rest },
      rngCursor,
      exhaustionStage,
      finalTurnsRemaining,
    };
  }

  if (deck.reshuffles === 0 && deck.discard.length > 0) {
    const rand = rngFromSeed(`${seed}:reshuffle:${rngCursor}`);
    const newDraw = [...deck.discard];
    shuffleInPlace(newDraw, rand);
    return drawOne({
      deck: {
        draw: newDraw,
        discard: [],
        reshuffles: 1,
      },
      seed,
      rngCursor: rngCursor + 1,
      exhaustionStage: 1,
      finalTurnsRemaining,
    });
  }

  return {
    cardId: undefined,
    deck: { ...deck, draw: [], reshuffles: 2 },
    rngCursor,
    exhaustionStage: 2,
    finalTurnsRemaining: finalTurnsRemaining ?? 2,
  };
}

export function discard(deck: DeckState, cardId: CardId): DeckState {
  return { ...deck, discard: [cardId, ...deck.discard] };
}
