import type { CardId } from './cards';
import { PROPERTY_CARDS, CROWN_CARDS, PAWN_CARDS, EXCUSE_CARD } from './cards';
import type { DeckState } from './types';

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
  crowns: CardId[];
  districts: CardId[]; // marker card IDs: four Pawns + the Excuse
}

export function initialSetup(seed: string): SetupResult {
  const rand = rngFromSeed(seed);

  const draw = PROPERTY_CARDS.map((c) => c.id);
  shuffleInPlace(draw, rand);

  const deck: DeckState = { draw, discard: [], reshuffles: 0 };

  const crowns = CROWN_CARDS.map((c) => c.id);

  const districts: CardId[] = [
    ...PAWN_CARDS.map((p) => p.id),
    EXCUSE_CARD.id,
  ];

  return { deck, crowns, districts };
}

export function drawOne(deck: DeckState): { cardId?: CardId; deck: DeckState } {
  if (deck.draw.length === 0) {
    if (deck.reshuffles >= 2) {
      return { cardId: undefined, deck };
    }
    const rand = rngFromSeed(
      `reshuffle:${deck.discard.length}:${deck.reshuffles}`
    );
    const newDraw = [...deck.discard];
    shuffleInPlace(newDraw, rand);
    return drawOne({
      draw: newDraw,
      discard: [],
      reshuffles: (deck.reshuffles + 1) as 1 | 2,
    });
  }
  const [cardId, ...rest] = deck.draw;
  return { cardId, deck: { ...deck, draw: rest } };
}

export function discard(deck: DeckState, cardId: CardId): DeckState {
  return { ...deck, discard: [cardId, ...deck.discard] };
}
