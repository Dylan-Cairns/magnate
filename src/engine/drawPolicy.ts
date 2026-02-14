import type { CardId } from './cards';
import { drawTop, reshuffleDiscardIntoDraw } from './deckCore';
import { rngFromSeed } from './rng';
import type { DeckState } from './types';

export interface DrawContext {
  deck: DeckState;
  seed: string;
  rngCursor: number;
  finalTurnsRemaining?: number;
}

export interface DrawResult {
  cardId?: CardId;
  deck: DeckState;
  rngCursor: number;
  finalTurnsRemaining?: number;
}

export function drawOne(context: DrawContext): DrawResult {
  const { deck, seed, rngCursor, finalTurnsRemaining } = context;

  const topDraw = drawTop(deck);
  if (topDraw.cardId) {
    return {
      cardId: topDraw.cardId,
      deck: topDraw.deck,
      rngCursor,
      finalTurnsRemaining,
    };
  }

  if (deck.reshuffles === 0 && deck.discard.length > 0) {
    const rand = rngFromSeed(`${seed}:reshuffle:${rngCursor}`);
    const reshuffledDeck = reshuffleDiscardIntoDraw(deck, rand);
    return drawOne({
      deck: reshuffledDeck,
      seed,
      rngCursor: rngCursor + 1,
      finalTurnsRemaining,
    });
  }

  return {
    cardId: undefined,
    deck: {
      ...deck,
      draw: [],
      reshuffles: 2,
    },
    rngCursor,
    finalTurnsRemaining: finalTurnsRemaining ?? 2,
  };
}
