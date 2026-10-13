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
    const exhaustedSecondDeck =
      deck.reshuffles === 1 && topDraw.deck.draw.length === 0;
    return {
      cardId: topDraw.cardId,
      deck: exhaustedSecondDeck
        ? { ...topDraw.deck, reshuffles: 2 }
        : topDraw.deck,
      rngCursor,
      finalTurnsRemaining: exhaustedSecondDeck
        ? (finalTurnsRemaining ?? 2)
        : finalTurnsRemaining,
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
