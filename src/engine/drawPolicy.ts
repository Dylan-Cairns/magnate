import type { CardId } from './cards';
import { drawTop, reshuffleDiscardIntoDraw } from './deckCore';
import { rngFromSeed } from './rng';
import type { DeckState } from './types';

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

  const topDraw = drawTop(deck);
  if (topDraw.cardId) {
    return {
      cardId: topDraw.cardId,
      deck: topDraw.deck,
      rngCursor,
      exhaustionStage,
      finalTurnsRemaining,
    };
  }

  if (deck.reshuffles === 0 && deck.discard.length > 0) {
    const rand = rngFromSeed(`${seed}:reshuffle:${rngCursor}`);
    const reshuffledDeck = reshuffleDiscardIntoDraw(deck, rand, 1);
    return drawOne({
      deck: reshuffledDeck,
      seed,
      rngCursor: rngCursor + 1,
      exhaustionStage: 1,
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
    exhaustionStage: 2,
    finalTurnsRemaining: finalTurnsRemaining ?? 2,
  };
}

