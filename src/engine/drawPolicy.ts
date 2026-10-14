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
  const ready = resolveExhaustion(context);
  const draw = drawTop(ready.deck);
  const resolved = resolveExhaustion({ ...ready, deck: draw.deck });

  return {
    cardId: draw.cardId,
    deck: resolved.deck,
    rngCursor: resolved.rngCursor,
    finalTurnsRemaining: resolved.finalTurnsRemaining,
  };
}

function resolveExhaustion(context: DrawContext): DrawContext {
  const { deck, seed, rngCursor, finalTurnsRemaining } = context;
  if (deck.draw.length > 0) {
    return context;
  }

  if (deck.reshuffles === 0 && deck.discard.length > 0) {
    const rand = rngFromSeed(`${seed}:reshuffle:${rngCursor}`);
    return {
      ...context,
      deck: reshuffleDiscardIntoDraw(deck, rand),
      rngCursor: rngCursor + 1,
    };
  }

  // An empty discard pile leaves the second deck empty too.
  return {
    ...context,
    deck: {
      ...deck,
      draw: [],
      reshuffles: 2,
    },
    finalTurnsRemaining: finalTurnsRemaining ?? 2,
  };
}
