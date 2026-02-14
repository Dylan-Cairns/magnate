import type { CardId } from './cards';
import { shuffleInPlace, type RandomFn } from './rng';
import type { DeckState } from './types';

export interface DrawTopResult {
  cardId?: CardId;
  deck: DeckState;
}

export function createDeck(draw: readonly CardId[]): DeckState {
  return {
    draw: [...draw],
    discard: [],
    reshuffles: 0,
  };
}

export function drawTop(deck: DeckState): DrawTopResult {
  if (deck.draw.length === 0) {
    return { cardId: undefined, deck };
  }
  const [cardId, ...rest] = deck.draw;
  return {
    cardId,
    deck: {
      ...deck,
      draw: rest,
    },
  };
}

export function reshuffleDiscardIntoDraw(
  deck: DeckState,
  rand: RandomFn,
  reshuffles: 1 | 2
): DeckState {
  const draw = [...deck.discard];
  shuffleInPlace(draw, rand);
  return {
    draw,
    discard: [],
    reshuffles,
  };
}

export function discardCard(deck: DeckState, cardId: CardId): DeckState {
  return {
    ...deck,
    discard: [cardId, ...deck.discard],
  };
}

