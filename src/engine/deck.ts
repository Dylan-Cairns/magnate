import type { CardId } from './cards';
import { discardCard } from './deckCore';
import type { DeckState } from './types';

export { initialSetup } from './setup';
export type { SetupResult } from './setup';
export { drawOne } from './drawPolicy';
export type { DrawContext, DrawResult } from './drawPolicy';

export function discard(deck: DeckState, cardId: CardId): DeckState {
  return discardCard(deck, cardId);
}
