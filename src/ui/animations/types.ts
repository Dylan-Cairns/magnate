import type { CardId } from '../../engine/cards';
import type { Suit } from '../../engine/types';
import type { CardPerspective } from '../components/CardTile';

export type ResourceFlight = {
  id: string;
  suit: Suit;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  delayMs: number;
  durationMs?: number;
  presentationLandingMs?: number;
  variant?: 'transfer' | 'payment' | 'tax-loss';
};

export type PendingResourceFlight = ResourceFlight;

export type CardFlight = {
  id: string;
  variant: 'play' | 'draw';
  visual: 'face' | 'back';
  cardId?: CardId;
  isDeed: boolean;
  perspective: CardPerspective;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  startWidth: number;
  startHeight: number;
  endWidth: number;
  endHeight: number;
  renderWidth?: number;
  renderHeight?: number;
  /**
   * Destination card scope metrics. When set, the flight lays its card out at
   * the destination's image-area size so the final frame matches the real card
   * exactly, whatever the lane size is. Both dimensions are propagated because
   * custom properties inherit as computed values.
   */
  endImageAreaWidth?: number;
  endImageAreaHeight?: number;
  /**
   * The flight lands on the discard pile, whose cards render with the deck-pile
   * card scope rather than the board card scope. CardFlightLayer marks the
   * flight so its final frame adopts the discard card's padding, meta strip and
   * image-area metrics instead of the board card's.
   */
  discardDestination?: boolean;
  delayMs: number;
  durationMs?: number;
  presentationLandingMs?: number;
};
