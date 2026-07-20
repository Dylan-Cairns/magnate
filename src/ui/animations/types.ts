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
  delayMs: number;
  durationMs?: number;
};
