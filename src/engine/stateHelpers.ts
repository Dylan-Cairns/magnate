import { CARD_BY_ID, CardId } from './cards';
import type { GameState, PlayerId, PropertyCard, Suit } from './types';

export const SUITS: readonly Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

export function findProperty(cardId: CardId): PropertyCard | undefined {
  const card = CARD_BY_ID[cardId];
  return card?.kind === 'Property' ? card : undefined;
}

export function canAfford(
  pool: GameState['players'][number]['resources'],
  cost: Partial<Record<Suit, number>>
): boolean {
  return SUITS.every((suit) => (cost[suit] ?? 0) <= pool[suit]);
}

export function applyDelta(
  pool: GameState['players'][number]['resources'],
  delta: Partial<Record<Suit, number>>
): GameState['players'][number]['resources'] {
  const next = { ...pool };
  SUITS.forEach((suit) => {
    next[suit] = next[suit] + (delta[suit] ?? 0);
  });
  return next;
}

export function mergeTokens(
  existing: Partial<Record<Suit, number>> | undefined,
  added: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const next: Partial<Record<Suit, number>> = { ...(existing ?? {}) };
  SUITS.forEach((suit) => {
    if (added[suit]) {
      next[suit] = (next[suit] ?? 0) + (added[suit] ?? 0);
    }
  });
  return next;
}

export function sumTokens(tokens: Partial<Record<Suit, number>>): number {
  return SUITS.reduce((total, suit) => total + (tokens[suit] ?? 0), 0);
}

export function deedCost(card: PropertyCard): Partial<Record<Suit, number>> {
  if (card.rank === 1) {
    return { [card.suits[0]]: 1 };
  }
  return card.suits.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
    acc[suit] = (acc[suit] ?? 0) + 1;
    return acc;
  }, {});
}

export function placementAllowed(
  card: PropertyCard,
  district: GameState['districts'][number],
  playerId: PlayerId
): boolean {
  const stack = district.stacks[playerId];
  if (stack?.deed) {
    return false;
  }
  const suits = new Set(card.suits);
  if (stack && stack.developed.length > 0) {
    const previous = findProperty(stack.developed[stack.developed.length - 1]);
    return Boolean(previous?.suits.some((suit) => suits.has(suit)));
  }
  return district.markerSuitMask.some((suit) => suits.has(suit));
}

export function defaultOutrightPayment(
  card: PropertyCard,
  pool: GameState['players'][number]['resources']
): Partial<Record<Suit, number>> | undefined {
  const required = new Set(card.suits);
  if (![...required].every((suit) => pool[suit] > 0)) {
    return undefined;
  }
  const payment = [...required].reduce<Partial<Record<Suit, number>>>(
    (acc, suit) => {
      acc[suit] = 1;
      return acc;
    },
    {}
  );
  const remainder = card.rank - required.size;
  if (remainder <= 0) {
    return payment;
  }
  const priorities = [...required].sort((a, b) => pool[b] - pool[a]);
  let outstanding = remainder;
  priorities.forEach((suit) => {
    if (outstanding === 0) {
      return;
    }
    const capacity = pool[suit] - (payment[suit] ?? 0);
    const spend = Math.min(capacity, outstanding);
    payment[suit] = (payment[suit] ?? 0) + spend;
    outstanding -= spend;
  });
  return outstanding === 0 ? payment : undefined;
}
