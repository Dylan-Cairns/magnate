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

function developmentTarget(card: PropertyCard): number {
  if (card.rank === 1 && card.suits.length === 1) {
    return 3;
  }
  return card.rank;
}

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

export function developmentCost(card: PropertyCard): number {
  return developmentTarget(card);
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
  const isExcuseDistrict = district.markerSuitMask.length === 0;
  if (isExcuseDistrict) {
    return true;
  }
  return district.markerSuitMask.some((suit) => suits.has(suit));
}

export function enumerateOutrightPayments(
  card: PropertyCard,
  pool: GameState['players'][number]['resources']
): Partial<Record<Suit, number>>[] {
  const suits = [...card.suits];
  if (suits.length === 0) {
    return [];
  }
  if (suits.some((suit) => pool[suit] <= 0)) {
    return [];
  }

  const cost = developmentTarget(card);
  if (cost < suits.length) {
    return [];
  }

  const basePayment = suits.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
    acc[suit] = 1;
    return acc;
  }, {});

  if (suits.some((suit) => pool[suit] < (basePayment[suit] ?? 0))) {
    return [];
  }

  const remainder = cost - suits.length;
  if (remainder === 0) {
    return [{ ...basePayment }];
  }

  const payments: Partial<Record<Suit, number>>[] = [];
  function backtrack(index: number, remaining: number) {
    if (index === suits.length) {
      if (remaining === 0) {
        payments.push({ ...basePayment });
      }
      return;
    }

    const suit = suits[index];
    const already = basePayment[suit] ?? 0;
    const maxExtra = Math.min(remaining, (pool[suit] ?? 0) - already);
    for (let extra = 0; extra <= maxExtra; extra++) {
      basePayment[suit] = already + extra;
      backtrack(index + 1, remaining - extra);
    }
    basePayment[suit] = already;
  }

  backtrack(0, remainder);
  return payments;
}
