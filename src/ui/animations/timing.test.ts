import { describe, expect, it } from 'vitest';

import {
  cardFlightSettleMs,
  resourceFlightSettleMs,
} from './timing';
import type { CardFlight, ResourceFlight } from './types';

describe('animation settle timing', () => {
  it('uses the latest resource flight end plus the commit buffer', () => {
    const flights: ResourceFlight[] = [
      makeResourceFlight({ delayMs: 75 }),
      makeResourceFlight({ delayMs: 10, durationMs: 600 }),
    ];

    expect(resourceFlightSettleMs([])).toBe(0);
    expect(resourceFlightSettleMs(flights)).toBe(630);
  });

  it('uses the latest card flight end plus the commit buffer', () => {
    const flights: CardFlight[] = [
      makeCardFlight({ delayMs: 280 }),
      makeCardFlight({ delayMs: 10, durationMs: 700 }),
    ];

    expect(cardFlightSettleMs([])).toBe(0);
    expect(cardFlightSettleMs(flights)).toBe(730);
  });
});

function makeResourceFlight(
  overrides: Partial<ResourceFlight> = {}
): ResourceFlight {
  return {
    id: 'resource-flight',
    suit: 'Moons',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    delayMs: 0,
    ...overrides,
  };
}

function makeCardFlight(overrides: Partial<CardFlight> = {}): CardFlight {
  return {
    id: 'card-flight',
    variant: 'play',
    visual: 'face',
    cardId: '6',
    isDeed: false,
    perspective: 'human',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    startWidth: 100,
    startHeight: 150,
    endWidth: 100,
    endHeight: 150,
    delayMs: 0,
    ...overrides,
  };
}
