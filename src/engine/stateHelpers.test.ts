import { describe, expect, it } from 'vitest';

import { CARD_BY_ID } from './cards';
import { developmentCost, placementAllowed } from './stateHelpers';
import { PLAYER_A, PLAYER_B, makeDistrict } from './__tests__/fixtures';

describe('placementAllowed', () => {
  it('requires suit overlap with non-Excuse district marker on first placement', () => {
    const card = CARD_BY_ID['6'];
    if (card.kind !== 'Property') {
      throw new Error('Expected property test card.');
    }

    const district = makeDistrict('D1', ['Waves']);
    expect(placementAllowed(card, district, PLAYER_A)).toBe(false);
  });

  it('allows first placement in Excuse district regardless of suit', () => {
    const card = CARD_BY_ID['7'];
    if (card.kind !== 'Property') {
      throw new Error('Expected property test card.');
    }

    const district = makeDistrict('D5', []);
    expect(placementAllowed(card, district, PLAYER_A)).toBe(true);
  });

  it('requires suit overlap with the previous property for subsequent Excuse placements', () => {
    const district = makeDistrict('D5', [], {
      [PLAYER_A]: { developed: ['6'] },
      [PLAYER_B]: { developed: [] },
    });

    const nonMatching = CARD_BY_ID['7'];
    const matching = CARD_BY_ID['13'];
    if (nonMatching.kind !== 'Property' || matching.kind !== 'Property') {
      throw new Error('Expected property test cards.');
    }

    expect(placementAllowed(nonMatching, district, PLAYER_A)).toBe(false);
    expect(placementAllowed(matching, district, PLAYER_A)).toBe(true);
  });

  it('requires overlap with previous developed property in the same district', () => {
    const district = makeDistrict('D1', ['Moons'], {
      [PLAYER_A]: { developed: ['6'] },
      [PLAYER_B]: { developed: [] },
    });

    const nonMatching = CARD_BY_ID['8'];
    const matching = CARD_BY_ID['13'];
    if (nonMatching.kind !== 'Property' || matching.kind !== 'Property') {
      throw new Error('Expected property test cards.');
    }

    expect(placementAllowed(nonMatching, district, PLAYER_A)).toBe(false);
    expect(placementAllowed(matching, district, PLAYER_A)).toBe(true);
  });

  it('disallows placement when player has an active deed in that district', () => {
    const district = makeDistrict('D2', ['Suns'], {
      [PLAYER_A]: {
        developed: [],
        deed: {
          cardId: '7',
          progress: 1,
          tokens: { Suns: 1 },
        },
      },
      [PLAYER_B]: { developed: [] },
    });

    const card = CARD_BY_ID['7'];
    if (card.kind !== 'Property') {
      throw new Error('Expected property test card.');
    }

    expect(placementAllowed(card, district, PLAYER_A)).toBe(false);
  });
});

describe('issue regressions', () => {
  it('issue 4: Excuse first-card placement stays legal', () => {
    const card = CARD_BY_ID['24'];
    if (card.kind !== 'Property') {
      throw new Error('Expected property test card.');
    }
    const excuseDistrict = makeDistrict('D5', []);
    expect(placementAllowed(card, excuseDistrict, PLAYER_A)).toBe(true);
  });

  it('ace completion interpretation: ace deed development target is 3 tokens', () => {
    const ace = CARD_BY_ID['0'];
    const rankTwo = CARD_BY_ID['6'];
    if (ace.kind !== 'Property' || rankTwo.kind !== 'Property') {
      throw new Error('Expected property test cards.');
    }

    expect(developmentCost(ace)).toBe(3);
    expect(developmentCost(rankTwo)).toBe(2);
  });
});
