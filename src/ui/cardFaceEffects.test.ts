import { describe, expect, it } from 'vitest';

import { ALL_CARDS } from '../engine/cards';
import { cardFaceEffect } from './cardFaceEffects';

describe('cardFaceEffects', () => {
  it('maps The Darkness to the darkness face effect', () => {
    const darkness = ALL_CARDS.find((card) => card.name === 'The Darkness');
    if (!darkness) {
      throw new Error('Expected The Darkness in the card catalog.');
    }
    expect(cardFaceEffect(darkness)).toBe('darkness');
  });

  it('leaves every other card without a face effect', () => {
    const others = ALL_CARDS.filter((card) => card.name !== 'The Darkness');
    expect(others.length).toBeGreaterThan(0);
    for (const card of others) {
      expect(cardFaceEffect(card)).toBeNull();
    }
  });
});
