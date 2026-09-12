import { describe, expect, it } from 'vitest';

import { deriveHandCardSlides, readHandCardPositions } from './handSlide';

describe('handSlide', () => {
  it('reads positions for occupied hand cards only', () => {
    const positions = readHandCardPositions([
      makeHandElement('6', 120),
      makeHandElement('7', 260),
      makeHandElement(null, 400),
    ]);

    expect([...positions]).toEqual([
      ['6', { left: 120 }],
      ['7', { left: 260 }],
    ]);
  });

  it('derives leftward slides for cards present in both layouts', () => {
    const previous = new Map([
      ['6', { left: 120 }],
      ['7', { left: 260 }],
      ['8', { left: 400 }],
    ]);
    const next = new Map([
      ['7', { left: 120 }],
      ['8', { left: 260 }],
      ['9', { left: 400 }],
    ]);

    expect(deriveHandCardSlides(previous, next)).toEqual([
      { cardId: '7', dx: 140 },
      { cardId: '8', dx: 140 },
    ]);
  });

  it('ignores cards that did not move', () => {
    const positions = new Map([['6', { left: 120 }]]);

    expect(deriveHandCardSlides(positions, positions)).toEqual([]);
  });
});

function makeHandElement(cardId: string | null, left: number): HTMLElement {
  return {
    getAttribute: (name: string) =>
      name === 'data-hand-card-id' ? cardId : null,
    getBoundingClientRect: () => ({ left }) as DOMRect,
  } as unknown as HTMLElement;
}
