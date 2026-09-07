import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { DeckPiles } from './DeckPiles';

describe('DeckPiles', () => {
  it('preserves deck and discard animation anchors', () => {
    const html = renderToStaticMarkup(
      <DeckPiles
        drawCount={5}
        reshuffles={1}
        discard={['6', '7']}
      />
    );

    expect(html).toContain('deck-pile-stack is-deck overlay-shift-2');
    expect(html).toContain('deck-pile-stack is-discard');
    expect(html).toContain('deck-pile-stack-card');
    expect(html).toContain('Shuffles 2/2');
    expect(html).toContain('status-badge');
    expect(html).toContain(
      'title="The deck is shuffled at the start of the game. When it runs out, the discard pile is shuffled to form a new draw pile. When that runs out, each player gets one final turn."'
    );
    expect(html).toContain('Discarded Cards: <strong>2</strong>');
    expect(html).toContain('The Desert');
    expect(html).toContain('The Author');
  });

  it('keeps empty pile anchors and reports the completed second shuffle', () => {
    const html = renderToStaticMarkup(
      <DeckPiles drawCount={0} reshuffles={1} discard={[]} />
    );

    expect(html).toContain('deck-pile-stack is-deck overlay-shift-0');
    expect(html).toContain('deck-pile-stack is-discard');
    expect(html).toContain('deck-pile-card-empty deck-pile-stack-card');
    expect(html).toContain('Shuffles 2/2');
  });

  it('counts the initial game shuffle as the first of two', () => {
    const html = renderToStaticMarkup(
      <DeckPiles drawCount={5} reshuffles={0} discard={[]} />
    );

    expect(html).toContain('Shuffles 1/2');
  });
});
