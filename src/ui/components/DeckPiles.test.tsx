import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { DeckPiles } from './DeckPiles';

describe('DeckPiles', () => {
  it('preserves deck and discard animation anchors while applying holdback', () => {
    const html = renderToStaticMarkup(
      <DeckPiles
        drawCount={5}
        reshuffles={1}
        discard={['6', '7']}
        pendingDiscardHoldback={1}
        terminal={false}
      />
    );

    expect(html).toContain('deck-pile-stack is-deck overlay-shift-2');
    expect(html).toContain('deck-pile-stack is-discard');
    expect(html).toContain('deck-pile-stack-card');
    expect(html).toContain('2nd shuffle');
    expect(html).toContain('Discarded Cards: <strong>1</strong>');
    expect(html).toContain('The Desert');
    expect(html).not.toContain('The Author');
  });

  it('keeps empty pile anchors and hides the shuffle label after terminal exhaustion', () => {
    const html = renderToStaticMarkup(
      <DeckPiles
        drawCount={0}
        reshuffles={1}
        discard={[]}
        pendingDiscardHoldback={0}
        terminal
      />
    );

    expect(html).toContain('deck-pile-stack is-deck overlay-shift-0');
    expect(html).toContain('deck-pile-stack is-discard');
    expect(html).toContain('deck-pile-card-empty deck-pile-stack-card');
    expect(html).not.toContain('2nd shuffle');
  });
});
