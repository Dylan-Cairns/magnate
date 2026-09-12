import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CardTile } from './CardTile';

describe('CardTile', () => {
  it('renders a deed ghost with zero progress and no interactive tooltip', () => {
    const html = renderToStaticMarkup(
      <CardTile
        cardId="6"
        preview
        inDevelopment
        deedProgress={0}
        deedTarget={2}
        animateDeedProgress={false}
      />
    );
    expect(html).toContain('is-in-development');
    expect(html).toContain('>0/2<');
    expect(html).toContain('deed-progress-ring-track');
    expect(html).not.toContain('deed-progress-ring-value');
    expect(html).not.toContain('data-card-id');
    expect(html).not.toContain('tooltip-anchor');
  });

  it('renders preview artwork without a real-card animation target or tooltip', () => {
    const html = renderToStaticMarkup(<CardTile cardId="6" preview />);
    expect(html).toContain('card-image');
    expect(html).toContain('card-rank');
    expect(html).not.toContain('data-card-id');
    expect(html).not.toContain('tooltip-anchor');
  });

  it('opens player-area card tooltips below their stack but keeps hand cards above', () => {
    const humanHtml = renderToStaticMarkup(<CardTile cardId="29" />);
    const handHtml = renderToStaticMarkup(
      <CardTile
        cardId="29"
        handOwnerId="PlayerA"
        handCardId="29"
        handSlotKind="occupied"
      />
    );
    const botHtml = renderToStaticMarkup(
      <CardTile cardId="29" perspective="bot" />
    );

    expect(humanHtml).toContain('tooltip-anchor tooltip-below');
    expect(handHtml).not.toContain('tooltip-anchor tooltip-below');
    expect(botHtml).not.toContain('tooltip-anchor tooltip-below');
  });

  it('renders no deed progress value arc at zero progress', () => {
    const html = renderToStaticMarkup(
      <CardTile
        cardId="29"
        deedProgress={0}
        deedTarget={9}
        inDevelopment
        animateDeedProgress
      />
    );

    expect(html).toContain('class="deed-progress-ring-track"');
    expect(html).not.toContain('class="deed-progress-ring-value"');
    expect(html).toContain('>0/9<');
  });

  it('renders deterministic deed progress arcs for high-cost cards', () => {
    const html = renderToStaticMarkup(
      <CardTile
        cardId="29"
        deedProgress={1}
        deedTarget={9}
        inDevelopment
        animateDeedProgress
      />
    );

    expect(html).toContain('class="deed-progress-ring-value"');
    expect(html).toContain(
      'd="M 18 2 A 16 16 0 0 1 28.284601754984628 5.743288910096352"'
    );
    expect(html).toContain('>1/9<');
  });
});
