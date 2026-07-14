import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { CardTile } from './CardTile';

describe('CardTile', () => {
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
