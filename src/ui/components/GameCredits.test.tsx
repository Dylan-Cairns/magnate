import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { GameCredits } from './GameCredits';

describe('GameCredits', () => {
  it('renders the Magnate, Decktet, and license credits', () => {
    const html = renderToStaticMarkup(<GameCredits />);

    expect(html).toContain('Cristyn Magnus');
    expect(html).toContain('P.D. Magnus');
    expect(html).toContain('The Decktet');
    expect(html).toContain('creativecommons.org/licenses/by-nc-sa/4.0/');
    expect(html).toContain('third-party-notices.txt');
  });
});
