import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { SuitText } from './SuitText';

describe('SuitText', () => {
  it('renders embedded suit tokens as inline chips and preserves surrounding text', () => {
    const html = renderToStaticMarkup(
      <SuitText text="Trade {Moons}x3 for {Knots}" />
    );

    expect(html).toContain('Trade ');
    expect(html).toContain('data-token-suit="Moons"');
    expect(html).toContain('x3 for ');
    expect(html).toContain('data-token-suit="Knots"');
    expect(html).toContain('inline-token-chip');
  });
});
