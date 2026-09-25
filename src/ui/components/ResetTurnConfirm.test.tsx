import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ResetTurnConfirm } from './ResetTurnConfirm';

const noop = () => {};

describe('ResetTurnConfirm', () => {
  it('offers the reset and a way out while open', () => {
    const html = renderToStaticMarkup(
      <ResetTurnConfirm open onCancel={noop} onConfirm={noop} />
    );

    expect(html).toContain('Reset turn?');
    expect(html).toContain('This undoes everything you have done this turn');
    expect(html).toContain('Cancel');
    expect(html).toContain('aria-modal="true"');
  });

  it('renders nothing while closed', () => {
    expect(
      renderToStaticMarkup(
        <ResetTurnConfirm open={false} onCancel={noop} onConfirm={noop} />
      )
    ).toBe('');
  });
});
