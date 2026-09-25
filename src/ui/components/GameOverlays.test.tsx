import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { StartupPreloadOverlay } from './GameOverlays';

const noop = () => {};

describe('StartupPreloadOverlay', () => {
  it('clamps progress display and renders retry state after an error', () => {
    const html = renderToStaticMarkup(
      <StartupPreloadOverlay
        ready={false}
        error="network failed"
        progress={{
          completed: 8,
          total: 5,
          percent: 120,
          message: 'Loading',
        }}
        onRetry={noop}
      />
    );

    expect(html).toContain('Loading Failed');
    expect(html).toContain('Could not preload assets: network failed');
    expect(html).toContain('aria-valuenow="100"');
    expect(html).toContain('width:100%');
    expect(html).toContain('5 / 5');
    expect(html).toContain('Retry');
  });
});
