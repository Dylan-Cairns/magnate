import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { ResourceFlightLayer } from './ResourceFlightLayer';

describe('ResourceFlightLayer', () => {
  it('renders flight variants and motion CSS variables', () => {
    const html = renderToStaticMarkup(
      <ResourceFlightLayer
        flights={[
          {
            id: 'flight-tax',
            suit: 'Suns',
            startX: 10,
            startY: 20,
            endX: 35,
            endY: 50,
            delayMs: 75,
            durationMs: 900,
            variant: 'tax-loss',
          },
          {
            id: 'flight-clear',
            suit: 'Knots',
            startX: 0,
            startY: 0,
            endX: 5,
            endY: -20,
            delayMs: 0,
            variant: 'terminal-clear',
          },
        ]}
      />
    );

    expect(html).toContain('resource-flight is-tax-loss');
    expect(html).toContain('resource-flight is-terminal-clear');
    expect(html).toContain('--resource-flight-dx:25px');
    expect(html).toContain('--resource-flight-dy:30px');
    expect(html).toContain('--resource-flight-delay-ms:75ms');
    expect(html).toContain('--resource-flight-duration-ms:900ms');
  });

  it('renders nothing without flights', () => {
    expect(renderToStaticMarkup(<ResourceFlightLayer flights={[]} />)).toBe('');
  });
});
