import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { CardFlight } from '../animations/types';
import { CardFlightLayer } from './CardFlightLayer';

const BASE_FLIGHT: CardFlight = {
  id: 'card-flight',
  variant: 'play',
  visual: 'face',
  cardId: '6',
  isDeed: true,
  perspective: 'human',
  startX: 10,
  startY: 20,
  endX: 50,
  endY: 80,
  startWidth: 50,
  startHeight: 100,
  endWidth: 100,
  endHeight: 200,
  renderWidth: 100,
  renderHeight: 200,
  delayMs: 25,
};

describe('CardFlightLayer', () => {
  it('renders face and hidden flights with variants and motion CSS variables', () => {
    const html = renderToStaticMarkup(
      <CardFlightLayer
        animationsEnabled
        flights={[
          { ...BASE_FLIGHT, variant: 'draw' },
          {
            ...BASE_FLIGHT,
            id: 'played-back',
            variant: 'play',
            visual: 'back',
            cardId: undefined,
          },
        ]}
      />
    );

    expect(html).toContain('card-flight is-draw');
    expect(html).toContain('--card-flight-dx:40px');
    expect(html).toContain('--card-flight-dy:60px');
    expect(html).toContain('--card-flight-start-scale-x:0.5');
    expect(html).toContain('--card-flight-start-scale-y:0.5');
    expect(html).toContain('--card-flight-end-scale-x:1');
    expect(html).toContain('--card-flight-end-scale-y:1');
    expect(html).toContain('width:100px');
    expect(html).toContain('height:200px');
    expect(html).toContain('data-card-id="6"');
    expect(html).toContain('card-tile card-back');
  });

  it('renders nothing without flights', () => {
    expect(
      renderToStaticMarkup(
        <CardFlightLayer flights={[]} animationsEnabled={false} />
      )
    ).toBe('');
  });
});
