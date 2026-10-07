import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { GameCelebration } from './GameCelebration';

describe('GameCelebration', () => {
  it('renders nothing when animations are disabled', () => {
    const html = renderToStaticMarkup(
      <GameCelebration outcome="win" animationsEnabled={false} />
    );

    expect(html).toBe('');
  });

  it('renders nothing without an outcome', () => {
    const html = renderToStaticMarkup(
      <GameCelebration outcome={null} animationsEnabled />
    );

    expect(html).toBe('');
  });

  it('renders a decorative win layer with a glow and chips', () => {
    const html = renderToStaticMarkup(
      <GameCelebration outcome="win" animationsEnabled />
    );

    expect(html).toContain('game-celebration is-win');
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain('celebration-glow');
    expect(html).toContain('celebration-chip');
    expect(html).not.toContain('celebration-veil');
    expect(html).not.toContain('celebration-draw-ring');
  });

  it('renders a subdued loss layer', () => {
    const html = renderToStaticMarkup(
      <GameCelebration outcome="loss" animationsEnabled />
    );

    expect(html).toContain('game-celebration is-loss');
    expect(html).toContain('celebration-veil');
    expect(html).toContain('is-settling');
  });

  it('renders a neutral draw layer with rings and no chips', () => {
    const html = renderToStaticMarkup(
      <GameCelebration outcome="draw" animationsEnabled />
    );

    expect(html).toContain('game-celebration is-draw');
    expect(html).toContain('celebration-draw-ring');
    expect(html).not.toContain('celebration-chip');
  });
});
