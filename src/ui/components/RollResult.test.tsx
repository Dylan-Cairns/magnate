import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { DiceVisualState } from '../runtime/types';
import { RollResult } from './RollResult';

const BASE_DICE: DiceVisualState = {
  incomeRoll: { die1: 4, die2: 7, rollId: 12 },
  taxSuit: undefined,
  incomePhase: 'settled',
  taxPhase: 'dimmed',
};

describe('RollResult', () => {
  it('renders an empty result without dice state', () => {
    expect(renderToStaticMarkup(<RollResult dice={null} />)).toBe(
      '<p class="roll-value">-</p>'
    );
  });

  it('renders the winning d10 with a steady glow and dims the loser', () => {
    const html = renderToStaticMarkup(
      <RollResult dice={BASE_DICE} gameKey="seed" />
    );

    expect(html).toContain('aria-label="d10: 4"');
    expect(html).toContain('aria-label="d10: 7"');
    expect(html).toContain('die-scene-d10 is-dimmed');
    expect(html).toContain('die-scene-d10 is-glowing');
    expect(html).not.toContain('is-pulsing');
    expect(html).toContain('die-scene-d6 is-dimmed');
  });

  it('renders a settled tax die with a steady glow', () => {
    const html = renderToStaticMarkup(
      <RollResult
        dice={{
          ...BASE_DICE,
          taxSuit: 'Moons',
          taxPhase: 'settled',
        }}
        gameKey="seed"
      />
    );

    expect(html).toContain('aria-label="d6: Moons"');
    expect(html).toContain('die-scene-d6 is-glowing');
    expect(html).not.toContain('is-pulsing');
  });

  it('keeps the tax die dimmed while only the income dice are rolling', () => {
    const html = renderToStaticMarkup(
      <RollResult
        dice={{
          ...BASE_DICE,
          incomePhase: 'rolling',
          taxPhase: 'hidden',
        }}
        gameKey="seed"
      />
    );

    expect(html).toContain('die-scene-d6 is-dimmed');
    expect(html).not.toContain('die-scene-d6 is-glowing');
  });

  it('undims the tax die when its own roll begins', () => {
    const html = renderToStaticMarkup(
      <RollResult
        dice={{
          ...BASE_DICE,
          taxSuit: 'Moons',
          taxPhase: 'rolling',
        }}
        gameKey="seed"
      />
    );

    expect(html).toContain('aria-label="d6: Moons"');
    expect(html).toContain('class="die-scene-d6"');
    expect(html).not.toContain('die-scene-d6 is-dimmed');
    expect(html).not.toContain('die-scene-d6 is-glowing');
  });
});
