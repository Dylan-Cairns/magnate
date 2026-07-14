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

  it('renders pulsing and dimmed d10 state from explicit dice phase', () => {
    const html = renderToStaticMarkup(
      <RollResult
        dice={{
          ...BASE_DICE,
          incomePhase: 'pulsing',
        }}
        gameKey="seed"
      />
    );

    expect(html).toContain('aria-label="d10: 4"');
    expect(html).toContain('aria-label="d10: 7"');
    expect(html).toContain('die-scene-d10 is-dimmed');
    expect(html).toContain('die-scene-d10 is-pulsing');
    expect(html).toContain('die-scene-d6 is-dimmed');
  });

  it('renders tax die visibility and pulse from explicit tax phase', () => {
    const html = renderToStaticMarkup(
      <RollResult
        dice={{
          ...BASE_DICE,
          taxSuit: 'Moons',
          taxPhase: 'pulsing',
        }}
        gameKey="seed"
      />
    );

    expect(html).toContain('aria-label="d6: Moons"');
    expect(html).toContain('die-scene-d6 is-pulsing');
  });
});
