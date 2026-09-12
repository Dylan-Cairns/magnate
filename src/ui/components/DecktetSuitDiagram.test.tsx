import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { Suit } from '../../engine/types';
import { DecktetSuitDiagram } from './DecktetSuitDiagram';

const DIM_IDS = new Set<string>();
const DIM_SUITS = new Set<Suit>();

function renderDiagram(ruleset: 'standard' | 'extended') {
  return renderToStaticMarkup(
    <DecktetSuitDiagram
      ruleset={ruleset}
      dimmedCardIds={DIM_IDS}
      dimmedSuits={DIM_SUITS}
    />
  );
}

describe('DecktetSuitDiagram', () => {
  it('omits the Courts row for the standard ruleset', () => {
    const html = renderDiagram('standard');

    expect(html).toContain('Deck Map');
    expect(html).not.toContain('suit-diagram-courts');
    expect(html).not.toContain('card-rank-court-icon');
  });

  it('renders each extended Court as its rank icon followed by its three suits', () => {
    const html = renderDiagram('extended');
    const chips = html
      .split('class="suit-diagram-court tooltip-trigger"')
      .slice(1);

    expect(chips).toHaveLength(4);

    const expectedAlts = [
      ['Court', 'Moons', 'Waves', 'Knots'],
      ['Court', 'Suns', 'Waves', 'Wyrms'],
      ['Court', 'Moons', 'Leaves', 'Wyrms'],
      ['Court', 'Suns', 'Leaves', 'Knots'],
    ];
    chips.forEach((chip, index) => {
      const labels = [
        ...chip.matchAll(/(?:alt|aria-label)="([^"]+)"/g),
      ].map((match) => match[1]);
      expect(labels).toEqual(expectedAlts[index]);
    });
  });

  it('dims a Court row once that Court has left circulation', () => {
    const html = renderToStaticMarkup(
      <DecktetSuitDiagram
        ruleset="extended"
        dimmedCardIds={new Set(['43'])}
        dimmedSuits={DIM_SUITS}
      />
    );
    const chips = html
      .split('class="suit-diagram-court tooltip-trigger')
      .slice(1);

    expect(chips).toHaveLength(4);
    expect(chips[0].startsWith(' is-dimmed')).toBe(false);
    expect(chips[2].startsWith(' is-dimmed')).toBe(true);
  });
});
