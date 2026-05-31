import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { GameLogEntry } from '../../engine/types';
import { LogPanel } from './LogPanel';

const LOG: GameLogEntry[] = [
  {
    turn: 0,
    player: 'PlayerA',
    phase: 'StartTurn',
    summary: 'Seed fixed-seed',
  },
  {
    turn: 1,
    player: 'PlayerA',
    phase: 'ActionWindow',
    summary: 'trade Moons for Knots',
  },
  {
    turn: 1,
    player: 'PlayerB',
    phase: 'CollectIncome',
    summary: 'income choice 6:Waves',
  },
];

describe('LogPanel', () => {
  it('renders reverse-chronological grouped entries, seed rows, and colored suit codes', () => {
    const html = renderToStaticMarkup(
      <LogPanel timelineLog={LOG} humanPlayerId="PlayerA" />
    );

    expect(html.indexOf('T1')).toBeLessThan(html.indexOf('Seed'));
    expect(html).toContain('PlayerA (You)');
    expect(html).toContain('[PlayerB] Income choice 2 ');
    expect(html).toContain('class="log-suit-code"');
    expect(html).toContain('>wa<');
    expect(html).toContain('fixed-seed');
  });

  it('renders an empty state without entries', () => {
    const html = renderToStaticMarkup(
      <LogPanel timelineLog={[]} humanPlayerId="PlayerA" />
    );

    expect(html).toContain('No actions yet.');
  });
});
