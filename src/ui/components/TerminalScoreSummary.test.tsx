import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { FinalScore } from '../../engine/types';
import { TerminalScoreSummary } from './TerminalScoreSummary';

const SCORE: FinalScore = {
  districtPoints: { PlayerA: 2, PlayerB: 1 },
  rankTotals: { PlayerA: 18, PlayerB: 14 },
  resourceTotals: { PlayerA: 6, PlayerB: 4 },
  winner: 'PlayerA',
  decidedBy: 'districts',
};

describe('TerminalScoreSummary', () => {
  it('renders winner, decider, player ordering, and district lists', () => {
    const html = renderToStaticMarkup(
      <TerminalScoreSummary
        score={SCORE}
        wonDistrictsByPlayer={{ PlayerA: ['D1', 'D3'], PlayerB: [] }}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );

    expect(html).toContain('Winner: <strong>You</strong>');
    expect(html).toContain('<strong>Districts</strong>');
    expect(html.indexOf('<h3>You</h3>')).toBeLessThan(
      html.indexOf('<h3>Bot</h3>')
    );
    expect(html).toContain('<strong>D1, D3</strong>');
    expect(html).toContain('<strong>None</strong>');
  });
});
