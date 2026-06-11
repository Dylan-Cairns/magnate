import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { FinalScore, ObservedPlayerState } from '../../engine/types';
import { BotHandPanel } from './BotHandPanel';

const SCORE: FinalScore = {
  districtPoints: { PlayerA: 2, PlayerB: 1 },
  rankTotals: { PlayerA: 18, PlayerB: 14 },
  resourceTotals: { PlayerA: 6, PlayerB: 4 },
  winner: 'PlayerA',
  decidedBy: 'districts',
};

describe('BotHandPanel', () => {
  it('renders hidden hand anchors for the actual bot hand count', () => {
    const player: ObservedPlayerState = {
      id: 'PlayerB',
      crowns: ['33', '34', '35'],
      resources: {
        Moons: 0,
        Suns: 2,
        Waves: 1,
        Leaves: 0,
        Wyrms: 0,
        Knots: 1,
      },
      hand: [],
      handCount: 3,
      handHidden: true,
    };

    const html = renderToStaticMarkup(
      <BotHandPanel
        player={player}
        isActive={false}
        score={SCORE}
        terminal={false}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );

    expect(html).toContain('data-player-id="PlayerB"');
    expect(html).toContain('data-hand-owner-id="PlayerB"');
    expect(html.match(/data-hand-slot-kind="hidden"/g)).toHaveLength(3);
    expect(html).toContain('--bot-hand-card-angle:-6deg');
    expect(html).toContain('--bot-hand-card-angle:0deg');
    expect(html).toContain('--bot-hand-card-angle:6deg');
    expect(html).not.toContain('data-hand-slot-kind="empty"');
  });
});
