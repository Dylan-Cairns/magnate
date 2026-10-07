import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { FinalScore, ObservedPlayerState } from '../../engine/types';
import { PlayerPanel } from './PlayerPanel';

const SCORE: FinalScore = {
  districtPoints: { PlayerA: 2, PlayerB: 1 },
  rankTotals: { PlayerA: 18, PlayerB: 14 },
  resourceTotals: { PlayerA: 6, PlayerB: 4 },
  winner: 'PlayerA',
  decidedBy: 'districts',
};

describe('PlayerPanel hand anchors', () => {
  it('renders player and hand-card data attributes for visible hand cards', () => {
    const player: ObservedPlayerState = {
      id: 'PlayerA',
      crowns: ['30', '31', '32'],
      resources: {
        Moons: 2,
        Suns: 1,
        Waves: 0,
        Leaves: 0,
        Wyrms: 1,
        Knots: 0,
      },
      hand: ['6', '7'],
      handCount: 2,
      handHidden: false,
    };

    const html = renderToStaticMarkup(
      <PlayerPanel
        player={player}
        isActive
        score={SCORE}
        terminal={false}
        handSlotCount={3}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );

    expect(html).toContain('data-player-id="PlayerA"');
    expect(html).toContain('data-hand-owner-id="PlayerA"');
    expect(html).toContain('data-hand-card-id="6"');
    expect(html).toContain('data-hand-card-id="7"');
    expect(html).toContain('data-hand-slot-kind="occupied"');
    expect(html).toContain('data-hand-slot-kind="empty"');
  });

  it('marks the leading player with a crown', () => {
    const player: ObservedPlayerState = {
      id: 'PlayerA',
      crowns: [],
      resources: {
        Moons: 0,
        Suns: 0,
        Waves: 0,
        Leaves: 0,
        Wyrms: 0,
        Knots: 0,
      },
      hand: [],
      handCount: 0,
      handHidden: false,
    };

    const leaderHtml = renderToStaticMarkup(
      <PlayerPanel
        player={player}
        isActive={false}
        score={SCORE}
        terminal={false}
        handSlotCount={3}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );
    const trailingHtml = renderToStaticMarkup(
      <PlayerPanel
        player={{ ...player, id: 'PlayerB' }}
        isActive={false}
        score={SCORE}
        terminal={false}
        handSlotCount={3}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );
    const drawHtml = renderToStaticMarkup(
      <PlayerPanel
        player={player}
        isActive={false}
        score={{ ...SCORE, winner: 'Draw', decidedBy: 'draw' }}
        terminal={false}
        handSlotCount={3}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );

    // The crown sits just left of the VP badge for the leader only, outside
    // the engraving.
    expect(leaderHtml).toContain('player-score-crown');
    expect(trailingHtml).not.toContain('player-score-crown');
    expect(drawHtml).not.toContain('player-score-crown');
    expect(leaderHtml.indexOf('player-score-crown')).toBeLessThan(
      leaderHtml.indexOf('class="engraving"')
    );
  });

  it('renders hidden hand slot markers for hidden hands', () => {
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
      handCount: 2,
      handHidden: true,
    };

    const html = renderToStaticMarkup(
      <PlayerPanel
        player={player}
        isActive={false}
        score={SCORE}
        terminal={false}
        handSlotCount={3}
        humanPlayerId="PlayerA"
        botPlayerId="PlayerB"
      />
    );

    expect(html).toContain('data-player-id="PlayerB"');
    expect(html).toContain('data-hand-owner-id="PlayerB"');
    expect(html).toContain('data-hand-slot-kind="hidden"');
    expect(html).toContain('data-hand-slot-kind="empty"');
  });
});
