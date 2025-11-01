import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { DistrictState, ObservedPlayerState } from '../../engine/types';
import { DistrictColumn, PlayerTokenRail } from './DistrictBoard';

describe('DistrictColumn', () => {
  it('uses canonical ace scoring for district lane totals', () => {
    const district: DistrictState = {
      id: 'D1',
      markerSuitMask: ['Moons'],
      stacks: {
        PlayerA: {
          developed: ['0', '6'],
        },
        PlayerB: {
          developed: [],
        },
      },
    };

    const html = renderToStaticMarkup(
      <DistrictColumn district={district} humanPlayerId="PlayerB" botPlayerId="PlayerA" />
    );

    const laneScores = [...html.matchAll(/aria-label="District score: (\d+)"/g)].map((match) =>
      Number.parseInt(match[1], 10)
    );
    expect(laneScores).toEqual([4, 0]);
  });

  it('marks only the leading district score as bold', () => {
    const district: DistrictState = {
      id: 'D2',
      markerSuitMask: ['Suns'],
      stacks: {
        PlayerA: {
          developed: ['10'],
        },
        PlayerB: {
          developed: ['6'],
        },
      },
    };

    const html = renderToStaticMarkup(
      <DistrictColumn district={district} humanPlayerId="PlayerB" botPlayerId="PlayerA" />
    );

    const scoreClassMatches = [
      ...html.matchAll(/class="[^"]*\bdistrict-lane-score\b[^"]*"/g),
    ];
    expect(scoreClassMatches).toHaveLength(2);

    const leadingScoreMatches = [
      ...html.matchAll(/class="[^"]*\bdistrict-lane-score\b[^"]*\bis-leading\b[^"]*"/g),
    ];
    expect(leadingScoreMatches).toHaveLength(1);
  });

  it('renders deed progress rings and values for both bot and human lane perspectives', () => {
    const district: DistrictState = {
      id: 'D3',
      markerSuitMask: ['Waves'],
      stacks: {
        PlayerA: {
          developed: [],
          deed: { cardId: '6', progress: 3, tokens: { Waves: 3 } },
        },
        PlayerB: {
          developed: [],
          deed: { cardId: '6', progress: 1, tokens: { Waves: 1 } },
        },
      },
    };

    const html = renderToStaticMarkup(
      <DistrictColumn district={district} humanPlayerId="PlayerB" botPlayerId="PlayerA" />
    );

    const deedProgressLabels = [...html.matchAll(/aria-label="development progress"/g)];
    expect(deedProgressLabels).toHaveLength(2);
    expect(html).toContain('>3/2<');
    expect(html).toContain('>1/2<');
    expect(html).toContain('card-tile perspective-bot is-in-development');
    expect(html).toContain('card-tile is-in-development');
    expect(html).toContain('data-card-id="6"');
    expect(html).toContain('data-in-development="true"');
    expect(html).toContain('data-token-suit="Waves"');
  });

  it('adds player-id data anchor on token rail for animation targeting', () => {
    const player: ObservedPlayerState = {
      id: 'PlayerA',
      crowns: ['30', '31', '32'],
      resources: {
        Moons: 1,
        Suns: 2,
        Waves: 0,
        Leaves: 1,
        Wyrms: 0,
        Knots: 3,
      },
      hand: [],
      handCount: 0,
      handHidden: true,
    };

    const html = renderToStaticMarkup(
      <PlayerTokenRail player={player} side="human" />
    );

    expect(html).toContain('data-token-rail-player-id="PlayerA"');
    expect(html).toContain('data-token-suit="Moons"');
    expect(html).toContain('data-token-suit="Knots"');
  });

  it('renders empty deed token rails for in-development cards with no deed tokens yet', () => {
    const district: DistrictState = {
      id: 'D4',
      markerSuitMask: ['Moons'],
      stacks: {
        PlayerA: {
          developed: [],
          deed: { cardId: '6', progress: 0, tokens: {} },
        },
        PlayerB: {
          developed: [],
        },
      },
    };

    const html = renderToStaticMarkup(
      <DistrictColumn district={district} humanPlayerId="PlayerB" botPlayerId="PlayerA" />
    );

    expect(html).toContain('data-deed-token-rail="left"');
    expect(html).toContain('data-deed-token-rail="right"');
  });
});
