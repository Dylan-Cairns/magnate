import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import type { DistrictState } from '../../engine/types';
import { DistrictColumn } from './DistrictBoard';

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

    expect(html).toContain('class="district-lane-score is-leading"');
    const nonLeadingScoreMatches = [...html.matchAll(/class="district-lane-score"/g)];
    expect(nonLeadingScoreMatches).toHaveLength(1);
  });
});
