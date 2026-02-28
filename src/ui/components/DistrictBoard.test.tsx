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
});

