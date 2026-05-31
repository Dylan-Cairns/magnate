import { describe, expect, it } from 'vitest';

import type { GameAction } from '../engine/types';
import {
  actionCategoryForItem,
  actionCategoryLabel,
  buildDevelopOutrightGroupPresentation,
} from './actionPanelModel';

const OUTRIGHT_ACTIONS: Array<
  Extract<GameAction, { type: 'develop-outright' }>
> = [
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D1',
    payment: { Moons: 2, Knots: 1 },
  },
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D2',
    payment: { Moons: 2, Knots: 1 },
  },
];

describe('action panel categories', () => {
  it('maps grouped and direct action items to stable labels', () => {
    expect(
      actionCategoryForItem({
        kind: 'trade-group',
        give: 'Moons',
        options: [{ type: 'trade', give: 'Moons', receive: 'Suns' }],
      })
    ).toBe('trade');
    expect(
      actionCategoryForItem({
        kind: 'action',
        action: { type: 'end-turn' },
      })
    ).toBe('end-turn');
    expect(actionCategoryLabel('develop-outright')).toBe('Develop Outright');
    expect(actionCategoryLabel('custom')).toBe('custom');
  });
});

describe('develop outright group presentation', () => {
  it('detects a shared payment pattern across districts', () => {
    expect(buildDevelopOutrightGroupPresentation(OUTRIGHT_ACTIONS)).toEqual({
      districtCount: 2,
      hasSinglePaymentPattern: true,
      firstPayment: { Moons: 2, Knots: 1 },
    });
  });

  it('detects multiple payment patterns', () => {
    expect(
      buildDevelopOutrightGroupPresentation([
        ...OUTRIGHT_ACTIONS,
        {
          type: 'develop-outright',
          cardId: '6',
          districtId: 'D1',
          payment: { Moons: 1, Knots: 2 },
        },
      ])
    ).toEqual({
      districtCount: 2,
      hasSinglePaymentPattern: false,
      firstPayment: { Moons: 2, Knots: 1 },
    });
  });
});
