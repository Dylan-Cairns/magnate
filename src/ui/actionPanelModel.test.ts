import { describe, expect, it } from 'vitest';

import type { GameAction, PlayerId } from '../engine/types';
import type { HumanActionListItem } from './actionPresentation';
import {
  actionCategoryForItem,
  actionCategoryLabel,
  buildDevelopOutrightGroupPresentation,
  hasVisibleIncomeChoiceActions,
  isHumanInputActive,
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
    expect(
      actionCategoryForItem({
        kind: 'income-choice-group',
        playerId: 'PlayerA',
        districtId: 'D1',
        cardId: '6',
        options: [
          {
            type: 'choose-income-suit',
            playerId: 'PlayerA',
            districtId: 'D1',
            cardId: '6',
            suit: 'Moons',
          },
        ],
      })
    ).toBe('choose-income-suit');
    expect(actionCategoryLabel('develop-outright')).toBe('Develop Outright');
    expect(actionCategoryLabel('choose-income-suit')).toBe('Choose Income');
    expect(actionCategoryLabel('custom')).toBe('custom');
  });
});

describe('human input activity', () => {
  const endTurnItems: readonly HumanActionListItem[] = [
    { kind: 'action', action: { type: 'end-turn' } },
  ];
  const incomeChoiceItems: readonly HumanActionListItem[] = [
    {
      kind: 'income-choice-group',
      playerId: 'PlayerA',
      districtId: 'D1',
      cardId: '6',
      options: [
        {
          type: 'choose-income-suit',
          playerId: 'PlayerA',
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
        },
      ],
    },
  ];

  function activeWith(overrides: {
    terminal?: boolean;
    activePlayerId?: PlayerId;
    visibleActionItems?: readonly HumanActionListItem[];
    humanActionUiBlockedByAnimation?: boolean;
    isIncomeChoicePhase?: boolean;
  }) {
    return isHumanInputActive({
      terminal: false,
      activePlayerId: 'PlayerA',
      humanPlayerId: 'PlayerA',
      visibleActionItems: endTurnItems,
      humanActionUiBlockedByAnimation: false,
      isIncomeChoicePhase: false,
      ...overrides,
    });
  }

  it('detects visible income choice actions', () => {
    expect(hasVisibleIncomeChoiceActions([])).toBe(false);
    expect(hasVisibleIncomeChoiceActions(endTurnItems)).toBe(false);
    expect(hasVisibleIncomeChoiceActions(incomeChoiceItems)).toBe(true);
  });

  it('is active exactly while human actions are rendered and unblocked', () => {
    expect(activeWith({})).toBe(true);
    expect(activeWith({ terminal: true })).toBe(false);
    expect(activeWith({ humanActionUiBlockedByAnimation: true })).toBe(false);
    expect(activeWith({ activePlayerId: 'PlayerB' })).toBe(false);
    expect(
      activeWith({
        activePlayerId: 'PlayerB',
        visibleActionItems: incomeChoiceItems,
        isIncomeChoicePhase: true,
      })
    ).toBe(true);
  });

  it('stays inactive during income selection when only the bot must choose', () => {
    expect(
      activeWith({ isIncomeChoicePhase: true, visibleActionItems: [] })
    ).toBe(false);
    expect(
      activeWith({
        isIncomeChoicePhase: true,
        visibleActionItems: endTurnItems,
      })
    ).toBe(false);
    expect(
      activeWith({
        activePlayerId: 'PlayerB',
        isIncomeChoicePhase: true,
        visibleActionItems: [],
      })
    ).toBe(false);
    expect(
      activeWith({
        isIncomeChoicePhase: true,
        visibleActionItems: incomeChoiceItems,
      })
    ).toBe(true);
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
