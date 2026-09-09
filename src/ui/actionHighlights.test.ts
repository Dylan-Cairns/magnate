import { describe, expect, it } from 'vitest';
import type { GameAction } from '../engine/types';
import {
  actionsForOpenPicker,
  type ActionPickerState,
} from './actionPickerModel';
import {
  actionHighlightTargets,
  highlightTargetKey,
  sharedActionHighlightTargets,
} from './actionHighlights';

const keys = (action: GameAction) =>
  actionHighlightTargets(action).map(highlightTargetKey);

describe('action highlights', () => {
  it('distinguishes deed and completed-card placement previews', () => {
    const deed = actionHighlightTargets({
      type: 'buy-deed',
      cardId: '6',
      districtId: 'D1',
    });
    const developed = actionHighlightTargets({
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Knots: 1 },
    });
    expect(deed).toContainEqual({
      kind: 'district-lane',
      cardId: '6',
      districtId: 'D1',
      placement: 'deed',
    });
    expect(developed).toContainEqual({
      kind: 'district-lane',
      cardId: '6',
      districtId: 'D1',
      placement: 'developed',
    });
  });

  it('keeps shared outright resources while a district submenu is open', () => {
    const actions: GameAction[] = ['D1', 'D2'].map((districtId) => ({
      type: 'develop-outright',
      cardId: '6',
      districtId,
      payment: { Moons: 1, Knots: 1 },
    }));
    const picker: ActionPickerState = {
      kind: 'develop-outright-district',
      cardId: '6',
      top: 0,
      left: 0,
    };
    expect(
      sharedActionHighlightTargets(actionsForOpenPicker(picker, actions)).map(
        highlightTargetKey
      )
    ).toEqual(['hand-card:6', 'resource:Moons', 'resource:Knots']);
  });

  it('keeps a chosen trade source without guessing the receiving suit', () => {
    const actions: GameAction[] = [
      { type: 'trade', give: 'Moons', receive: 'Suns' },
      { type: 'trade', give: 'Moons', receive: 'Waves' },
      { type: 'trade', give: 'Knots', receive: 'Suns' },
    ];
    for (const picker of [
      { kind: 'trade', give: 'Moons', top: 0, left: 0 },
      { kind: 'trade-combined', selectedGive: 'Moons', top: 0, left: 0 },
    ] satisfies ActionPickerState[]) {
      expect(
        sharedActionHighlightTargets(actionsForOpenPicker(picker, actions))
      ).toEqual([{ kind: 'resource', suit: 'Moons' }]);
    }
    expect(
      actionsForOpenPicker(
        { kind: 'trade', give: 'Moons', top: 0, left: 0 },
        []
      )
    ).toEqual([]);
  });

  it('retains a selected outright destination while choosing payment', () => {
    const actions: GameAction[] = [
      {
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D1',
        payment: { Waves: 1, Leaves: 3 },
      },
      {
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D1',
        payment: { Waves: 3, Leaves: 1 },
      },
      {
        type: 'develop-outright',
        cardId: '14',
        districtId: 'D2',
        payment: { Waves: 3, Leaves: 1 },
      },
    ];
    expect(
      sharedActionHighlightTargets(
        actionsForOpenPicker(
          {
            kind: 'develop-outright-combined',
            cardId: '14',
            selectedDistrictId: 'D1',
            top: 0,
            left: 0,
          },
          actions
        )
      )
    ).toContainEqual({
      kind: 'district-lane',
      cardId: '14',
      districtId: 'D1',
      placement: 'developed',
    });
  });

  it.each<[GameAction, string[]]>([
    [
      { type: 'buy-deed', cardId: '6', districtId: 'D2' },
      [
        'hand-card:6',
        'resource:Moons',
        'resource:Knots',
        'district-lane:D2:6:deed',
      ],
    ],
    [
      {
        type: 'develop-outright',
        cardId: '6',
        districtId: 'D1',
        payment: { Moons: 1, Knots: 1, Suns: 0 },
      },
      [
        'hand-card:6',
        'resource:Moons',
        'resource:Knots',
        'district-lane:D1:6:developed',
      ],
    ],
    [
      {
        type: 'develop-deed',
        cardId: '6',
        districtId: 'D1',
        tokens: { Knots: 1, Moons: 0 },
      },
      ['played-card:6', 'resource:Knots'],
    ],
    [
      { type: 'sell-card', cardId: '6' },
      ['hand-card:6', 'resource:Moons', 'resource:Knots', 'pile:discard'],
    ],
    [
      { type: 'trade', give: 'Suns', receive: 'Waves' },
      ['resource:Suns', 'resource:Waves'],
    ],
    [
      {
        type: 'choose-income-suit',
        playerId: 'PlayerA',
        cardId: '6',
        districtId: 'D1',
        suit: 'Knots',
      },
      ['played-card:6', 'resource:Knots'],
    ],
    [{ type: 'end-turn' }, []],
  ])('maps $type to its exact UI targets', (action, expected) => {
    expect(keys(action)).toEqual(expected);
  });

  it('highlights an Ace suit once for both buying and selling', () => {
    expect(keys({ type: 'buy-deed', cardId: '0', districtId: 'D1' })).toEqual([
      'hand-card:0',
      'resource:Knots',
      'district-lane:D1:0:deed',
    ]);
    expect(keys({ type: 'sell-card', cardId: '0' })).toEqual([
      'hand-card:0',
      'resource:Knots',
      'pile:discard',
    ]);
  });

  it('does not guess a destination for a grouped purchase', () => {
    const targets = sharedActionHighlightTargets([
      { type: 'buy-deed', cardId: '6', districtId: 'D1' },
      { type: 'buy-deed', cardId: '6', districtId: 'D2' },
    ]);
    expect(targets.map(highlightTargetKey)).toEqual([
      'hand-card:6',
      'resource:Moons',
      'resource:Knots',
    ]);
  });

  it('does not guess a payment suit for partial development', () => {
    const targets = sharedActionHighlightTargets([
      {
        type: 'develop-deed',
        cardId: '6',
        districtId: 'D1',
        tokens: { Moons: 1 },
      },
      {
        type: 'develop-deed',
        cardId: '6',
        districtId: 'D1',
        tokens: { Knots: 1 },
      },
    ]);
    expect(targets.map(highlightTargetKey)).toEqual(['played-card:6']);
    expect(sharedActionHighlightTargets([])).toEqual([]);
  });
});
