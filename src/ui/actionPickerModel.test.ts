import { describe, expect, it } from 'vitest';

import type { GameAction } from '../engine/types';
import { paymentSignature } from './actionPresentation';
import {
  actionPickerTitle,
  buildDevelopOutrightCompositeOptions,
  developOutrightCompositePickerStillLegal,
  resolveDevelopOutrightCompositeAction,
  resolveTradeCompositeAction,
  toPickerQuery,
  tradeActionsForPicker,
  tradeCompositePickerStillLegal,
  tradeReceiveOptions,
  type ActionPickerState,
} from './actionPickerModel';
import { SUIT_TEXT_TOKEN } from './suitIcons';

const TRADE_ACTIONS: GameAction[] = [
  { type: 'trade', give: 'Moons', receive: 'Suns' },
  { type: 'trade', give: 'Moons', receive: 'Waves' },
  { type: 'trade', give: 'Suns', receive: 'Moons' },
  { type: 'trade', give: 'Suns', receive: 'Waves' },
];

const OUTRIGHT_ACTIONS: GameAction[] = [
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D1',
    payment: { Moons: 2, Knots: 1 },
  },
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D1',
    payment: { Moons: 1, Knots: 2 },
  },
  {
    type: 'develop-outright',
    cardId: '6',
    districtId: 'D2',
    payment: { Moons: 2, Knots: 1 },
  },
];

describe('combined trade picker model', () => {
  it('preserves receive order and resolves a complete selection to one action', () => {
    const actions = tradeActionsForPicker(TRADE_ACTIONS);

    expect(tradeReceiveOptions(actions)).toEqual(['Suns', 'Waves', 'Moons']);
    expect(
      resolveTradeCompositeAction(actions, {
        selectedGive: 'Suns',
        selectedReceive: 'Waves',
      })
    ).toEqual({ type: 'trade', give: 'Suns', receive: 'Waves' });
  });

  it('closes when trade sources collapse or the selected give suit becomes illegal', () => {
    expect(tradeCompositePickerStillLegal({}, TRADE_ACTIONS)).toBe(true);
    expect(
      tradeCompositePickerStillLegal(
        { selectedGive: 'Suns' },
        TRADE_ACTIONS.filter(
          (action) => action.type === 'trade' && action.give === 'Moons'
        )
      )
    ).toBe(false);
    expect(
      tradeCompositePickerStillLegal(
        {},
        TRADE_ACTIONS.filter(
          (action) => action.type === 'trade' && action.give === 'Moons'
        )
      )
    ).toBe(false);
  });
});

describe('combined develop-outright picker model', () => {
  it('preserves first-seen district and payment options and resolves a complete selection', () => {
    const model = buildDevelopOutrightCompositeOptions(OUTRIGHT_ACTIONS, '6');
    const selectedPaymentKey = paymentSignature({ Moons: 1, Knots: 2 });

    expect(model.districtOptions.map((action) => action.districtId)).toEqual([
      'D1',
      'D2',
    ]);
    expect(model.paymentOptions.map(([key]) => key)).toEqual([
      paymentSignature({ Moons: 2, Knots: 1 }),
      selectedPaymentKey,
    ]);
    expect(
      resolveDevelopOutrightCompositeAction(model.outrightOptions, {
        cardId: '6',
        selectedDistrictId: 'D1',
        selectedPaymentKey,
      })
    ).toEqual(OUTRIGHT_ACTIONS[1]);
  });

  it('closes when options collapse or a selected district or payment disappears', () => {
    expect(
      developOutrightCompositePickerStillLegal(
        { cardId: '6' },
        OUTRIGHT_ACTIONS
      )
    ).toBe(true);
    expect(
      developOutrightCompositePickerStillLegal(
        { cardId: '6', selectedDistrictId: 'D3' },
        OUTRIGHT_ACTIONS
      )
    ).toBe(false);
    expect(
      developOutrightCompositePickerStillLegal(
        { cardId: '6', selectedPaymentKey: 'missing-payment' },
        OUTRIGHT_ACTIONS
      )
    ).toBe(false);
    expect(
      developOutrightCompositePickerStillLegal(
        { cardId: '6' },
        OUTRIGHT_ACTIONS.slice(0, 1)
      )
    ).toBe(false);
  });
});

describe('positioned picker model', () => {
  it('converts every standard positioned picker variant back to a query', () => {
    const pickers: ActionPickerState[] = [
      { kind: 'trade', give: 'Moons', top: 1, left: 2 },
      {
        kind: 'district',
        actionType: 'buy-deed',
        cardId: '6',
        top: 1,
        left: 2,
      },
      {
        kind: 'develop-outright-district',
        cardId: '6',
        top: 1,
        left: 2,
      },
      {
        kind: 'develop-outright-payment',
        cardId: '6',
        districtId: 'D1',
        top: 1,
        left: 2,
      },
      {
        kind: 'deed-payment',
        cardId: '6',
        districtId: 'D1',
        top: 1,
        left: 2,
      },
    ];

    expect(
      pickers.map((picker) => {
        if (
          picker.kind === 'trade-combined' ||
          picker.kind === 'develop-outright-combined'
        ) {
          throw new Error('Unexpected composite picker');
        }
        return toPickerQuery(picker);
      })
    ).toEqual([
      { kind: 'trade', give: 'Moons' },
      { kind: 'district', actionType: 'buy-deed', cardId: '6' },
      { kind: 'develop-outright-district', cardId: '6' },
      { kind: 'develop-outright-payment', cardId: '6', districtId: 'D1' },
      { kind: 'deed-payment', cardId: '6', districtId: 'D1' },
    ]);
  });

  it('builds stable titles for standard and composite pickers', () => {
    expect(
      actionPickerTitle(
        { kind: 'trade-combined', top: 1, left: 2 },
        SUIT_TEXT_TOKEN
      )
    ).toBe('Trade resources');
    expect(
      actionPickerTitle(
        {
          kind: 'develop-outright-combined',
          cardId: '6',
          top: 1,
          left: 2,
        },
        SUIT_TEXT_TOKEN
      )
    ).toBe('Develop 2{Moons}{Knots}');
    expect(
      actionPickerTitle(
        { kind: 'trade', give: 'Moons', top: 1, left: 2 },
        SUIT_TEXT_TOKEN
      )
    ).toBe('Trade {Moons}x3 for');
  });
});
