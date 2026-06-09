import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import {
  PLAYER_A,
  PLAYER_B,
  makeGameState,
  makePlayer,
  makeResources,
  withDeed,
} from '../engine/__tests__/fixtures';
import type { GameAction } from '../engine/types';
import {
  actionStableKey,
  buildHumanActionList,
  buildTradeSourceGroups,
  buildPickerOptions,
  pickerStillLegal,
  pickerTitle,
  type ActionPickerQuery,
} from './actionPresentation';
import { SUIT_TEXT_TOKEN } from './suitIcons';

describe('buildHumanActionList', () => {
  it('groups trade actions by give suit', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Suns: 3 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const grouped = buildHumanActionList(legalActions(state));
    const tradeGroups = grouped.filter(
      (
        item
      ): item is Extract<(typeof grouped)[number], { kind: 'trade-group' }> =>
        item.kind === 'trade-group'
    );

    expect(tradeGroups).toHaveLength(2);
    expect(new Set(tradeGroups.map((group) => group.give))).toEqual(
      new Set(['Moons', 'Suns'])
    );
    expect(tradeGroups.every((group) => group.options.length === 5)).toBe(true);
  });

  it('groups deed development actions by card and district', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const grouped = buildHumanActionList(legalActions(state));
    const deedGroups = grouped.filter(
      (
        item
      ): item is Extract<
        (typeof grouped)[number],
        { kind: 'develop-deed-group' }
      > => item.kind === 'develop-deed-group'
    );

    expect(deedGroups).toHaveLength(1);
    expect(deedGroups[0].cardId).toBe('6');
    expect(deedGroups[0].districtId).toBe('D1');
    expect(deedGroups[0].options).toHaveLength(2);
  });

  it('places trade options after non-trade options during pre-card action windows', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 3 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const grouped = buildHumanActionList(legalActions(state));
    const firstTradeIndex = grouped.findIndex(
      (item) => item.kind === 'trade-group'
    );
    const nonTradeIndex = grouped.findIndex(
      (item) => item.kind !== 'trade-group'
    );

    expect(firstTradeIndex).toBeGreaterThan(nonTradeIndex);
    expect(
      grouped
        .slice(firstTradeIndex)
        .every((item) => item.kind === 'trade-group')
    ).toBe(true);
  });

  it('places trade options before end-turn during post-card action windows', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const grouped = buildHumanActionList(legalActions(state));
    const firstTradeIndex = grouped.findIndex(
      (item) => item.kind === 'trade-group'
    );
    const endTurnIndex = grouped.findIndex(
      (item) => item.kind === 'action' && item.action.type === 'end-turn'
    );

    expect(firstTradeIndex).toBeGreaterThanOrEqual(0);
    expect(endTurnIndex).toBeGreaterThan(firstTradeIndex);
    expect(grouped.at(-1)).toMatchObject({
      kind: 'action',
      action: { type: 'end-turn' },
    });
  });

  it('orders pre-card action categories by fixed precedence', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 3, Knots: 3, Suns: 3 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const grouped = buildHumanActionList(legalActions(state));
    expect(actionCategorySequence(grouped)).toEqual([
      'develop-outright',
      'buy-deed',
      'sell-card',
      'develop-deed',
      'trade',
    ]);
  });

  it('groups outright development actions by card instead of payment pattern', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({
            Moons: 6,
            Suns: 6,
            Waves: 6,
            Leaves: 6,
            Wyrms: 6,
            Knots: 6,
          }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const actions = legalActions(state).filter(
      (action): action is Extract<GameAction, { type: 'develop-outright' }> =>
        action.type === 'develop-outright' && action.cardId === '6'
    );
    const grouped = buildHumanActionList(legalActions(state));
    const outrightGroups = grouped.filter(
      (
        item
      ): item is Extract<
        (typeof grouped)[number],
        { kind: 'develop-outright-group' }
      > => item.kind === 'develop-outright-group' && item.cardId === '6'
    );

    expect(actions.length).toBeGreaterThan(1);
    expect(outrightGroups).toHaveLength(1);
    expect(outrightGroups[0].options).toHaveLength(actions.length);
  });

  it('keeps income choices flat when only one income card is pending', () => {
    const actions: GameAction[] = [
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Moons',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Knots',
      },
    ];

    const grouped = buildHumanActionList(actions);

    expect(grouped).toEqual([
      { kind: 'action', action: actions[0] },
      { kind: 'action', action: actions[1] },
    ]);
  });

  it('groups income choices by card when multiple income cards are pending', () => {
    const actions: GameAction[] = [
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Moons',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Knots',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D2',
        cardId: '7',
        suit: 'Suns',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D2',
        cardId: '7',
        suit: 'Wyrms',
      },
    ];

    const grouped = buildHumanActionList(actions);
    const incomeGroups = grouped.filter(
      (
        item
      ): item is Extract<
        (typeof grouped)[number],
        { kind: 'income-choice-group' }
      > => item.kind === 'income-choice-group'
    );

    expect(incomeGroups).toHaveLength(2);
    expect(incomeGroups.map((group) => group.cardId)).toEqual(['6', '7']);
    expect(incomeGroups.map((group) => group.options.length)).toEqual([2, 2]);
  });

  it('orders post-card action categories by fixed precedence subset', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const grouped = buildHumanActionList(legalActions(state));
    expect(actionCategorySequence(grouped)).toEqual([
      'develop-deed',
      'trade',
      'end-turn',
    ]);
  });
});

describe('buildTradeSourceGroups', () => {
  it('returns grouped trade sources in first-seen action order', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Suns: 3 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const groups = buildTradeSourceGroups(legalActions(state));
    expect(groups).toHaveLength(2);
    expect(groups.map((group) => group.give)).toEqual(['Moons', 'Suns']);
    expect(groups.every((group) => group.options.length === 5)).toBe(true);
  });
});

describe('picker helpers', () => {
  it('pickerStillLegal tracks grouped deed payment options', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });
    const actions = legalActions(state);

    const picker: ActionPickerQuery = {
      kind: 'deed-payment',
      cardId: '6',
      districtId: 'D1',
    };
    expect(pickerStillLegal(picker, actions)).toBe(true);

    const narrowed = actions.filter((action) =>
      action.type === 'develop-deed' ? action.tokens.Moons === 1 : true
    );
    expect(pickerStillLegal(picker, narrowed)).toBe(false);
  });

  it('buildPickerOptions and pickerTitle for trade reflect available receives', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const actions = legalActions(state);
    const picker: ActionPickerQuery = { kind: 'trade', give: 'Moons' };

    const options = buildPickerOptions(picker, actions, SUIT_TEXT_TOKEN);
    expect(options).toHaveLength(5);
    expect(options.map((option) => option.label)).toContain('{Suns} x1');
    expect(pickerTitle(picker, SUIT_TEXT_TOKEN)).toBe('Trade {Moons}x3 for');
  });

  it('builds district-then-payment picker options for develop-outright', () => {
    const actions: GameAction[] = [
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
        payment: { Moons: 3 },
      },
    ];

    const districtPicker: ActionPickerQuery = {
      kind: 'develop-outright-district',
      cardId: '6',
    };
    const districtOptions = buildPickerOptions(
      districtPicker,
      actions,
      SUIT_TEXT_TOKEN
    );
    expect(districtOptions.map((option) => option.label)).toEqual(['D1', 'D2']);
    expect(pickerStillLegal(districtPicker, actions)).toBe(true);
    expect(pickerTitle(districtPicker, SUIT_TEXT_TOKEN)).toMatch(
      /^Develop .* in$/
    );

    const paymentPicker: ActionPickerQuery = {
      kind: 'develop-outright-payment',
      cardId: '6',
      districtId: 'D1',
    };
    const paymentOptions = buildPickerOptions(
      paymentPicker,
      actions,
      SUIT_TEXT_TOKEN
    );
    expect(paymentOptions).toHaveLength(2);
    expect(paymentOptions.map((option) => option.label)).toContain(
      '{Moons}x2 {Knots}x1'
    );
    expect(paymentOptions.map((option) => option.label)).toContain(
      '{Moons}x1 {Knots}x2'
    );
    expect(pickerStillLegal(paymentPicker, actions)).toBe(true);
    expect(pickerTitle(paymentPicker, SUIT_TEXT_TOKEN)).toMatch(
      /^Develop .* in D1 with$/
    );
  });

  it('builds income-choice picker options by card', () => {
    const actions: GameAction[] = [
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Moons',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Knots',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D2',
        cardId: '7',
        suit: 'Suns',
      },
    ];
    const picker: ActionPickerQuery = {
      kind: 'income-choice',
      playerId: PLAYER_A,
      cardId: '6',
      districtId: 'D1',
    };

    const options = buildPickerOptions(picker, actions, SUIT_TEXT_TOKEN);

    expect(options.map((option) => option.label)).toEqual([
      '{Moons} x1',
      '{Knots} x1',
    ]);
    expect(options.map((option) => option.action)).toEqual([
      actions[0],
      actions[1],
    ]);
    expect(pickerStillLegal(picker, actions)).toBe(true);
    expect(pickerStillLegal(picker, [actions[2]])).toBe(false);
    expect(pickerTitle(picker, SUIT_TEXT_TOKEN)).toBe(
      'Choose income 2{Moons}{Knots} in D1'
    );
  });
});

describe('actionStableKey', () => {
  it('produces deterministic keys for equivalent action payloads', () => {
    const base: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D2',
      payment: { Moons: 1, Knots: 1 },
    };
    const clone: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D2',
      payment: { Knots: 1, Moons: 1 },
    };

    expect(actionStableKey(base)).toBe(actionStableKey(clone));
  });
});

function actionCategorySequence(
  grouped: ReturnType<typeof buildHumanActionList>
): string[] {
  const sequence: string[] = [];

  for (const item of grouped) {
    const category =
      item.kind === 'action'
        ? item.action.type
        : item.kind === 'trade-group'
          ? 'trade'
          : item.kind === 'buy-deed-group'
            ? 'buy-deed'
            : item.kind === 'develop-deed-group'
              ? 'develop-deed'
              : item.kind === 'develop-outright-group'
                ? 'develop-outright'
                : 'choose-income-suit';

    if (sequence.at(-1) !== category) {
      sequence.push(category);
    }
  }

  return sequence;
}
