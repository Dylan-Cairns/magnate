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
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3, Suns: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const grouped = buildHumanActionList(legalActions(state));
    const tradeGroups = grouped.filter(
      (item): item is Extract<(typeof grouped)[number], { kind: 'trade-group' }> =>
        item.kind === 'trade-group'
    );

    expect(tradeGroups).toHaveLength(2);
    expect(new Set(tradeGroups.map((group) => group.give))).toEqual(new Set(['Moons', 'Suns']));
    expect(tradeGroups.every((group) => group.options.length === 5)).toBe(true);
  });

  it('groups deed development actions by card and district', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 1, Knots: 1 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, { cardId: '6', progress: 0, tokens: {} });

    const grouped = buildHumanActionList(legalActions(state));
    const deedGroups = grouped.filter(
      (item): item is Extract<(typeof grouped)[number], { kind: 'develop-deed-group' }> =>
        item.kind === 'develop-deed-group'
    );

    expect(deedGroups).toHaveLength(1);
    expect(deedGroups[0].cardId).toBe('6');
    expect(deedGroups[0].districtId).toBe('D1');
    expect(deedGroups[0].options).toHaveLength(2);
  });
});

describe('picker helpers', () => {
  it('pickerStillLegal tracks grouped deed payment options', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 1, Knots: 1 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, { cardId: '6', progress: 0, tokens: {} });
    const actions = legalActions(state);

    const picker: ActionPickerQuery = { kind: 'deed-payment', cardId: '6', districtId: 'D1' };
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
