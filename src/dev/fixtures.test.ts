import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import {
  humanActionsAcceptingInputForState,
  incomeChoiceActionsForPlayer,
} from '../ui/gameControllerModel';
import { buildHumanActionList } from '../ui/actionPresentation';
import {
  createDevFixtureSession,
  devFixtureIdFromSearch,
} from './fixtures';

describe('dev fixtures', () => {
  it('derives multiple partial-income choices from real CollectIncome resolution', () => {
    const state = createDevFixtureSession('multi-income', 'PlayerA');

    expect(state.phase).toBe('CollectIncome');
    expect(state.lastIncomeRoll).toEqual({ die1: 2, die2: 2 });
    expect(state.pendingIncomeChoices?.map((choice) => choice.cardId)).toEqual([
      '6',
      '7',
      '8',
    ]);
    expect(state.pendingIncomeChoices?.map((choice) => choice.playerId)).toEqual(
      ['PlayerA', 'PlayerA', 'PlayerB']
    );
    expect(
      state.pendingIncomeChoices?.every((choice) => choice.suits.length === 2)
    ).toBe(true);
  });

  it('presents the human side as one grouped action per human income card', () => {
    const state = createDevFixtureSession('multi-income', 'PlayerA');
    const actions = humanActionsAcceptingInputForState({
      state,
      humanPlayerId: 'PlayerA',
      actionCommitPending: false,
      allowHumanActionsWhileCommitPending: false,
    });
    const grouped = buildHumanActionList(actions);

    expect(legalActions(state)).toHaveLength(6);
    expect(grouped).toHaveLength(2);
    expect(
      grouped.map((item) =>
        item.kind === 'income-choice-group' ? item.cardId : null
      )
    ).toEqual(['6', '7']);
  });

  it('also gives the bot owned partial-income choices', () => {
    const state = createDevFixtureSession('multi-income', 'PlayerA');
    const botIncomeActions = incomeChoiceActionsForPlayer(
      legalActions(state),
      'PlayerB'
    );

    expect(botIncomeActions.map((action) => action.cardId)).toEqual(['8', '8']);
    expect(botIncomeActions.map((action) => action.suit)).toEqual([
      'Waves',
      'Leaves',
    ]);
  });

  it('parses only known dev fixture query values', () => {
    expect(devFixtureIdFromSearch('?fixture=multi-income')).toBe(
      'multi-income'
    );
    expect(devFixtureIdFromSearch('?fixture=unknown')).toBeNull();
    expect(devFixtureIdFromSearch('')).toBeNull();
  });
});
