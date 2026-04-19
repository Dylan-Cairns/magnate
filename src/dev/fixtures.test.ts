import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { isTerminal, scoreLive } from '../engine/scoring';
import type { GameState, PlayerId } from '../engine/types';
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

  it('rolls out a legal late-game board state close to game end', () => {
    const state = createDevFixtureSession('late-game', 'PlayerA');
    const developedCounts = developedCountsByPlayer(state);

    expect(state.phase).toBe('ActionWindow');
    expect(state.finalTurnsRemaining).toBe(2);
    expect(state.deck.reshuffles).toBe(2);
    expect(isTerminal(state)).toBe(false);
    expect(legalActions(state).length).toBeGreaterThan(0);
    expect(developedCounts.PlayerA).toBeGreaterThanOrEqual(6);
    expect(developedCounts.PlayerB).toBeGreaterThanOrEqual(6);
    expect(scoreLive(state).winner).toMatch(/PlayerA|PlayerB|Draw/);
  });

  it('parses only known dev fixture query values', () => {
    expect(devFixtureIdFromSearch('?fixture=multi-income')).toBe(
      'multi-income'
    );
    expect(devFixtureIdFromSearch('?fixture=late-game')).toBe('late-game');
    expect(devFixtureIdFromSearch('?fixture=unknown')).toBeNull();
    expect(devFixtureIdFromSearch('')).toBeNull();
  });
});

function developedCountsByPlayer(state: GameState): Record<PlayerId, number> {
  return state.districts.reduce<Record<PlayerId, number>>(
    (counts, district) => ({
      PlayerA: counts.PlayerA + district.stacks.PlayerA.developed.length,
      PlayerB: counts.PlayerB + district.stacks.PlayerB.developed.length,
    }),
    { PlayerA: 0, PlayerB: 0 }
  );
}
