import { beforeEach, describe, expect, it, vi } from 'vitest';

const reactHookHarness = vi.hoisted(() => {
  let cursor = 0;
  let slots: unknown[] = [];

  return {
    beginRender() {
      cursor = 0;
    },
    reset() {
      cursor = 0;
      slots = [];
    },
    useState<T>(initialValue: T) {
      const index = cursor;
      cursor += 1;
      if (!(index in slots)) {
        slots[index] = initialValue;
      }
      const setValue = (nextValue: T) => {
        slots[index] = nextValue;
      };
      return [slots[index] as T, setValue] as const;
    },
  };
});

vi.mock('react', () => ({
  useState: reactHookHarness.useState,
}));

import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../../engine/decisionActor';
import { isTerminal } from '../../engine/scoring';
import { stepToDecision } from '../../engine/session';
import type { GameState } from '../../engine/types';
import { createDevFixtureSession } from '../../dev/fixtures';
import { selectHeuristicAction } from '../../policies/heuristicScorer';
import { useVisibleDiceState } from './useVisibleDiceState';

beforeEach(() => {
  reactHookHarness.reset();
});

describe('useVisibleDiceState', () => {
  it('keeps the final late-game roll visible after GameOver clears it', () => {
    let state = createDevFixtureSession('late-game', 'PlayerA');
    let lastVisibleRoll = state.lastIncomeRoll;
    let lastVisibleTaxSuit = state.lastTaxSuit;
    let visibleDice = VisibleDiceHarness(state);

    for (let decisionCount = 0; decisionCount < 20; decisionCount += 1) {
      if (isTerminal(state)) {
        break;
      }
      if (state.lastIncomeRoll) {
        lastVisibleRoll = state.lastIncomeRoll;
        lastVisibleTaxSuit = state.lastTaxSuit;
      }
      state = playHeuristicDecision(state);
      visibleDice = VisibleDiceHarness(state);
    }

    expect(state.phase).toBe('GameOver');
    expect(state.lastIncomeRoll).toBeUndefined();
    expect(visibleDice).toMatchObject({
      incomeRoll: lastVisibleRoll,
      taxSuit: lastVisibleTaxSuit,
      incomePhase: 'settled',
    });
  });

  it('does not carry retained dice into a new game', () => {
    const state = createDevFixtureSession('late-game', 'PlayerA');

    expect(VisibleDiceHarness(state)).not.toBeNull();
    expect(
      VisibleDiceHarness({
        ...state,
        seed: 'different-game',
        phase: 'StartTurn',
        lastIncomeRoll: undefined,
        lastTaxSuit: undefined,
      })
    ).toBeNull();
  });
});

function VisibleDiceHarness(state: GameState) {
  reactHookHarness.beginRender();
  return useVisibleDiceState({
    animationDice: null,
    gameKey: state.seed,
    incomeRoll: state.lastIncomeRoll,
    taxSuit: state.lastTaxSuit,
    terminal: isTerminal(state),
  });
}

function playHeuristicDecision(state: GameState): GameState {
  const playerId = decisionPlayerIdForState(state);
  if (!playerId) {
    throw new Error('Expected a late-game decision player.');
  }
  const actions = legalActionsForDecisionPlayer(state, playerId);
  const action = selectHeuristicAction({
    state,
    view: toDecisionPlayerView(state, playerId),
    legalActions: actions,
  });
  if (!action) {
    throw new Error('Expected a legal late-game action.');
  }
  return stepToDecision(state, action);
}
