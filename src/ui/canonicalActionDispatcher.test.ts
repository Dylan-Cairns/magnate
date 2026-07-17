import { describe, expect, it, vi } from 'vitest';

import {
  makeGameState,
  PLAYER_A,
  PLAYER_B,
} from '../engine/__tests__/fixtures';
import type { GameAction, GameState } from '../engine/types';
import type { ActionDispatchDependencies } from './actionDispatcher';
import { prepareCanonicalActionDispatch } from './canonicalActionDispatcher';

const TRADE: GameAction = {
  type: 'trade',
  give: 'Moons',
  receive: 'Suns',
};

describe('prepareCanonicalActionDispatch', () => {
  it('builds an actor-neutral transition with a unique canonical ordinal', () => {
    const currentState = makeGameState({ seed: 'canonical-dispatch', turn: 4 });
    const nextState = makeGameState({ seed: 'canonical-dispatch', turn: 4 });

    const plan = prepareCanonicalActionDispatch({
      currentState,
      sourceState: currentState,
      action: TRADE,
      actingPlayerId: PLAYER_A,
      actionOrdinal: 7,
      dependencies: dependenciesReturning(nextState),
    });

    expect(plan).toMatchObject({
      previousState: currentState,
      nextState,
      action: TRADE,
      actingPlayerId: PLAYER_A,
      actionOrdinal: 7,
      transactionId: 'canonical-dispatch:action:7:4:ActionWindow:trade',
    });
  });

  it('chains consecutive plans from each fully resolved canonical state', () => {
    const state0 = makeGameState({ seed: 'canonical-chain' });
    const state1 = makeGameState({ seed: 'canonical-chain' });
    const state2 = makeGameState({ seed: 'canonical-chain' });
    const stepToDecision = vi
      .fn<(state: GameState, action: GameAction) => GameState>()
      .mockReturnValueOnce(state1)
      .mockReturnValueOnce(state2);
    const dependencies = { stepToDecision };

    const first = prepareCanonicalActionDispatch({
      currentState: state0,
      sourceState: state0,
      action: TRADE,
      actingPlayerId: PLAYER_A,
      actionOrdinal: 0,
      dependencies,
    });
    const second = prepareCanonicalActionDispatch({
      currentState: first.nextState,
      sourceState: state1,
      action: TRADE,
      actingPlayerId: PLAYER_A,
      actionOrdinal: 1,
      dependencies,
    });

    expect(stepToDecision.mock.calls.map(([state]) => state)).toEqual([
      state0,
      state1,
    ]);
    expect(second.previousState).toBe(first.nextState);
    expect(second.nextState).toBe(state2);
    expect(second.transactionId).not.toBe(first.transactionId);
  });

  it('rejects stale source state before invoking the engine', () => {
    const currentState = makeGameState();
    const staleState = makeGameState();
    const stepToDecision = vi.fn(() => makeGameState());

    expect(() =>
      prepareCanonicalActionDispatch({
        currentState,
        sourceState: staleState,
        action: TRADE,
        actingPlayerId: PLAYER_A,
        actionOrdinal: 0,
        dependencies: { stepToDecision },
      })
    ).toThrow('stale canonical state');
    expect(stepToDecision).not.toHaveBeenCalled();
  });

  it('requires normal actions to match turn ownership and income choices to match their owner', () => {
    const currentState = makeGameState();
    const dependencies = dependenciesReturning(makeGameState());

    expect(() =>
      prepareCanonicalActionDispatch({
        currentState,
        sourceState: currentState,
        action: TRADE,
        actingPlayerId: PLAYER_B,
        actionOrdinal: 0,
        dependencies,
      })
    ).toThrow('Action actor mismatch');

    const incomeAction: GameAction = {
      type: 'choose-income-suit',
      playerId: PLAYER_B,
      districtId: 'D1',
      cardId: '6',
      suit: 'Moons',
    };
    expect(() =>
      prepareCanonicalActionDispatch({
        currentState,
        sourceState: currentState,
        action: incomeAction,
        actingPlayerId: PLAYER_A,
        actionOrdinal: 1,
        dependencies,
      })
    ).toThrow('Action actor mismatch');
  });

  it('rejects invalid action ordinals before invoking the engine', () => {
    const currentState = makeGameState();
    const stepToDecision = vi.fn(() => makeGameState());

    expect(() =>
      prepareCanonicalActionDispatch({
        currentState,
        sourceState: currentState,
        action: TRADE,
        actingPlayerId: PLAYER_A,
        actionOrdinal: -1,
        dependencies: { stepToDecision },
      })
    ).toThrow('Invalid canonical action ordinal');
    expect(stepToDecision).not.toHaveBeenCalled();
  });
});

function dependenciesReturning(
  nextState: GameState
): ActionDispatchDependencies {
  return {
    stepToDecision: () => nextState,
  };
}
