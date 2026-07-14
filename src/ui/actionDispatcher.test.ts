import { describe, expect, it } from 'vitest';

import { makeGameState } from '../engine/__tests__/fixtures';
import type { GameAction, GameState } from '../engine/types';
import {
  prepareActionDispatch,
  type ActionDispatchDependencies,
} from './actionDispatcher';

const ACTION: GameAction = {
  type: 'trade',
  give: 'Moons',
  receive: 'Suns',
};

describe('prepareActionDispatch', () => {
  it('validates through the engine before returning a dispatch plan', () => {
    const previousState = makeGameState();
    const nextState = makeGameState();
    const calls: string[] = [];

    prepareActionDispatch({
      previousState,
      action: ACTION,
      dependencies: makeDependencies(nextState, calls),
    });

    expect(calls).toEqual(['step']);
  });

  it('propagates engine validation failures', () => {
    const calls: string[] = [];
    const dependencies = makeDependencies(makeGameState(), calls);
    dependencies.stepToDecision = () => {
      calls.push('step');
      throw new Error('Illegal action');
    };

    expect(() =>
      prepareActionDispatch({
        previousState: makeGameState(),
        action: ACTION,
        dependencies,
      })
    ).toThrow('Illegal action');
    expect(calls).toEqual(['step']);
  });

  it('reports when the transition enters terminal state', () => {
    const previousState = makeGameState();
    const nextState = makeGameState({ phase: 'GameOver' });
    const calls: string[] = [];

    const plan = prepareActionDispatch({
      previousState,
      action: ACTION,
      dependencies: makeDependencies(nextState, calls),
    });

    expect(calls).toEqual(['step']);
    expect(plan).toMatchObject({
      previousState,
      nextState,
      action: ACTION,
      enteredTerminal: true,
    });
  });
});

function makeDependencies(
  nextState: GameState,
  calls: string[] = []
): ActionDispatchDependencies {
  return {
    stepToDecision: () => {
      calls.push('step');
      return nextState;
    },
  };
}
