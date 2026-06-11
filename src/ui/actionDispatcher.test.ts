import { describe, expect, it } from 'vitest';

import { makeGameState } from '../engine/__tests__/fixtures';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import type { CardFlight, ResourceFlight } from './animations/types';
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
  it('validates before animation planning and forwards the acting player', () => {
    const previousState = makeGameState();
    const nextState = makeGameState();
    const calls: string[] = [];
    const actingPlayerIds: PlayerId[] = [];
    const dependencies = makeDependencies(nextState, calls);
    dependencies.collectDeedResourceFlights = (
      _state,
      _action,
      actingPlayerId
    ) => {
      calls.push('deed');
      actingPlayerIds.push(actingPlayerId);
      return [];
    };
    dependencies.collectCardPlayFlights = (
      _state,
      _nextState,
      _action,
      actingPlayerId
    ) => {
      calls.push('card');
      actingPlayerIds.push(actingPlayerId);
      return [];
    };

    prepareActionDispatch({
      previousState,
      action: ACTION,
      actingPlayerId: 'PlayerB',
      animationsEnabled: true,
      makeResourceFlightId: makeIds('resource'),
      makeCardFlightId: makeIds('card'),
      dependencies,
    });

    expect(calls).toEqual([
      'step',
      'deed',
      'income-choice',
      'card',
      'terminal',
      'turn-cycle',
    ]);
    expect(actingPlayerIds).toEqual(['PlayerB', 'PlayerB']);
  });

  it('does not run UI planners when engine validation fails', () => {
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
        actingPlayerId: 'PlayerA',
        animationsEnabled: true,
        makeResourceFlightId: makeIds('resource'),
        makeCardFlightId: makeIds('card'),
        dependencies,
      })
    ).toThrow('Illegal action');
    expect(calls).toEqual(['step']);
  });

  it('skips DOM-dependent planners when animations are disabled', () => {
    const previousState = makeGameState();
    const nextState = makeGameState({ phase: 'GameOver' });
    const calls: string[] = [];

    const plan = prepareActionDispatch({
      previousState,
      action: ACTION,
      actingPlayerId: 'PlayerA',
      animationsEnabled: false,
      makeResourceFlightId: makeIds('resource'),
      makeCardFlightId: makeIds('card'),
      dependencies: makeDependencies(nextState, calls),
    });

    expect(calls).toEqual(['step']);
    expect(plan).toMatchObject({
      previousState,
      nextState,
      action: ACTION,
      resourceFlights: [],
      cardFlights: [],
      turnCyclePlan: null,
      enteredTerminal: true,
    });
  });

  it('appends terminal cleanup flights after action flights', () => {
    const previousState = makeGameState();
    const nextState = makeGameState({ phase: 'GameOver' });
    const actionResourceFlight = makeResourceFlight('action-resource');
    const terminalResourceFlight = makeResourceFlight('terminal-resource');
    const actionCardFlight = makeCardFlight('action-card', 200, 10);
    const terminalCardFlight = makeCardFlight('terminal-card', 900);
    const dependencies = makeDependencies(nextState);
    dependencies.collectDeedResourceFlights = () => [actionResourceFlight];
    dependencies.collectCardPlayFlights = () => [actionCardFlight];
    dependencies.collectTerminalCleanupFlights = () => ({
      resourceFlights: [terminalResourceFlight],
      cardFlights: [terminalCardFlight],
    });

    const plan = prepareActionDispatch({
      previousState,
      action: ACTION,
      actingPlayerId: 'PlayerA',
      animationsEnabled: true,
      makeResourceFlightId: makeIds('resource'),
      makeCardFlightId: makeIds('card'),
      dependencies,
    });

    expect(plan.resourceFlights).toEqual([
      actionResourceFlight,
      terminalResourceFlight,
    ]);
    expect(plan.cardFlights).toEqual([actionCardFlight, terminalCardFlight]);
    expect(plan.enteredTerminal).toBe(true);
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
    collectDeedResourceFlights: () => {
      calls.push('deed');
      return [];
    },
    collectIncomeChoiceResourceFlights: () => {
      calls.push('income-choice');
      return [];
    },
    collectCardPlayFlights: () => {
      calls.push('card');
      return [];
    },
    collectTerminalCleanupFlights: () => {
      calls.push('terminal');
      return null;
    },
    collectTurnCycleAnimationPlan: () => {
      calls.push('turn-cycle');
      return null;
    },
  };
}

function makeIds(prefix: string): () => string {
  let next = 0;
  return () => {
    next += 1;
    return `${prefix}-${next}`;
  };
}

function makeResourceFlight(id: string): ResourceFlight {
  return {
    id,
    suit: 'Moons',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    delayMs: 0,
  };
}

function makeCardFlight(
  id: string,
  durationMs: number,
  delayMs = 0
): CardFlight {
  return {
    id,
    variant: 'play',
    visual: 'face',
    isDeed: false,
    perspective: 'human',
    startX: 0,
    startY: 0,
    endX: 10,
    endY: 10,
    startWidth: 10,
    startHeight: 10,
    endWidth: 10,
    endHeight: 10,
    delayMs,
    durationMs,
  };
}
