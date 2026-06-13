import { stepToDecision } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import { collectTerminalCleanupFlights } from './animations/flightPlans';
import type { CardFlight, ResourceFlight } from './animations/types';

export type ActionDispatchDependencies = {
  stepToDecision: typeof stepToDecision;
  collectTerminalCleanupFlights: typeof collectTerminalCleanupFlights;
};

export type ActionDispatchPlan = {
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  resourceFlights: readonly ResourceFlight[];
  cardFlights: readonly CardFlight[];
  enteredTerminal: boolean;
};

type PrepareActionDispatchOptions = {
  previousState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  animationsEnabled: boolean;
  makeResourceFlightId: () => string;
  makeCardFlightId: () => string;
  dependencies?: ActionDispatchDependencies;
};

const browserActionDispatchDependencies: ActionDispatchDependencies = {
  stepToDecision,
  collectTerminalCleanupFlights,
};

export function prepareActionDispatch({
  previousState,
  action,
  animationsEnabled,
  dependencies = browserActionDispatchDependencies,
}: PrepareActionDispatchOptions): ActionDispatchPlan {
  const nextState = dependencies.stepToDecision(previousState, action);
  const enteredTerminal = isTerminal(nextState);
  if (!animationsEnabled) {
    return {
      previousState,
      nextState,
      action,
      resourceFlights: [],
      cardFlights: [],
      enteredTerminal,
    };
  }

  const terminalCleanupPlan = dependencies.collectTerminalCleanupFlights();

  return {
    previousState,
    nextState,
    action,
    resourceFlights: terminalCleanupPlan?.resourceFlights ?? [],
    cardFlights: terminalCleanupPlan?.cardFlights ?? [],
    enteredTerminal,
  };
}
