import { stepToDecision } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import {
  collectCardPlayFlights,
  collectDeedResourceFlights,
  collectTerminalCleanupFlights,
} from './animations/flightPlans';
import type { CardFlight, ResourceFlight } from './animations/types';

export type ActionDispatchDependencies = {
  stepToDecision: typeof stepToDecision;
  collectDeedResourceFlights: typeof collectDeedResourceFlights;
  collectCardPlayFlights: typeof collectCardPlayFlights;
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
  collectDeedResourceFlights,
  collectCardPlayFlights,
  collectTerminalCleanupFlights,
};

export function prepareActionDispatch({
  previousState,
  action,
  actingPlayerId,
  animationsEnabled,
  makeResourceFlightId,
  makeCardFlightId,
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

  const actionResourceFlights = [
    ...dependencies.collectDeedResourceFlights(
      previousState,
      action,
      actingPlayerId,
      makeResourceFlightId
    ),
  ];
  const actionCardFlights = dependencies.collectCardPlayFlights(
    previousState,
    nextState,
    action,
    actingPlayerId,
    makeCardFlightId
  );
  const terminalCleanupPlan = dependencies.collectTerminalCleanupFlights();

  return {
    previousState,
    nextState,
    action,
    resourceFlights: terminalCleanupPlan
      ? [...actionResourceFlights, ...terminalCleanupPlan.resourceFlights]
      : actionResourceFlights,
    cardFlights: terminalCleanupPlan
      ? [...actionCardFlights, ...terminalCleanupPlan.cardFlights]
      : actionCardFlights,
    enteredTerminal,
  };
}
