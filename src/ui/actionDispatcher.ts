import { stepToDecision } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import {
  collectCardPlayFlights,
  collectDeedResourceFlights,
  collectIncomeChoiceResourceFlights,
  collectTerminalCleanupFlights,
} from './animations/flightPlans';
import { cardFlightSettleMs } from './animations/timing';
import {
  collectTurnCycleAnimationPlan,
  type TurnCycleAnimationPlan,
} from './animations/turnCycleVisualPlan';
import type { CardFlight, ResourceFlight } from './animations/types';

export type ActionDispatchDependencies = {
  stepToDecision: typeof stepToDecision;
  collectDeedResourceFlights: typeof collectDeedResourceFlights;
  collectIncomeChoiceResourceFlights: typeof collectIncomeChoiceResourceFlights;
  collectCardPlayFlights: typeof collectCardPlayFlights;
  collectTerminalCleanupFlights: typeof collectTerminalCleanupFlights;
  collectTurnCycleAnimationPlan: typeof collectTurnCycleAnimationPlan;
};

export type ActionDispatchPlan = {
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  resourceFlights: readonly ResourceFlight[];
  cardFlights: readonly CardFlight[];
  turnCyclePlan: TurnCycleAnimationPlan | null;
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
  collectIncomeChoiceResourceFlights,
  collectCardPlayFlights,
  collectTerminalCleanupFlights,
  collectTurnCycleAnimationPlan,
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
      turnCyclePlan: null,
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
    ...dependencies.collectIncomeChoiceResourceFlights(
      action,
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
  const turnCyclePlan = dependencies.collectTurnCycleAnimationPlan(
    previousState,
    nextState,
    action,
    cardFlightSettleMs(actionCardFlights)
  );

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
    turnCyclePlan,
    enteredTerminal,
  };
}
