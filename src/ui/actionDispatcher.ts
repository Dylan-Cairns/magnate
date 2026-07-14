import { stepToDecision } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type { GameAction, GameState } from '../engine/types';

export type ActionDispatchDependencies = {
  stepToDecision: typeof stepToDecision;
};

export type ActionDispatchPlan = {
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  enteredTerminal: boolean;
};

type PrepareActionDispatchOptions = {
  previousState: GameState;
  action: GameAction;
  dependencies?: ActionDispatchDependencies;
};

const browserActionDispatchDependencies: ActionDispatchDependencies = {
  stepToDecision,
};

export function prepareActionDispatch({
  previousState,
  action,
  dependencies = browserActionDispatchDependencies,
}: PrepareActionDispatchOptions): ActionDispatchPlan {
  const nextState = dependencies.stepToDecision(previousState, action);
  const enteredTerminal = isTerminal(nextState);

  return {
    previousState,
    nextState,
    action,
    enteredTerminal,
  };
}
