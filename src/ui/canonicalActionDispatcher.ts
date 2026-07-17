import { turnOwnerIdForState } from '../engine/decisionActor';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import {
  prepareActionDispatch,
  type ActionDispatchDependencies,
  type ActionDispatchPlan,
} from './actionDispatcher';

export type CanonicalActionDispatchPlan = ActionDispatchPlan & {
  actingPlayerId: PlayerId;
  actionOrdinal: number;
  transactionId: string;
};

export type PrepareCanonicalActionDispatchOptions = {
  currentState: GameState;
  sourceState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  actionOrdinal: number;
  dependencies?: ActionDispatchDependencies;
};

export function prepareCanonicalActionDispatch({
  currentState,
  sourceState,
  action,
  actingPlayerId,
  actionOrdinal,
  dependencies,
}: PrepareCanonicalActionDispatchOptions): CanonicalActionDispatchPlan {
  if (sourceState !== currentState) {
    throw new Error('Cannot dispatch an action from a stale canonical state.');
  }
  if (!Number.isSafeInteger(actionOrdinal) || actionOrdinal < 0) {
    throw new Error(
      `Invalid canonical action ordinal ${String(actionOrdinal)}.`
    );
  }

  const expectedActorId =
    action.type === 'choose-income-suit'
      ? action.playerId
      : turnOwnerIdForState(currentState);
  if (expectedActorId !== actingPlayerId) {
    throw new Error(
      `Action actor mismatch: expected ${String(expectedActorId)}, received ${actingPlayerId}.`
    );
  }

  const plan = prepareActionDispatch({
    previousState: currentState,
    action,
    dependencies,
  });
  return {
    ...plan,
    actingPlayerId,
    actionOrdinal,
    transactionId: canonicalTransactionId(currentState, action, actionOrdinal),
  };
}

function canonicalTransactionId(
  state: GameState,
  action: GameAction,
  actionOrdinal: number
): string {
  return `${state.seed}:action:${String(actionOrdinal)}:${String(state.turn)}:${state.phase}:${action.type}`;
}
