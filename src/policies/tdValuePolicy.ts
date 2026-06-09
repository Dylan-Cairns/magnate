import { actionStableKey } from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import { sampleHiddenWorldStates } from './determinization';
import type { ActionPolicy } from './types';
import { encodeObservation } from './trainingEncoding';
import {
  DEFAULT_TD_VALUE_MODEL_INDEX_PATH,
  preloadTdValueBrowserModel,
} from './modelRuntimeCache';
import { cachedAsyncLoader, forcedAction } from './policyUtils';
import { type TdValueScorer } from './tdValueModelPack';

const DEFAULT_TD_VALUE_WORLDS = 8;

export interface TdValuePolicyOptions {
  worlds?: number;
  modelIndexPath?: string;
  loadModel?: () => Promise<TdValueScorer>;
}

export function createTdValuePolicy(
  options: TdValuePolicyOptions = {}
): ActionPolicy {
  const worlds = integerWithFloor(options.worlds ?? DEFAULT_TD_VALUE_WORLDS, 1);
  const configuredLoader =
    options.loadModel ??
    (() =>
      preloadTdValueBrowserModel(
        options.modelIndexPath ?? DEFAULT_TD_VALUE_MODEL_INDEX_PATH
      ));

  const getModel = cachedAsyncLoader(configuredLoader);

  return {
    async selectAction({
      view,
      state,
      legalActions: candidateActions,
      random,
    }) {
      const forced = forcedAction(candidateActions);
      if (forced !== null) {
        return forced;
      }

      const model = await getModel();
      const rootPlayer = view.activePlayerId;
      const worldStates = sampleHiddenWorldStates({
        state,
        view,
        rootPlayer,
        worldCount: worlds,
        random,
        errorPrefix: 'TD value',
      });
      const rankedActions = [...candidateActions].sort((left, right) =>
        actionStableKey(left).localeCompare(actionStableKey(right))
      );
      if (worldStates.length === 0) {
        return rankedActions[0];
      }

      let bestAction = rankedActions[0];
      let bestActionKey = actionStableKey(bestAction);
      let bestScore = Number.NEGATIVE_INFINITY;
      for (const action of rankedActions) {
        const actionKey = actionStableKey(action);
        let total = 0;
        for (const world of worldStates) {
          total += scoreActionInWorld({
            world,
            action,
            rootPlayer,
            scorer: model,
          });
        }
        const score = total / worldStates.length;
        if (
          score > bestScore ||
          (approximatelyEqual(score, bestScore) &&
            actionKey.localeCompare(bestActionKey) < 0)
        ) {
          bestAction = action;
          bestActionKey = actionKey;
          bestScore = score;
        }
      }
      return bestAction;
    },
  };
}

function scoreActionInWorld({
  world,
  action,
  rootPlayer,
  scorer,
}: {
  world: GameState;
  action: GameAction;
  rootPlayer: PlayerId;
  scorer: TdValueScorer;
}): number {
  const next = stepToDecision(world, action);
  if (isTerminal(next)) {
    return terminalValue(next, rootPlayer);
  }

  const activePlayer = decisionPlayerIdForState(next);
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error(
      'TD value policy could not resolve active player from next state.'
    );
  }
  const nextView = toDecisionPlayerView(next, activePlayer);
  const observation = encodeObservation(nextView);
  const activeValue = scorer.predict(observation);
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function terminalValue(state: GameState, rootPlayer: PlayerId): number {
  const winner = state.finalScore?.winner;
  if (!winner || winner === 'Draw') {
    return 0;
  }
  return winner === rootPlayer ? 1 : -1;
}

function integerWithFloor(value: number, floor: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(
      `TD value policy expected a finite number; received ${String(value)}.`
    );
  }
  const rounded = Math.trunc(value);
  if (rounded < floor) {
    throw new Error(
      `TD value policy value must be >= ${String(floor)}; received ${String(value)}.`
    );
  }
  return rounded;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
