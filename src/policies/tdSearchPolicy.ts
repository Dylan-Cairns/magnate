import {
  actionStableKey,
  toKeyedActions,
  type KeyedAction,
} from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import { sampleHiddenWorldStates } from './determinization';
import {
  progressiveTargetActionCount,
  selectBestRootActionKey,
  selectRootUcbAction,
} from './searchRoot';
import type { ActionPolicy } from './types';
import { encodeActionCandidates, encodeObservation } from './trainingEncoding';
import {
  DEFAULT_TD_SEARCH_MODEL_INDEX_PATH,
  preloadTdSearchBrowserModel,
} from './modelRuntimeCache';
import { heuristicPriorsByKey, rankHeuristicActions } from './heuristicScorer';
import { type LoadedTdSearchModel } from './tdSearchModelPack';
import { cachedAsyncLoader, forcedAction } from './policyUtils';

export interface TdSearchPolicyConfig {
  worlds: number;
  rollouts: number;
  depth: number;
  maxRootActions: number;
  rolloutEpsilon: number;
  opponentTemperature: number;
  sampleOpponentActions: boolean;
}

export type TdSearchPolicyOptions = Partial<TdSearchPolicyConfig> & {
  modelIndexPath?: string;
  loadModel?: () => Promise<LoadedTdSearchModel>;
};

const DEFAULT_TD_SEARCH_POLICY_CONFIG: TdSearchPolicyConfig = {
  worlds: 6,
  rollouts: 1,
  depth: 14,
  maxRootActions: 6,
  rolloutEpsilon: 0.04,
  opponentTemperature: 1.0,
  sampleOpponentActions: false,
};

export function createTdSearchPolicy(
  options: TdSearchPolicyOptions = {}
): ActionPolicy {
  const config = resolveTdSearchConfig(options);
  const configuredLoader =
    options.loadModel ??
    (() =>
      preloadTdSearchBrowserModel(
        options.modelIndexPath ?? DEFAULT_TD_SEARCH_MODEL_INDEX_PATH
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
      const heuristicContext = { state, view };
      const rankedRootActions = rankHeuristicActions(
        candidateActions,
        heuristicContext
      );
      const worldStates = sampleHiddenWorldStates({
        state,
        view,
        rootPlayer,
        worldCount: config.worlds,
        random,
        errorPrefix: 'TD search',
      });
      if (worldStates.length === 0) {
        return rankedRootActions[0].action;
      }

      const actionByKey = new Map(
        rankedRootActions.map((candidate) => [
          candidate.actionKey,
          candidate.action,
        ])
      );
      const rootPriorByKey = heuristicPriorsByKey(
        candidateActions,
        heuristicContext
      );
      const expandedCount = Math.min(
        rankedRootActions.length,
        config.maxRootActions
      );
      const expandedKeys = rankedRootActions
        .slice(0, expandedCount)
        .map((candidate) => candidate.actionKey);
      const pendingUnvisited = [...expandedKeys];

      const rootVisits = new Map<string, number>();
      const rootValueSum = new Map<string, number>();
      for (const candidate of rankedRootActions) {
        rootVisits.set(candidate.actionKey, 0);
        rootValueSum.set(candidate.actionKey, 0);
      }

      const visitCount = worldStates.length * config.rollouts;
      const rootBudget =
        Math.max(1, visitCount) * Math.max(1, config.maxRootActions);
      for (let visitIndex = 0; visitIndex < rootBudget; visitIndex += 1) {
        const targetCount = progressiveTargetActionCount(
          rankedRootActions.length,
          config.maxRootActions,
          visitIndex
        );
        while (expandedKeys.length < targetCount) {
          const nextKey = rankedRootActions[expandedKeys.length].actionKey;
          expandedKeys.push(nextKey);
          pendingUnvisited.push(nextKey);
        }

        const actionKey =
          pendingUnvisited.length > 0
            ? pendingUnvisited.shift()!
            : selectRootUcbAction(
                expandedKeys,
                rootVisits,
                rootValueSum,
                rootPriorByKey,
                visitIndex,
                1
              );

        const worldIndex = visitIndex % worldStates.length;
        const score = runRollout({
          initialState: worldStates[worldIndex],
          rootPlayer,
          rootActionKey: actionKey,
          config,
          random,
          model,
        });
        rootVisits.set(actionKey, (rootVisits.get(actionKey) ?? 0) + 1);
        rootValueSum.set(actionKey, (rootValueSum.get(actionKey) ?? 0) + score);
      }

      const bestActionKey = selectBestRootActionKey({
        expandedKeys,
        visitsByKey: rootVisits,
        valueSumByKey: rootValueSum,
        priorsByKey: rootPriorByKey,
      });

      const selected = actionByKey.get(bestActionKey);
      if (!selected) {
        throw new Error(
          `TD search policy selected root action key that is no longer legal: ${bestActionKey}.`
        );
      }
      return selected;
    },
  };
}

function resolveTdSearchConfig(
  options: TdSearchPolicyOptions
): TdSearchPolicyConfig {
  const worlds = integerWithFloor(
    options.worlds ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.worlds,
    1
  );
  const rollouts = integerWithFloor(
    options.rollouts ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.rollouts,
    1
  );
  const depth = integerWithFloor(
    options.depth ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.depth,
    1
  );
  const maxRootActions = integerWithFloor(
    options.maxRootActions ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.maxRootActions,
    1
  );
  const rolloutEpsilon =
    options.rolloutEpsilon ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.rolloutEpsilon;
  if (
    !Number.isFinite(rolloutEpsilon) ||
    rolloutEpsilon < 0 ||
    rolloutEpsilon > 1
  ) {
    throw new Error(
      `TD search rolloutEpsilon must be in [0, 1]; received ${String(rolloutEpsilon)}.`
    );
  }

  const opponentTemperature =
    options.opponentTemperature ??
    DEFAULT_TD_SEARCH_POLICY_CONFIG.opponentTemperature;
  if (!Number.isFinite(opponentTemperature) || opponentTemperature <= 0) {
    throw new Error(
      `TD search opponentTemperature must be > 0; received ${String(opponentTemperature)}.`
    );
  }

  return {
    worlds,
    rollouts,
    depth,
    maxRootActions,
    rolloutEpsilon,
    opponentTemperature,
    sampleOpponentActions:
      options.sampleOpponentActions ??
      DEFAULT_TD_SEARCH_POLICY_CONFIG.sampleOpponentActions,
  };
}

function runRollout({
  initialState,
  rootPlayer,
  rootActionKey,
  config,
  random,
  model,
}: {
  initialState: GameState;
  rootPlayer: PlayerId;
  rootActionKey: string;
  config: TdSearchPolicyConfig;
  random: () => number;
  model: LoadedTdSearchModel;
}): number {
  let state = stepByActionKey(initialState, rootActionKey);
  let depth = 0;

  while (!isTerminal(state) && depth < config.depth) {
    const activePlayer = decisionPlayerIdForState(state);
    if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
      throw new Error('TD search rollout could not resolve active player.');
    }
    const actions = legalActionsForDecisionPlayer(state, activePlayer);
    if (actions.length === 0) {
      break;
    }

    const nextActionKey =
      activePlayer === rootPlayer
        ? chooseRootRolloutActionKey(
            state,
            actions,
            activePlayer,
            random,
            config.rolloutEpsilon
          )
        : chooseOpponentRolloutActionKey({
            actions,
            state,
            random,
            model,
            config,
            activePlayer,
          });
    state = stepByActionKey(state, nextActionKey);
    depth += 1;
  }

  if (isTerminal(state)) {
    return terminalValue(state, rootPlayer);
  }
  return tdLeafValue(state, rootPlayer, model);
}

function chooseRootRolloutActionKey(
  state: GameState,
  actions: readonly GameAction[],
  activePlayer: PlayerId,
  random: () => number,
  rolloutEpsilon: number
): string {
  const keyed = toKeyedActions(actions);
  if (random() < rolloutEpsilon) {
    return keyed[Math.floor(random() * keyed.length)].actionKey;
  }
  return rankHeuristicActions(actions, {
    state,
    view: toDecisionPlayerView(state, activePlayer),
  })[0].actionKey;
}

function chooseOpponentRolloutActionKey({
  actions,
  state,
  random,
  model,
  config,
  activePlayer,
}: {
  actions: readonly GameAction[];
  state: GameState;
  random: () => number;
  model: LoadedTdSearchModel;
  config: TdSearchPolicyConfig;
  activePlayer: PlayerId;
}): string {
  if (random() < config.rolloutEpsilon) {
    const keyed = toKeyedActions(actions);
    return keyed[Math.floor(random() * keyed.length)].actionKey;
  }

  const keyed = toKeyedActions(actions);
  const view = toDecisionPlayerView(state, activePlayer);
  const observation = encodeObservation(view);
  const actionFeatures = encodeActionCandidates(actions);
  const logits = model.opponentScorer.logits(observation, actionFeatures);
  if (logits.length !== keyed.length) {
    throw new Error(
      `TD search opponent logits length mismatch. logits=${String(logits.length)} actions=${String(keyed.length)}.`
    );
  }

  const scaledLogits = Array.from(
    logits,
    (value) => value / config.opponentTemperature
  );
  if (config.sampleOpponentActions) {
    const sampledIndex = sampleLogitsIndex(scaledLogits, random);
    return keyed[sampledIndex].actionKey;
  }
  return keyed[argmaxLogitsIndex(scaledLogits, keyed)].actionKey;
}

function tdLeafValue(
  state: GameState,
  rootPlayer: PlayerId,
  model: LoadedTdSearchModel
): number {
  const activePlayer = decisionPlayerIdForState(state);
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD search leaf value could not resolve active player.');
  }
  const view = toDecisionPlayerView(state, activePlayer);
  const observation = encodeObservation(view);
  const activeValue = model.valueScorer.predict(observation);
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function argmaxLogitsIndex(
  logits: readonly number[],
  keyed: readonly KeyedAction[]
): number {
  if (logits.length === 0) {
    throw new Error('argmaxLogitsIndex requires at least one logit.');
  }
  let bestIndex = 0;
  let bestValue = logits[0];
  let bestKey = keyed[0].actionKey;
  for (let index = 1; index < logits.length; index += 1) {
    const value = logits[index];
    const key = keyed[index].actionKey;
    if (
      value > bestValue ||
      (approximatelyEqual(value, bestValue) && key.localeCompare(bestKey) < 0)
    ) {
      bestIndex = index;
      bestValue = value;
      bestKey = key;
    }
  }
  return bestIndex;
}

function sampleLogitsIndex(
  logits: readonly number[],
  random: () => number
): number {
  if (logits.length === 0) {
    throw new Error('sampleLogitsIndex requires at least one logit.');
  }
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    maxLogit = Math.max(maxLogit, value);
  }
  const expValues = logits.map((value) => Math.exp(value - maxLogit));
  const total = expValues.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return 0;
  }
  let threshold = random() * total;
  for (let index = 0; index < expValues.length; index += 1) {
    threshold -= expValues[index];
    if (threshold <= 0) {
      return index;
    }
  }
  return expValues.length - 1;
}

function stepByActionKey(state: GameState, actionKey: string): GameState {
  const activePlayer = decisionPlayerIdForState(state);
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD search rollout could not resolve active player.');
  }
  const actions = legalActionsForDecisionPlayer(state, activePlayer);
  const action = actions.find(
    (candidate) => actionStableKey(candidate) === actionKey
  );
  if (!action) {
    throw new Error(
      `TD search rollout could not find legal action for key ${actionKey}.`
    );
  }
  return stepToDecision(state, action);
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
      `TD search expected a finite number; received ${String(value)}.`
    );
  }
  const rounded = Math.trunc(value);
  if (rounded < floor) {
    throw new Error(
      `TD search value must be >= ${String(floor)}; received ${String(value)}.`
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
