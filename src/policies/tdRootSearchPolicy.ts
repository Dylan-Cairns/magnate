import { toKeyedActions } from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import {
  DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
  preloadTdRootBrowserModel,
} from './modelRuntimeCache';
import {
  selectRolloutSearchActionSync,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchRootGuide,
  type RolloutSearchRootGuideInput,
} from './rolloutSearchCore';
import {
  resolveSearchConfig,
  type SearchPolicyConfig,
  type SearchPolicyOptions,
} from './searchConfig';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import { encodeActionCandidates, encodeObservation } from './trainingEncoding';
import type { ActionPolicy } from './types';

export type TdRootSearchPolicyConfig = SearchPolicyConfig;

export type TdRootSearchPolicyOptions = SearchPolicyOptions & {
  modelIndexPath?: string;
  loadModel?: () => Promise<LoadedTdGuidanceModel>;
  rootPriorTemperature?: number;
};

const DEFAULT_ROOT_PRIOR_TEMPERATURE = 1.0;

export function createTdRootSearchPolicy(
  options: TdRootSearchPolicyOptions = {}
): ActionPolicy {
  const config = resolveSearchConfig(options);
  const rootPriorTemperature = resolveRootPriorTemperature(
    options.rootPriorTemperature ?? DEFAULT_ROOT_PRIOR_TEMPERATURE
  );
  const configuredLoader =
    options.loadModel ??
    (() =>
      preloadTdRootBrowserModel(
        options.modelIndexPath ?? DEFAULT_TD_ROOT_MODEL_INDEX_PATH
      ));

  let modelPromise: Promise<LoadedTdGuidanceModel> | null = null;
  function getModel(): Promise<LoadedTdGuidanceModel> {
    if (modelPromise === null) {
      modelPromise = configuredLoader();
    }
    return modelPromise;
  }

  return {
    async selectAction({
      state,
      view,
      legalActions: candidateActions,
      random,
      randomSeed,
      onSearchDiagnostics,
      onProgress,
    }) {
      if (candidateActions.length === 0) {
        return undefined;
      }
      if (candidateActions.length === 1) {
        return candidateActions[0];
      }

      const model = await getModel();
      const rolloutGuidance = createTdRootSearchRolloutGuidance({ model });
      return selectRolloutSearchActionSync({
        state,
        view,
        candidateActions,
        config,
        random,
        ...(randomSeed ? { randomSeed } : {}),
        createRootGuide(input) {
          return createTdRootSearchRootGuide({
            ...input,
            model,
            rootPriorTemperature,
          });
        },
        rolloutGuidance,
        guidanceKind: 'td-root',
        onSearchDiagnostics,
        onProgress,
      });
    },
  };
}

export function createTdRootSearchRootGuide({
  view,
  candidateActions,
  model,
  rootPriorTemperature = DEFAULT_ROOT_PRIOR_TEMPERATURE,
}: RolloutSearchRootGuideInput & {
  model: LoadedTdGuidanceModel;
  rootPriorTemperature?: number;
}): RolloutSearchRootGuide {
  const temperature = resolveRootPriorTemperature(rootPriorTemperature);
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length !== candidateActions.length) {
    throw new Error(
      'TD root search action keying did not preserve action count.'
    );
  }

  const logits = tdRootLogits({
    view,
    actions: keyed.map((candidate) => candidate.action),
    actionKeys: keyed.map((candidate) => candidate.actionKey),
    model,
    label: 'root',
  });
  const scored = keyed.map((candidate, index) => {
    return {
      action: candidate.action,
      actionKey: candidate.actionKey,
      score: logits[index],
    };
  });

  scored.sort((left, right) => {
    const scoreDelta = right.score - left.score;
    if (!approximatelyEqual(scoreDelta, 0)) {
      return scoreDelta > 0 ? 1 : -1;
    }
    return left.actionKey.localeCompare(right.actionKey);
  });

  return {
    rankedRootActions: scored.map(({ action, actionKey }) => ({
      action,
      actionKey,
    })),
    rootPriorByKey: tdRootPriorsByKey({
      actionKeys: keyed.map((candidate) => candidate.actionKey),
      logits,
      temperature,
    }),
  };
}

export function createTdRootSearchRolloutGuidance({
  model,
}: {
  model: LoadedTdGuidanceModel;
}): RolloutSearchRuntimeGuidance {
  return {
    evaluateLeaf({ state, rootPlayer }) {
      return tdLeafValue({ state, rootPlayer, model });
    },
    chooseRolloutAction({ state, actions, decisionPlayer }) {
      return chooseTdRolloutAction({
        state,
        actions,
        decisionPlayer,
        model,
      });
    },
  };
}

function tdLeafValue({
  state,
  rootPlayer,
  model,
}: {
  state: GameState;
  rootPlayer: PlayerId;
  model: LoadedTdGuidanceModel;
}): number {
  const activePlayer = decisionPlayerIdForState(state);
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD root search leaf could not resolve active player.');
  }
  const view = toDecisionPlayerView(state, activePlayer);
  const activeValue = model.valueScorer.predict(encodeObservation(view));
  if (!Number.isFinite(activeValue)) {
    throw new Error(
      `TD root search value scorer returned ${String(activeValue)}.`
    );
  }
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function chooseTdRolloutAction({
  state,
  actions,
  decisionPlayer,
  model,
}: {
  state: GameState;
  actions: readonly GameAction[];
  decisionPlayer: PlayerId;
  model: LoadedTdGuidanceModel;
}): GameAction {
  const keyed = toKeyedActions(actions);
  const view = toDecisionPlayerView(state, decisionPlayer);
  const logits = tdRootLogits({
    view,
    actions: keyed.map((candidate) => candidate.action),
    actionKeys: keyed.map((candidate) => candidate.actionKey),
    model,
    label: 'rollout',
  });
  return keyed[argmaxLogitsIndex(logits, keyed)].action;
}

function tdRootLogits({
  view,
  actions,
  actionKeys,
  model,
  label,
}: {
  view: RolloutSearchRootGuideInput['view'];
  actions: readonly GameAction[];
  actionKeys: readonly string[];
  model: LoadedTdGuidanceModel;
  label: string;
}): Float32Array {
  const observation = encodeObservation(view);
  const logits = model.opponentScorer.logitsForActions
    ? model.opponentScorer.logitsForActions(observation, actions)
    : model.opponentScorer.logits(observation, encodeActionCandidates(actions));
  if (logits.length !== actionKeys.length) {
    throw new Error(
      `TD root search ${label} logits length mismatch. logits=${String(logits.length)} actions=${String(actionKeys.length)}.`
    );
  }
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new Error(
        `TD root search ${label} logit is not finite: ${String(logit)}.`
      );
    }
  }
  return logits;
}

function tdRootPriorsByKey({
  actionKeys,
  logits,
  temperature,
}: {
  actionKeys: readonly string[];
  logits: ArrayLike<number>;
  temperature: number;
}): Map<string, number> {
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    const logit = logits[index];
    maxLogit = Math.max(maxLogit, logit / temperature);
  }

  const expValues = Array.from({ length: logits.length }, (_unused, index) =>
    Math.exp(logits[index] / temperature - maxLogit)
  );
  const total = expValues.reduce((sum, value) => sum + value, 0);
  const priors = new Map<string, number>();
  if (!Number.isFinite(total) || total <= 0) {
    const uniform = 1 / Math.max(1, actionKeys.length);
    for (const actionKey of actionKeys) {
      priors.set(actionKey, uniform);
    }
    return priors;
  }

  actionKeys.forEach((actionKey, index) => {
    priors.set(actionKey, expValues[index] / total);
  });
  return priors;
}

function argmaxLogitsIndex(
  logits: ArrayLike<number>,
  keyed: ReturnType<typeof toKeyedActions>
): number {
  if (logits.length === 0) {
    throw new Error('TD root search requires at least one rollout logit.');
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

function resolveRootPriorTemperature(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(
      `TD root search rootPriorTemperature must be > 0; received ${String(value)}.`
    );
  }
  return value;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
