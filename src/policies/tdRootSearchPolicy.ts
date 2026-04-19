import { toKeyedActions } from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import {
  DEFAULT_TD_SEARCH_MODEL_INDEX_PATH,
  preloadTdSearchBrowserModel,
} from './modelRuntimeCache';
import {
  selectRolloutSearchActionSync,
  type RolloutSearchRootGuide,
  type RolloutSearchRootGuideInput,
} from './rolloutSearchCore';
import {
  resolveSearchConfig,
  type SearchPolicyConfig,
  type SearchPolicyOptions,
} from './searchConfig';
import { evaluateSearchTerminalState } from './searchStateEvaluator';
import type { LoadedTdSearchModel } from './tdSearchModelPack';
import { encodeActionCandidates, encodeObservation } from './trainingEncoding';
import type { ActionPolicy } from './types';

export type TdRootSearchPolicyConfig = SearchPolicyConfig;

export type TdRootSearchPolicyOptions = SearchPolicyOptions & {
  modelIndexPath?: string;
  loadModel?: () => Promise<LoadedTdSearchModel>;
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
      preloadTdSearchBrowserModel(
        options.modelIndexPath ?? DEFAULT_TD_SEARCH_MODEL_INDEX_PATH
      ));

  let modelPromise: Promise<LoadedTdSearchModel> | null = null;
  function getModel(): Promise<LoadedTdSearchModel> {
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
        onSearchDiagnostics,
        onProgress,
      });
    },
  };
}

export function createTdRootSearchRootGuide({
  view,
  candidateActions,
  worldStates,
  rootPlayer,
  model,
  rootPriorTemperature = DEFAULT_ROOT_PRIOR_TEMPERATURE,
}: RolloutSearchRootGuideInput & {
  model: LoadedTdSearchModel;
  rootPriorTemperature?: number;
}): RolloutSearchRootGuide {
  const temperature = resolveRootPriorTemperature(rootPriorTemperature);
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length !== candidateActions.length) {
    throw new Error(
      'TD root search action keying did not preserve action count.'
    );
  }
  if (worldStates.length === 0) {
    throw new Error('TD root search requires at least one sampled world.');
  }

  const scored = keyed.map((candidate) => {
    let total = 0;
    for (const world of worldStates) {
      total += scoreRootActionInWorld({
        world,
        action: candidate.action,
        rootPlayer,
        model,
      });
    }
    return {
      action: candidate.action,
      actionKey: candidate.actionKey,
      score: total / worldStates.length,
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
      view,
      actions: keyed.map((candidate) => candidate.action),
      actionKeys: keyed.map((candidate) => candidate.actionKey),
      model,
      temperature,
    }),
  };
}

function scoreRootActionInWorld({
  world,
  action,
  rootPlayer,
  model,
}: {
  world: GameState;
  action: GameAction;
  rootPlayer: PlayerId;
  model: LoadedTdSearchModel;
}): number {
  const next = stepToDecision(world, action);
  if (isTerminal(next)) {
    return evaluateSearchTerminalState(next, rootPlayer);
  }

  const activePlayer = decisionPlayerIdForState(next);
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD root search could not resolve active player.');
  }
  const nextView = toDecisionPlayerView(next, activePlayer);
  const activeValue = model.valueScorer.predict(encodeObservation(nextView));
  if (!Number.isFinite(activeValue)) {
    throw new Error(
      `TD root search value scorer returned ${String(activeValue)}.`
    );
  }
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function tdRootPriorsByKey({
  view,
  actions,
  actionKeys,
  model,
  temperature,
}: {
  view: RolloutSearchRootGuideInput['view'];
  actions: readonly GameAction[];
  actionKeys: readonly string[];
  model: LoadedTdSearchModel;
  temperature: number;
}): Map<string, number> {
  const observation = encodeObservation(view);
  const actionFeatures = encodeActionCandidates(actions);
  const logits = model.opponentScorer.logits(observation, actionFeatures);
  if (logits.length !== actionKeys.length) {
    throw new Error(
      `TD root search logits length mismatch. logits=${String(logits.length)} actions=${String(actionKeys.length)}.`
    );
  }

  let maxLogit = Number.NEGATIVE_INFINITY;
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new Error(
        `TD root search prior logit is not finite: ${String(logit)}.`
      );
    }
    maxLogit = Math.max(maxLogit, logit / temperature);
  }

  const expValues = Array.from(logits, (logit) =>
    Math.exp(logit / temperature - maxLogit)
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
