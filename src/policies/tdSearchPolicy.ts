import { legalActions } from '../engine/actionBuilders';
import {
  actionStableKey,
  toKeyedActions,
  type KeyedAction,
} from '../engine/actionSurface';
import { PROPERTY_CARDS } from '../engine/cards';
import { shuffleInPlace } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { GameAction, GameState, PlayerId, PlayerView } from '../engine/types';
import type { ActionPolicy } from './types';
import {
  encodeActionCandidates,
  encodeObservation,
} from './trainingEncoding';
import {
  loadTdSearchModelFromIndexUrl,
  type LoadedTdSearchModel,
} from './tdSearchModelPack';
import { resolvePublicAssetUrl } from './tdValueModelPack';

const PROPERTY_CARD_IDS = PROPERTY_CARDS.map((card) => card.id);

export const DEFAULT_TD_SEARCH_MODEL_INDEX_PATH = 'model-packs/index.json';

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
    (async (): Promise<LoadedTdSearchModel> => {
      const indexPath =
        options.modelIndexPath ?? DEFAULT_TD_SEARCH_MODEL_INDEX_PATH;
      const indexUrl = resolvePublicAssetUrl(indexPath);
      return loadTdSearchModelFromIndexUrl(indexUrl);
    });

  let modelPromise: Promise<LoadedTdSearchModel> | null = null;
  function getModel(): Promise<LoadedTdSearchModel> {
    if (modelPromise === null) {
      modelPromise = configuredLoader();
    }
    return modelPromise;
  }

  return {
    async selectAction({ view, state, legalActions: candidateActions, random }) {
      if (candidateActions.length === 0) {
        return undefined;
      }
      if (candidateActions.length === 1) {
        return candidateActions[0];
      }

      const model = await getModel();
      const rootPlayer = view.activePlayerId;
      const rankedRootActions = rankRootActions(candidateActions);
      const worldStates = sampleWorldStates(
        state,
        view,
        rootPlayer,
        config.worlds,
        random
      );
      if (worldStates.length === 0) {
        return rankedRootActions[0].action;
      }

      const actionByKey = new Map(
        rankedRootActions.map((candidate) => [candidate.actionKey, candidate.action])
      );
      const rootPriorByKey = rootPriorsByKey(candidateActions);
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

      let bestActionKey = expandedKeys[0];
      let bestVisits = rootVisits.get(bestActionKey) ?? 0;
      let bestValue = safeDiv(rootValueSum.get(bestActionKey) ?? 0, bestVisits);
      let bestPrior = rootPriorByKey.get(bestActionKey) ?? 0;
      for (const actionKey of expandedKeys.slice(1)) {
        const visits = rootVisits.get(actionKey) ?? 0;
        const value = safeDiv(rootValueSum.get(actionKey) ?? 0, visits);
        const prior = rootPriorByKey.get(actionKey) ?? 0;
        if (
          visits > bestVisits ||
          (visits === bestVisits &&
            (value > bestValue ||
              (approximatelyEqual(value, bestValue) &&
                (prior > bestPrior ||
                  (approximatelyEqual(prior, bestPrior) &&
                    actionKey.localeCompare(bestActionKey) < 0)))))
        ) {
          bestActionKey = actionKey;
          bestVisits = visits;
          bestValue = value;
          bestPrior = prior;
        }
      }

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

function resolveTdSearchConfig(options: TdSearchPolicyOptions): TdSearchPolicyConfig {
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
    options.opponentTemperature
    ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.opponentTemperature;
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
      options.sampleOpponentActions
      ?? DEFAULT_TD_SEARCH_POLICY_CONFIG.sampleOpponentActions,
  };
}

function rankRootActions(candidateActions: readonly GameAction[]): KeyedAction[] {
  const keyed = toKeyedActions(candidateActions);
  const ranked = [...keyed].sort((left, right) => {
    const scoreDelta = actionScore(right.action) - actionScore(left.action);
    if (!approximatelyEqual(scoreDelta, 0)) {
      return scoreDelta > 0 ? 1 : -1;
    }
    return left.actionKey.localeCompare(right.actionKey);
  });
  return ranked;
}

function rootPriorsByKey(candidateActions: readonly GameAction[]): Map<string, number> {
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length === 0) {
    return new Map<string, number>();
  }
  const scores = keyed.map((candidate) => actionScore(candidate.action));
  let maxScore = Number.NEGATIVE_INFINITY;
  for (const score of scores) {
    maxScore = Math.max(maxScore, score);
  }
  const expScores = scores.map((score) => Math.exp(score - maxScore));
  const normalizer = expScores.reduce((sum, score) => sum + score, 0);
  const priors = new Map<string, number>();
  if (!Number.isFinite(normalizer) || normalizer <= 0) {
    const uniform = 1 / keyed.length;
    for (const candidate of keyed) {
      priors.set(candidate.actionKey, uniform);
    }
    return priors;
  }
  keyed.forEach((candidate, index) => {
    priors.set(candidate.actionKey, expScores[index] / normalizer);
  });
  return priors;
}

function progressiveTargetActionCount(
  totalActions: number,
  initialActions: number,
  visits: number
): number {
  if (totalActions <= 0) {
    return 0;
  }
  const base = Math.max(1, Math.min(initialActions, totalActions));
  const widened = base + Math.floor(Math.sqrt(Math.max(0, visits) + 1));
  return Math.min(totalActions, widened);
}

function selectRootUcbAction(
  actionKeys: readonly string[],
  visitsByKey: ReadonlyMap<string, number>,
  valueSumByKey: ReadonlyMap<string, number>,
  priorsByKey: ReadonlyMap<string, number>,
  totalVisits: number,
  cPuct = 1
): string {
  if (actionKeys.length === 0) {
    throw new Error('selectRootUcbAction requires at least one action key.');
  }
  if (!(cPuct > 0)) {
    throw new Error('selectRootUcbAction requires cPuct > 0.');
  }

  const sqrtParent = Math.sqrt(Math.max(0, totalVisits) + 1);
  let bestActionKey = actionKeys[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const actionKey of actionKeys) {
    const visits = visitsByKey.get(actionKey) ?? 0;
    const valueSum = valueSumByKey.get(actionKey) ?? 0;
    const q = safeDiv(valueSum, visits);
    const prior = priorsByKey.get(actionKey) ?? 0;
    const score = q + (cPuct * prior * sqrtParent) / (1 + visits);
    if (
      score > bestScore ||
      (approximatelyEqual(score, bestScore) &&
        actionKey.localeCompare(bestActionKey) < 0)
    ) {
      bestActionKey = actionKey;
      bestScore = score;
    }
  }
  return bestActionKey;
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
    const actions = legalActions(state);
    if (actions.length === 0) {
      break;
    }
    const activePlayer = state.players[state.activePlayerIndex]?.id;
    if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
      throw new Error('TD search rollout could not resolve active player.');
    }

    const nextActionKey =
      activePlayer === rootPlayer
        ? chooseRootRolloutActionKey(actions, random, config.rolloutEpsilon)
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
  actions: readonly GameAction[],
  random: () => number,
  rolloutEpsilon: number
): string {
  const keyed = toKeyedActions(actions);
  if (random() < rolloutEpsilon) {
    return keyed[Math.floor(random() * keyed.length)].actionKey;
  }
  return rankRootActions(actions)[0].actionKey;
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
  const view = toPlayerView(state, activePlayer);
  const observation = encodeObservation(view);
  const actionFeatures = encodeActionCandidates(actions);
  const logits = model.opponentScorer.logits(observation, actionFeatures);
  if (logits.length !== keyed.length) {
    throw new Error(
      `TD search opponent logits length mismatch. logits=${String(logits.length)} actions=${String(keyed.length)}.`
    );
  }

  const scaledLogits = Array.from(logits, (value) => value / config.opponentTemperature);
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
  const activePlayer = state.players[state.activePlayerIndex]?.id;
  if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
    throw new Error('TD search leaf value could not resolve active player.');
  }
  const view = toPlayerView(state, activePlayer);
  const observation = encodeObservation(view);
  const activeValue = model.valueScorer.predict(observation);
  const rootValue = activePlayer === rootPlayer ? activeValue : -activeValue;
  return clamp(rootValue, -1, 1);
}

function argmaxLogitsIndex(logits: readonly number[], keyed: readonly KeyedAction[]): number {
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

function sampleLogitsIndex(logits: readonly number[], random: () => number): number {
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
  const actions = legalActions(state);
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

function sampleWorldStates(
  state: GameState,
  view: PlayerView,
  rootPlayer: PlayerId,
  worldCount: number,
  random: () => number
): GameState[] {
  const opponentPlayer = otherPlayerId(rootPlayer);
  const rootView = requiredPlayerView(view, rootPlayer);
  const opponentView = requiredPlayerView(view, opponentPlayer);

  const rootHand = [...rootView.hand];
  const opponentHandCount = opponentView.handCount;
  const drawCount = view.deck.drawCount;

  const knownCards = new Set<string>(rootHand);
  for (const cardId of view.deck.discard) {
    knownCards.add(cardId);
  }
  for (const cardId of districtPropertyCards(view)) {
    knownCards.add(cardId);
  }

  const hiddenPool = PROPERTY_CARD_IDS.filter(
    (cardId) => !knownCards.has(cardId)
  );
  const expectedHiddenCount = opponentHandCount + drawCount;
  if (hiddenPool.length !== expectedHiddenCount) {
    throw new Error(
      `TD search determinization mismatch: expected hidden=${String(expectedHiddenCount)} but found ${String(hiddenPool.length)}.`
    );
  }

  const worlds: GameState[] = [];
  for (let index = 0; index < worldCount; index += 1) {
    const sampledHidden = [...hiddenPool];
    shuffleInPlace(sampledHidden, random);
    const opponentHand = sampledHidden.slice(0, opponentHandCount);
    const draw = sampledHidden.slice(
      opponentHandCount,
      opponentHandCount + drawCount
    );

    const world = structuredClone(state) as GameState;
    world.players = world.players.map((player) => {
      if (player.id === rootPlayer) {
        return { ...player, hand: [...rootHand] };
      }
      if (player.id === opponentPlayer) {
        return { ...player, hand: [...opponentHand] };
      }
      return player;
    });
    world.deck = {
      ...world.deck,
      draw: [...draw],
    };
    worlds.push(world);
  }
  return worlds;
}

function requiredPlayerView(
  view: PlayerView,
  playerId: PlayerId
): PlayerView['players'][number] {
  const player = view.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`TD search view is missing player ${playerId}.`);
  }
  return player;
}

function districtPropertyCards(view: PlayerView): Set<string> {
  const cards = new Set<string>();
  for (const district of view.districts) {
    for (const playerId of ['PlayerA', 'PlayerB'] as const) {
      const stack = district.stacks[playerId];
      for (const cardId of stack.developed) {
        cards.add(cardId);
      }
      if (stack.deed) {
        cards.add(stack.deed.cardId);
      }
    }
  }
  return cards;
}

function actionScore(action: GameAction): number {
  let score = {
    'develop-outright': 8,
    'develop-deed': 6,
    'buy-deed': 5,
    'choose-income-suit': 4,
    trade: 2,
    'sell-card': 1,
    'end-turn': 0,
  }[action.type];

  const cardRank = 'cardId' in action ? propertyRank(action.cardId) : 0;

  if (action.type === 'develop-outright' || action.type === 'develop-deed') {
    score += cardRank * 0.4;
  }
  if (action.type === 'buy-deed') {
    score += cardRank * 0.25;
    if (cardRank <= 2) {
      score -= 1.5;
    }
  }
  if (action.type === 'sell-card') {
    score -= cardRank * 0.3;
  }
  if (action.type === 'trade') {
    if (action.give === action.receive) {
      score -= 10;
    } else {
      score += 0.2;
    }
  }
  return score;
}

function propertyRank(cardId: string): number {
  const numeric = Number.parseInt(cardId, 10);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  if (numeric >= 0 && numeric <= 5) {
    return 1;
  }
  if (numeric >= 6 && numeric <= 29) {
    return 2 + Math.floor((numeric - 6) / 3);
  }
  if (numeric >= 30 && numeric <= 35) {
    return 10;
  }
  return 0;
}

function terminalValue(state: GameState, rootPlayer: PlayerId): number {
  const winner = state.finalScore?.winner;
  if (!winner || winner === 'Draw') {
    return 0;
  }
  return winner === rootPlayer ? 1 : -1;
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
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

function safeDiv(total: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return total / count;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
