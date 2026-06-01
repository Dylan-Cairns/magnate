import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { PROPERTY_CARDS } from '../engine/cards';
import { shuffleInPlace } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import type {
  GameAction,
  GameState,
  PlayerId,
  PlayerView,
} from '../engine/types';
import { heuristicPriorsByKey, rankHeuristicActions } from './heuristicScorer';
import { evaluateSearchLeafState } from './searchStateEvaluator';
import type { ActionPolicy, SearchDecisionDiagnostics } from './types';

const PROPERTY_CARD_IDS = PROPERTY_CARDS.map((card) => card.id);

export interface SearchPolicyConfig {
  worlds: number;
  rollouts: number;
  depth: number;
  maxRootActions: number;
  rolloutEpsilon: number;
}

export type SearchPolicyOptions = Partial<SearchPolicyConfig>;

const DEFAULT_SEARCH_POLICY_CONFIG: SearchPolicyConfig = {
  worlds: 4,
  rollouts: 1,
  depth: 12,
  maxRootActions: 6,
  rolloutEpsilon: 0.04,
};

export function createSearchPolicy(
  options: SearchPolicyOptions = {}
): ActionPolicy {
  const config = resolveSearchConfig(options);
  return {
    selectAction({
      view,
      state,
      legalActions: candidateActions,
      random,
      onSearchDiagnostics,
    }) {
      if (candidateActions.length === 0) {
        return undefined;
      }
      if (candidateActions.length === 1) {
        return candidateActions[0];
      }

      const rootPlayer = view.activePlayerId;
      const heuristicContext = { state, view };
      const rankedRootActions = rankHeuristicActions(
        candidateActions,
        heuristicContext
      );
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
      let simulatedActionSteps = 0;
      let terminalRollouts = 0;
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
        const rollout = runRollout(
          worldStates[worldIndex],
          rootPlayer,
          actionKey,
          config,
          random
        );
        const score = rollout.score;
        simulatedActionSteps += rollout.simulatedActionSteps;
        terminalRollouts += rollout.terminatedBeforeDepthLimit ? 1 : 0;
        rootVisits.set(actionKey, (rootVisits.get(actionKey) ?? 0) + 1);
        rootValueSum.set(actionKey, (rootValueSum.get(actionKey) ?? 0) + score);
      }

      onSearchDiagnostics?.(
        searchDecisionDiagnostics({
          config,
          legalRootActions: candidateActions.length,
          expandedRootActions: expandedKeys.length,
          rootVisitBudget: rootBudget,
          simulatedActionSteps,
          terminalRollouts,
        })
      );

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
          `Search policy selected root action key that is no longer legal: ${bestActionKey}.`
        );
      }
      return selected;
    },
  };
}

function resolveSearchConfig(options: SearchPolicyOptions): SearchPolicyConfig {
  const worlds = integerWithFloor(
    options.worlds ?? DEFAULT_SEARCH_POLICY_CONFIG.worlds,
    1
  );
  const rollouts = integerWithFloor(
    options.rollouts ?? DEFAULT_SEARCH_POLICY_CONFIG.rollouts,
    1
  );
  const depth = integerWithFloor(
    options.depth ?? DEFAULT_SEARCH_POLICY_CONFIG.depth,
    1
  );
  const maxRootActions = integerWithFloor(
    options.maxRootActions ?? DEFAULT_SEARCH_POLICY_CONFIG.maxRootActions,
    1
  );
  const rolloutEpsilon =
    options.rolloutEpsilon ?? DEFAULT_SEARCH_POLICY_CONFIG.rolloutEpsilon;
  if (
    !Number.isFinite(rolloutEpsilon) ||
    rolloutEpsilon < 0 ||
    rolloutEpsilon > 1
  ) {
    throw new Error(
      `Search policy rolloutEpsilon must be in [0, 1]; received ${String(rolloutEpsilon)}.`
    );
  }
  return {
    worlds,
    rollouts,
    depth,
    maxRootActions,
    rolloutEpsilon,
  };
}

function integerWithFloor(value: number, floor: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(
      `Search policy expected a finite number; received ${String(value)}.`
    );
  }
  const rounded = Math.trunc(value);
  if (rounded < floor) {
    throw new Error(
      `Search policy value must be >= ${String(floor)}; received ${String(value)}.`
    );
  }
  return rounded;
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

function runRollout(
  initialState: GameState,
  rootPlayer: PlayerId,
  rootActionKey: string,
  config: SearchPolicyConfig,
  random: () => number
): {
  score: number;
  simulatedActionSteps: number;
  terminatedBeforeDepthLimit: boolean;
} {
  let state = stepByActionKey(initialState, rootActionKey);
  let depth = 0;
  let simulatedActionSteps = 1;

  while (!isTerminal(state) && depth < config.depth) {
    const actions = legalActions(state);
    if (actions.length === 0) {
      break;
    }
    const nextActionKey = chooseRolloutActionKey(
      state,
      actions,
      random,
      config.rolloutEpsilon
    );
    state = stepByActionKey(state, nextActionKey);
    depth += 1;
    simulatedActionSteps += 1;
  }

  const terminatedBeforeDepthLimit = isTerminal(state) && depth < config.depth;
  if (isTerminal(state)) {
    return {
      score: terminalValue(state, rootPlayer),
      simulatedActionSteps,
      terminatedBeforeDepthLimit,
    };
  }
  return {
    score: evaluateSearchLeafState(state, rootPlayer),
    simulatedActionSteps,
    terminatedBeforeDepthLimit,
  };
}

function searchDecisionDiagnostics({
  config,
  legalRootActions,
  expandedRootActions,
  rootVisitBudget,
  simulatedActionSteps,
  terminalRollouts,
}: {
  config: SearchPolicyConfig;
  legalRootActions: number;
  expandedRootActions: number;
  rootVisitBudget: number;
  simulatedActionSteps: number;
  terminalRollouts: number;
}): SearchDecisionDiagnostics {
  return {
    kind: 'search',
    legalRootActions,
    expandedRootActions,
    rootVisitBudget,
    configProxyCost: rootVisitBudget * config.depth,
    maxSimulatedActionSteps: rootVisitBudget * (config.depth + 1),
    simulatedActionSteps,
    terminalRollouts,
  };
}

function chooseRolloutActionKey(
  state: GameState,
  actions: readonly GameAction[],
  random: () => number,
  rolloutEpsilon: number
): string {
  const keyed = toKeyedActions(actions);
  if (random() < rolloutEpsilon) {
    return keyed[Math.floor(random() * keyed.length)].actionKey;
  }
  return rankHeuristicActions(actions, { state })[0].actionKey;
}

function stepByActionKey(state: GameState, actionKey: string): GameState {
  const actions = legalActions(state);
  const action = actions.find(
    (candidate) => actionStableKey(candidate) === actionKey
  );
  if (!action) {
    throw new Error(
      `Search policy rollout could not find legal action for key ${actionKey}.`
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
      `Search determinization mismatch: expected hidden=${String(expectedHiddenCount)} but found ${String(hiddenPool.length)}.`
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
    throw new Error(`Search policy view is missing player ${playerId}.`);
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

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}

function safeDiv(total: number, count: number): number {
  if (count <= 0) {
    return 0;
  }
  return total / count;
}
