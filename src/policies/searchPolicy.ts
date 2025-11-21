import { legalActions } from '../engine/actionBuilders';
import {
  actionStableKey,
  toKeyedActions,
  type KeyedAction,
} from '../engine/actionSurface';
import { CARD_BY_ID, PROPERTY_CARDS } from '../engine/cards';
import { shuffleInPlace } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { stepToDecision } from '../engine/session';
import { SUITS } from '../engine/stateHelpers';
import type {
  DistrictStack,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PlayerView,
} from '../engine/types';
import type { ActionPolicy } from './types';

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
  rolloutEpsilon: 0.1,
};

export function createSearchPolicy(
  options: SearchPolicyOptions = {}
): ActionPolicy {
  const config = resolveSearchConfig(options);
  return {
    selectAction({ view, state, legalActions: candidateActions, random }) {
      if (candidateActions.length === 0) {
        return undefined;
      }
      if (candidateActions.length === 1) {
        return candidateActions[0];
      }

      const rootPlayer = view.activePlayerId;
      const rootCandidates = rankRootCandidates(
        candidateActions,
        config.maxRootActions
      );
      const worldStates = sampleWorldStates(
        state,
        view,
        rootPlayer,
        config.worlds,
        random
      );

      let selected = rootCandidates[0];
      let bestScore = Number.NEGATIVE_INFINITY;
      let bestPrior = actionScore(selected.action);

      for (const candidate of rootCandidates) {
        const score = evaluateCandidateAcrossWorlds(
          worldStates,
          rootPlayer,
          candidate.actionKey,
          config,
          random
        );
        const prior = actionScore(candidate.action);
        if (
          score > bestScore ||
          (approximatelyEqual(score, bestScore) &&
            (prior > bestPrior ||
              (approximatelyEqual(prior, bestPrior) &&
                candidate.actionKey < selected.actionKey)))
        ) {
          selected = candidate;
          bestScore = score;
          bestPrior = prior;
        }
      }

      return selected.action;
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

function rankRootCandidates(
  candidateActions: readonly GameAction[],
  maxRootActions: number
): KeyedAction[] {
  const keyed = toKeyedActions(candidateActions);
  const ranked = [...keyed].sort((left, right) => {
    const scoreDelta = actionScore(right.action) - actionScore(left.action);
    if (!approximatelyEqual(scoreDelta, 0)) {
      return scoreDelta > 0 ? 1 : -1;
    }
    return left.actionKey.localeCompare(right.actionKey);
  });
  return ranked.slice(0, Math.min(ranked.length, maxRootActions));
}

function evaluateCandidateAcrossWorlds(
  worldStates: readonly GameState[],
  rootPlayer: PlayerId,
  rootActionKey: string,
  config: SearchPolicyConfig,
  random: () => number
): number {
  let total = 0;
  let count = 0;
  for (const worldState of worldStates) {
    for (let index = 0; index < config.rollouts; index += 1) {
      total += runRollout(
        worldState,
        rootPlayer,
        rootActionKey,
        config,
        random
      );
      count += 1;
    }
  }
  if (count === 0) {
    return 0;
  }
  return total / count;
}

function runRollout(
  initialState: GameState,
  rootPlayer: PlayerId,
  rootActionKey: string,
  config: SearchPolicyConfig,
  random: () => number
): number {
  let state = stepByActionKey(initialState, rootActionKey);
  let depth = 0;

  while (!isTerminal(state) && depth < config.depth) {
    const actions = legalActions(state);
    if (actions.length === 0) {
      break;
    }
    const nextActionKey = chooseRolloutActionKey(
      actions,
      random,
      config.rolloutEpsilon
    );
    state = stepByActionKey(state, nextActionKey);
    depth += 1;
  }

  if (isTerminal(state)) {
    return terminalValue(state, rootPlayer);
  }
  return heuristicStateValue(state, rootPlayer);
}

function chooseRolloutActionKey(
  actions: readonly GameAction[],
  random: () => number,
  rolloutEpsilon: number
): string {
  const keyed = toKeyedActions(actions);
  if (random() < rolloutEpsilon) {
    return keyed[Math.floor(random() * keyed.length)].actionKey;
  }
  return rankRootCandidates(actions, 1)[0].actionKey;
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
  const card = CARD_BY_ID[cardId];
  return card && card.kind === 'Property' ? card.rank : 0;
}

function terminalValue(state: GameState, rootPlayer: PlayerId): number {
  const winner = state.finalScore?.winner;
  if (!winner || winner === 'Draw') {
    return 0;
  }
  return winner === rootPlayer ? 1 : -1;
}

function heuristicStateValue(state: GameState, rootPlayer: PlayerId): number {
  const opponent = otherPlayerId(rootPlayer);
  const root = requiredPlayerState(state, rootPlayer);
  const opponentState = requiredPlayerState(state, opponent);
  const handDiff = root.hand.length - opponentState.hand.length;
  const resourceDiff = resourceTotal(root) - resourceTotal(opponentState);

  let districtLead = 0;
  let rankDiff = 0;
  let progressDiff = 0;
  for (const district of state.districts) {
    const rootStack = district.stacks[rootPlayer];
    const opponentStack = district.stacks[opponent];
    const rootScore = stackScore(rootStack);
    const opponentScore = stackScore(opponentStack);
    rankDiff += rootScore.rankTotal - opponentScore.rankTotal;
    progressDiff += rootScore.progressTotal - opponentScore.progressTotal;
    if (rootScore.rankTotal > opponentScore.rankTotal) {
      districtLead += 1;
    } else if (rootScore.rankTotal < opponentScore.rankTotal) {
      districtLead -= 1;
    }
  }

  const districtTerm = districtLead / Math.max(1, state.districts.length);
  const rankTerm = Math.tanh(rankDiff / 18);
  const progressTerm = Math.tanh(progressDiff / 8);
  const resourceTerm = Math.tanh(resourceDiff / 10);
  const handTerm = Math.tanh(handDiff / 4);

  const score =
    0.55 * districtTerm +
    0.2 * rankTerm +
    0.1 * progressTerm +
    0.1 * resourceTerm +
    0.05 * handTerm;
  return Math.max(-1, Math.min(1, score));
}

function requiredPlayerState(
  state: GameState,
  playerId: PlayerId
): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Search policy state is missing player ${playerId}.`);
  }
  return player;
}

function resourceTotal(player: PlayerState): number {
  let total = 0;
  for (const suit of SUITS) {
    total += player.resources[suit];
  }
  return total;
}

function stackScore(stack: DistrictStack): {
  rankTotal: number;
  progressTotal: number;
} {
  let rankTotal = 0;
  let progressTotal = 0;

  for (const cardId of stack.developed) {
    rankTotal += propertyRank(cardId);
  }

  if (stack.deed) {
    rankTotal += propertyRank(stack.deed.cardId);
    progressTotal += stack.deed.progress;
    rankTotal += 0.35 * stack.deed.progress;
  }
  return { rankTotal, progressTotal };
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}
