import { toKeyedActions, type KeyedAction } from '../engine/actionSurface';
import { CARD_BY_ID, type CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import {
  canAfford,
  deedCost,
  developmentCost,
  enumerateOutrightPayments,
  findProperty,
  placementAllowed,
  sumTokens,
  SUITS,
} from '../engine/stateHelpers';
import type {
  DistrictId,
  DistrictStack,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PlayerView,
  PropertyCard,
  ResourcePool,
  Suit,
} from '../engine/types';

export interface HeuristicSelectionContext {
  state: GameState;
  view: PlayerView;
  legalActions: readonly GameAction[];
}

export interface HeuristicScoredAction extends KeyedAction {
  score: number;
  prior: number;
  rank: number;
}

type HeuristicEvaluationContext = Partial<
  Pick<HeuristicSelectionContext, 'state' | 'view' | 'legalActions'>
>;

type DistrictWinner = PlayerId | 'Tie';

interface DistrictStatus {
  district: DistrictState;
  activeScore: number;
  opponentScore: number;
  margin: number;
  winner: DistrictWinner;
}

interface DistrictPlan {
  mainDistrictIds: ReadonlySet<DistrictId>;
  controlledCount: number;
}

const ACTION_BASE_SCORE: Record<GameAction['type'], number> = {
  'develop-outright': 8,
  'develop-deed': 6,
  'buy-deed': 5,
  'choose-income-suit': 4,
  trade: -1,
  'sell-card': 1,
  'end-turn': 0,
};

const TARGET_CONTROLLED_DISTRICTS = 3;
const LATE_GAME_DRAW_COUNT = 6;
const HIGH_RANK_DEED_THRESHOLD = 7;
const HIGH_RANK_ACCESS_THRESHOLD = 8;
const OVERKILL_MARGIN = 7;

export function selectHeuristicAction(
  context: HeuristicSelectionContext
): GameAction | undefined {
  const ranked = rankHeuristicActions(context.legalActions, context);
  return ranked[0]?.action;
}

export function rankHeuristicActions(
  candidateActions: readonly GameAction[],
  context: HeuristicEvaluationContext = {}
): KeyedAction[] {
  const keyed = toKeyedActions(candidateActions);
  return [...keyed].sort((left, right) =>
    compareKeyedActionsByHeuristic(left, right, context)
  );
}

export function scoreHeuristicActions(
  candidateActions: readonly GameAction[],
  context: HeuristicEvaluationContext = {}
): HeuristicScoredAction[] {
  const priors = heuristicPriorsByKey(candidateActions, context);
  return rankHeuristicActions(candidateActions, context).map((candidate, index) => ({
    ...candidate,
    score: scoreHeuristicAction(candidate.action, context),
    prior: priors.get(candidate.actionKey) ?? 0,
    rank: index,
  }));
}

export function heuristicPriorsByKey(
  candidateActions: readonly GameAction[],
  context: HeuristicEvaluationContext = {}
): Map<string, number> {
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length === 0) {
    return new Map<string, number>();
  }

  const scores = keyed.map((candidate) =>
    scoreHeuristicAction(candidate.action, context)
  );
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

export function scoreHeuristicAction(
  action: GameAction,
  context: HeuristicEvaluationContext = {}
): number {
  let score = ACTION_BASE_SCORE[action.type];
  const card = 'cardId' in action ? propertyCard(action.cardId) : undefined;
  const cardRank = card?.rank ?? 0;

  if (action.type === 'develop-outright' || action.type === 'develop-deed') {
    score += cardRank * 0.4;
  }
  if (action.type === 'buy-deed') {
    score += cardRank * 0.25;
    if (cardRank <= 2) {
      score -= 100.0;
    }
    if (cardRank >= HIGH_RANK_DEED_THRESHOLD && isLateGame(context.state)) {
      score -= 35.0 + cardRank;
    }
  }
  if (action.type === 'sell-card') {
    score -= cardRank * 0.3;
  }
  if (action.type === 'trade') {
    if (action.give === action.receive) {
      score -= 10;
    }
  }
  score += contextScore(action, context);
  return score;
}

function compareKeyedActionsByHeuristic(
  left: KeyedAction,
  right: KeyedAction,
  context: HeuristicEvaluationContext
): number {
  const scoreDelta =
    scoreHeuristicAction(right.action, context) -
    scoreHeuristicAction(left.action, context);
  if (!approximatelyEqual(scoreDelta, 0)) {
    return scoreDelta > 0 ? 1 : -1;
  }
  return left.actionKey.localeCompare(right.actionKey);
}

function contextScore(
  action: GameAction,
  context: HeuristicEvaluationContext
): number {
  const resolved = resolveContext(context);
  if (!resolved) {
    return 0;
  }

  const { state, activePlayerId, activePlayer } = resolved;
  const opponentId = otherPlayerId(activePlayerId);
  const plan = districtPlan(state, activePlayerId);
  let score = resourceDeltaScore(action, activePlayer);

  if (action.type === 'trade') {
    score += tradeContextScore({
      action,
      state,
      activePlayerId,
      activePlayer,
      opponentId,
      plan,
    });
  }

  if (action.type === 'buy-deed') {
    score += deedStartScore({
      action,
      state,
      activePlayerId,
      activePlayer,
    });
  }

  if (isDistrictInvestment(action)) {
    score += districtInvestmentScore({
      action,
      state,
      activePlayerId,
      activePlayer,
      opponentId,
      plan,
    });
  }

  if (action.type === 'choose-income-suit') {
    score += incomeChoiceScore(action, state, activePlayerId);
  }

  if (isLateGame(state)) {
    score += endgameActionScore(action, state, activePlayerId, opponentId);
  }

  return score;
}

function resolveContext(context: HeuristicEvaluationContext):
  | {
      state: GameState;
      activePlayerId: PlayerId;
      activePlayer: PlayerState;
    }
  | undefined {
  const state = context.state;
  if (!state) {
    return undefined;
  }
  const activePlayerId =
    context.view?.activePlayerId ?? state.players[state.activePlayerIndex]?.id;
  if (activePlayerId !== 'PlayerA' && activePlayerId !== 'PlayerB') {
    return undefined;
  }
  const activePlayer = state.players.find((player) => player.id === activePlayerId);
  if (!activePlayer) {
    return undefined;
  }
  return {
    state,
    activePlayerId,
    activePlayer,
  };
}

function resourceDeltaScore(action: GameAction, player: PlayerState): number {
  const delta = resourceDeltaForAction(action);
  const before = player.resources;
  const after = applyResourceDelta(before, delta);
  const beforeExposure = taxExposure(before);
  const afterExposure = taxExposure(after);
  const crownCounts = crownSuitCounts(player);
  let score = (beforeExposure - afterExposure) * 0.65;

  for (const suit of SUITS) {
    const beforeCount = before[suit];
    const afterCount = after[suit];
    if (beforeCount <= 0 && afterCount >= 1) {
      score += 1.6;
    }
    if (beforeCount >= 1 && afterCount <= 0) {
      score -= (crownCounts[suit] ?? 0) > 0 ? 1.3 : 2.4;
    }
    if (beforeCount > 1 && afterCount === 1) {
      score += 0.4;
    }
    if (beforeCount === 1 && afterCount > 1) {
      score -= 0.15;
    }
    if (afterCount >= 3 && afterCount > beforeCount) {
      score -= Math.min(3.0, (afterCount - 2) * 0.7);
    }
  }
  if (action.type === 'develop-outright') {
    score += surplusTokenCount(action.payment, before) * 0.7;
  }
  return score;
}

function tradeContextScore({
  action,
  state,
  activePlayerId,
  activePlayer,
  opponentId,
  plan,
}: {
  action: Extract<GameAction, { type: 'trade' }>;
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  opponentId: PlayerId;
  plan: DistrictPlan;
}): number {
  if (action.give === action.receive) {
    return 0;
  }

  const before = activePlayer.resources;
  const after = applyResourceDelta(before, resourceDeltaForAction(action));
  const unlockScore = bestTradeUnlockScore({
    state,
    activePlayerId,
    activePlayer,
    opponentId,
    plan,
    before,
    after,
  });
  return unlockScore > 0 ? unlockScore : -4.0;
}

function bestTradeUnlockScore({
  state,
  activePlayerId,
  activePlayer,
  opponentId,
  plan,
  before,
  after,
}: {
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  opponentId: PlayerId;
  plan: DistrictPlan;
  before: ResourcePool;
  after: ResourcePool;
}): number {
  const playerAfterTrade: PlayerState = {
    ...activePlayer,
    resources: after,
  };
  let best = 0;

  for (const district of state.districts) {
    const deed = district.stacks[activePlayerId].deed;
    if (!deed) {
      continue;
    }
    const card = propertyCard(deed.cardId);
    if (!card) {
      continue;
    }
    for (const suit of card.suits) {
      if (before[suit] > 0 || after[suit] <= 0) {
        continue;
      }
      const developAction: Extract<GameAction, { type: 'develop-deed' }> = {
        type: 'develop-deed',
        districtId: district.id,
        cardId: deed.cardId,
        tokens: { [suit]: 1 },
      };
      const completion = projectedDevelopDeedCompletion(
        developAction,
        district,
        activePlayerId
      );
      const districtScoreAfterTrade = districtInvestmentScore({
        action: developAction,
        state,
        activePlayerId,
        activePlayer: playerAfterTrade,
        opponentId,
        plan,
      });
      if (completion.completes) {
        best = Math.max(best, 14.0);
      } else if (districtScoreAfterTrade >= 8.0) {
        best = Math.max(best, 8.0);
      }
    }
  }

  for (const cardId of activePlayer.hand) {
    const card = propertyCard(cardId);
    if (!card) {
      continue;
    }
    const beforePayments = enumerateOutrightPayments(card, before);
    const afterPayments = enumerateOutrightPayments(card, after);
    if (beforePayments.length === 0 && afterPayments.length > 0) {
      best = Math.max(
        best,
        bestUnlockedOutrightScore({
          state,
          activePlayerId,
          activePlayer: playerAfterTrade,
          opponentId,
          plan,
          card,
          payment: afterPayments[0],
        })
      );
    }

    const deedCostForCard = deedCost(card);
    const buyUnlocked = !canAfford(before, deedCostForCard) &&
      canAfford(after, deedCostForCard);
    if (buyUnlocked && card.rank > 2 && !isLateGame(state)) {
      best = Math.max(
        best,
        bestUnlockedBuyDeedScore({
          state,
          activePlayerId,
          activePlayer: playerAfterTrade,
          opponentId,
          plan,
          card,
        })
      );
    }
  }

  return best;
}

function bestUnlockedOutrightScore({
  state,
  activePlayerId,
  activePlayer,
  opponentId,
  plan,
  card,
  payment,
}: {
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  opponentId: PlayerId;
  plan: DistrictPlan;
  card: PropertyCard;
  payment: Partial<Record<Suit, number>>;
}): number {
  let best = 0;
  for (const district of state.districts) {
    if (!placementAllowed(card, district, activePlayerId)) {
      continue;
    }
    const action: Extract<GameAction, { type: 'develop-outright' }> = {
      type: 'develop-outright',
      cardId: card.id,
      districtId: district.id,
      payment,
    };
    const score = districtInvestmentScore({
      action,
      state,
      activePlayerId,
      activePlayer,
      opponentId,
      plan,
    });
    if (score >= 20.0) {
      best = Math.max(best, 12.0);
    } else if (score >= 8.0) {
      best = Math.max(best, 8.0);
    }
  }
  return best;
}

function bestUnlockedBuyDeedScore({
  state,
  activePlayerId,
  activePlayer,
  opponentId,
  plan,
  card,
}: {
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  opponentId: PlayerId;
  plan: DistrictPlan;
  card: PropertyCard;
}): number {
  let best = 0;
  for (const district of state.districts) {
    if (!placementAllowed(card, district, activePlayerId)) {
      continue;
    }
    const action: Extract<GameAction, { type: 'buy-deed' }> = {
      type: 'buy-deed',
      cardId: card.id,
      districtId: district.id,
    };
    const score = districtInvestmentScore({
      action,
      state,
      activePlayerId,
      activePlayer,
      opponentId,
      plan,
    });
    if (plan.mainDistrictIds.has(district.id) && score >= 8.0) {
      best = Math.max(best, 5.0);
    }
  }
  return best;
}

function deedStartScore({
  action,
  state,
  activePlayerId,
  activePlayer,
}: {
  action: Extract<GameAction, { type: 'buy-deed' }>;
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
}): number {
  const card = propertyCard(action.cardId);
  if (!card) {
    return 0;
  }

  let score = 0;
  const unfinishedDeedCount = countUnfinishedDeeds(state, activePlayerId);
  if (unfinishedDeedCount > 0) {
    score -= Math.min(12.0, unfinishedDeedCount * (card.rank >= 7 ? 4.0 : 3.0));
  }

  if (card.rank >= HIGH_RANK_ACCESS_THRESHOLD) {
    const afterPurchase = applyResourceDelta(
      activePlayer.resources,
      negateTokens(deedCost(card))
    );
    const missingSuitAccess = card.suits.filter(
      (suit) =>
        !hasRealisticSuitAccess({
          suit,
          state,
          activePlayerId,
          activePlayer,
          resourcesAfterAction: afterPurchase,
          excludedCardId: action.cardId,
        })
    );
    score -= missingSuitAccess.length * (card.rank >= 9 ? 7.0 : 5.0);
  }

  return score;
}

function endgameActionScore(
  action: GameAction,
  state: GameState,
  activePlayerId: PlayerId,
  opponentId: PlayerId
): number {
  let score = 0;
  if (isDistrictInvestment(action)) {
    const district = state.districts.find(
      (candidate) => candidate.id === action.districtId
    );
    const card = propertyCard(action.cardId);
    if (!district || !card) {
      return 0;
    }
    const current = districtStatus(district, activePlayerId, opponentId);
    const projected = projectedDistrictStatus({
      action,
      district,
      activePlayerId,
      opponentId,
    });
    const districtPointDelta =
      districtControlValue(projected.winner, activePlayerId) -
      districtControlValue(current.winner, activePlayerId);
    score += districtPointDelta * 14.0;

    if (projected.winner === activePlayerId) {
      score += Math.min(4.0, card.rank * 0.3);
    }
    if (action.type === 'buy-deed') {
      score -= 7.0 + card.rank * 0.6;
    }
    if (action.type === 'develop-deed') {
      const completion = projectedDevelopDeedCompletion(
        action,
        district,
        activePlayerId
      );
      const remainingAfter = Math.max(0, completion.target - completion.progress);
      if (completion.completes) {
        score += 3.0;
      } else {
        score -= 6.0 + remainingAfter * 0.4;
      }
    }
    if (action.type === 'develop-outright' && districtPointDelta <= 0) {
      score -= projected.winner === activePlayerId ? 0.5 : 2.5;
    }
  }

  if (action.type === 'sell-card') {
    const card = propertyCard(action.cardId);
    score += (card?.rank ?? 0) * 0.15;
  }
  if (action.type === 'trade') {
    score -= 1.5;
  }
  return score;
}

function districtInvestmentScore({
  action,
  state,
  activePlayerId,
  activePlayer,
  opponentId,
  plan,
}: {
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >;
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  opponentId: PlayerId;
  plan: DistrictPlan;
}): number {
  const district = state.districts.find((candidate) => candidate.id === action.districtId);
  if (!district) {
    return 0;
  }
  const card = propertyCard(action.cardId);
  if (!card) {
    return 0;
  }

  const current = districtStatus(district, activePlayerId, opponentId);
  const projected = projectedDistrictStatus({
    action,
    district,
    activePlayerId,
    opponentId,
  });
  const mainDistrict = plan.mainDistrictIds.has(district.id);
  const flipsDistrict =
    current.winner !== activePlayerId && projected.winner === activePlayerId;
  const contestsDistrict =
    current.winner !== activePlayerId &&
    projected.winner === 'Tie' &&
    projected.margin >= 0;
  const opponentControlledCount = controlledDistrictCount(
    state,
    opponentId,
    activePlayerId
  );

  let score = 0;
  if (flipsDistrict) {
    score += plan.controlledCount < TARGET_CONTROLLED_DISTRICTS ? 28 : 20;
  } else if (contestsDistrict) {
    score += plan.controlledCount < TARGET_CONTROLLED_DISTRICTS ? 11 : 6;
  }
  if (
    opponentControlledCount >= TARGET_CONTROLLED_DISTRICTS &&
    current.winner === opponentId
  ) {
    if (flipsDistrict) {
      score += 8.0;
    } else if (contestsDistrict) {
      score += 5.0;
    }
  }

  if (current.winner === activePlayerId) {
    score += mainDistrict ? 4.0 : 1.0;
    const opponentCanThreaten =
      opponentEventualMargin(district, activePlayerId, opponentId) >= 0;
    if (current.margin <= 2 || opponentCanThreaten) {
      score += mainDistrict ? 6.0 : 3.5;
    }
    if (plan.controlledCount >= TARGET_CONTROLLED_DISTRICTS && mainDistrict) {
      score += 2.5;
    }
  } else if (current.winner === 'Tie') {
    score += mainDistrict ? 5.0 : 2.0;
  }

  if (mainDistrict) {
    score += 3.0;
  }

  const eventualMargin = eventualDistrictMargin(district, activePlayerId, opponentId);
  const canEventuallyControl = eventualMargin > 0 || flipsDistrict || contestsDistrict;
  if (!mainDistrict && !flipsDistrict) {
    score -= 5.0 + card.rank * 0.6;
  }
  if (!canEventuallyControl && current.winner !== activePlayerId) {
    score -= 8.0 + Math.min(8, Math.max(0, -eventualMargin));
  }
  if (
    current.winner === activePlayerId &&
    projected.winner === activePlayerId &&
    current.margin >= OVERKILL_MARGIN &&
    !isLateGame(state)
  ) {
    score -= Math.min(8.0, (current.margin - OVERKILL_MARGIN + 1) * 1.2);
  }
  if (
    plan.controlledCount >= TARGET_CONTROLLED_DISTRICTS &&
    current.winner !== activePlayerId &&
    !flipsDistrict &&
    !contestsDistrict
  ) {
    score -= mainDistrict ? 2.0 : 5.0;
  }

  if (action.type === 'develop-deed') {
    const completion = projectedDevelopDeedCompletion(action, district, activePlayerId);
    const remainingAfter = Math.max(0, completion.target - completion.progress);
    if (completion.completes) {
      score += 8.0;
    } else if (canEventuallyControl || mainDistrict) {
      score += 3.5;
      if (remainingAfter <= 2) {
        score += 2.5;
      }
    }
    const shelteredSurplus = surplusTokenCount(action.tokens, activePlayer.resources);
    if (shelteredSurplus > 0) {
      const productiveShelter =
        completion.completes ||
        canEventuallyControl ||
        mainDistrict ||
        current.winner === activePlayerId;
      score += shelteredSurplus * (productiveShelter ? 1.8 : -0.7);
    }
  }

  return score;
}

function incomeChoiceScore(
  action: Extract<GameAction, { type: 'choose-income-suit' }>,
  state: GameState,
  activePlayerId: PlayerId
): number {
  const district = state.districts.find((candidate) => candidate.id === action.districtId);
  if (!district) {
    return 0;
  }
  const deed = district.stacks[activePlayerId]?.deed;
  if (!deed) {
    return 0;
  }
  const card = propertyCard(deed.cardId);
  if (!card || !card.suits.includes(action.suit)) {
    return 0;
  }
  const progressNeeded = Math.max(0, developmentCost(card) - deed.progress);
  if (progressNeeded <= 0) {
    return 0;
  }
  return 1.0 + Math.min(2.0, progressNeeded * 0.25);
}

function projectedDistrictStatus({
  action,
  district,
  activePlayerId,
  opponentId,
}: {
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >;
  district: DistrictState;
  activePlayerId: PlayerId;
  opponentId: PlayerId;
}): DistrictStatus {
  const activeStack = district.stacks[activePlayerId];
  const projectedStack = projectedActiveStack(action, activeStack);
  const activeScore = scoreStackDeveloped(projectedStack.developed);
  const opponentScore = districtScore(district.stacks[opponentId]);
  return statusFromScores({
    district,
    activeScore,
    opponentScore,
    activePlayerId,
    opponentId,
  });
}

function projectedActiveStack(
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >,
  stack: DistrictStack
): DistrictStack {
  if (action.type === 'develop-outright') {
    return {
      ...stack,
      developed: [...stack.developed, action.cardId],
    };
  }
  if (action.type === 'develop-deed') {
    const completion = projectedStackDevelopDeedCompletion(action, stack);
    if (!completion.completes) {
      return stack;
    }
    return {
      ...stack,
      developed: [...stack.developed, action.cardId],
      deed: undefined,
    };
  }
  return stack;
}

function projectedDevelopDeedCompletion(
  action: Extract<GameAction, { type: 'develop-deed' }>,
  district: DistrictState,
  activePlayerId: PlayerId
): { completes: boolean; progress: number; target: number } {
  const deed = district.stacks[activePlayerId]?.deed;
  const card = propertyCard(action.cardId);
  const target = card ? developmentCost(card) : 0;
  const progress = (deed?.progress ?? 0) + sumTokens(action.tokens);
  return {
    completes: target > 0 && progress >= target,
    progress,
    target,
  };
}

function projectedStackDevelopDeedCompletion(
  action: Extract<GameAction, { type: 'develop-deed' }>,
  stack: DistrictStack
): { completes: boolean; progress: number; target: number } {
  const card = propertyCard(action.cardId);
  const target = card ? developmentCost(card) : 0;
  const progress = (stack.deed?.progress ?? 0) + sumTokens(action.tokens);
  return {
    completes: target > 0 && progress >= target,
    progress,
    target,
  };
}

function districtPlan(state: GameState, activePlayerId: PlayerId): DistrictPlan {
  const opponentId = otherPlayerId(activePlayerId);
  const rows = state.districts.map((district) => {
    const status = districtStatus(district, activePlayerId, opponentId);
    const stack = district.stacks[activePlayerId];
    const eventualMargin = eventualDistrictMargin(district, activePlayerId, opponentId);
    let score = 0;
    if (status.winner === activePlayerId) {
      score += 120 + Math.min(20, status.margin * 2);
    } else if (status.winner === 'Tie') {
      score += 70 + status.activeScore;
    } else {
      score += Math.max(0, 60 + eventualMargin * 6);
    }
    score += stack.developed.length * 2;
    if (stack.deed) {
      score += 5 + stack.deed.progress;
    }
    return { districtId: district.id, score, winner: status.winner };
  });
  rows.sort((left, right) => {
    if (!approximatelyEqual(right.score - left.score, 0)) {
      return right.score - left.score;
    }
    return left.districtId.localeCompare(right.districtId);
  });
  return {
    mainDistrictIds: new Set(
      rows.slice(0, TARGET_CONTROLLED_DISTRICTS).map((row) => row.districtId)
    ),
    controlledCount: rows.filter((row) => row.winner === activePlayerId).length,
  };
}

function controlledDistrictCount(
  state: GameState,
  playerId: PlayerId,
  opponentId: PlayerId
): number {
  return state.districts.filter(
    (district) => districtStatus(district, playerId, opponentId).winner === playerId
  ).length;
}

function districtStatus(
  district: DistrictState,
  activePlayerId: PlayerId,
  opponentId: PlayerId
): DistrictStatus {
  const activeScore = districtScore(district.stacks[activePlayerId]);
  const opponentScore = districtScore(district.stacks[opponentId]);
  return statusFromScores({
    district,
    activeScore,
    opponentScore,
    activePlayerId,
    opponentId,
  });
}

function statusFromScores({
  district,
  activeScore,
  opponentScore,
  activePlayerId,
  opponentId,
}: {
  district: DistrictState;
  activeScore: number;
  opponentScore: number;
  activePlayerId: PlayerId;
  opponentId: PlayerId;
}): DistrictStatus {
  const margin = activeScore - opponentScore;
  let winner: DistrictWinner = 'Tie';
  if (margin > 0) {
    winner = activePlayerId;
  } else if (margin < 0) {
    winner = opponentId;
  }
  return {
    district,
    activeScore,
    opponentScore,
    margin,
    winner,
  };
}

function eventualDistrictMargin(
  district: DistrictState,
  activePlayerId: PlayerId,
  opponentId: PlayerId
): number {
  const activeStack = district.stacks[activePlayerId];
  const opponentScore = districtScore(district.stacks[opponentId]);
  const activeDeveloped = [...activeStack.developed];
  if (activeStack.deed) {
    activeDeveloped.push(activeStack.deed.cardId);
  }
  return scoreStackDeveloped(activeDeveloped) - opponentScore;
}

function opponentEventualMargin(
  district: DistrictState,
  activePlayerId: PlayerId,
  opponentId: PlayerId
): number {
  const activeScore = districtScore(district.stacks[activePlayerId]);
  const opponentStack = district.stacks[opponentId];
  const opponentDeveloped = [...opponentStack.developed];
  if (opponentStack.deed) {
    opponentDeveloped.push(opponentStack.deed.cardId);
  }
  return scoreStackDeveloped(opponentDeveloped) - activeScore;
}

function districtControlValue(
  winner: DistrictWinner,
  activePlayerId: PlayerId
): number {
  if (winner === activePlayerId) {
    return 1;
  }
  if (winner === 'Tie') {
    return 0;
  }
  return -1;
}

function countUnfinishedDeeds(state: GameState, activePlayerId: PlayerId): number {
  return state.districts.filter((district) => {
    const deed = district.stacks[activePlayerId].deed;
    if (!deed) {
      return false;
    }
    const card = propertyCard(deed.cardId);
    return Boolean(card && deed.progress < developmentCost(card));
  }).length;
}

function hasRealisticSuitAccess({
  suit,
  state,
  activePlayerId,
  activePlayer,
  resourcesAfterAction,
  excludedCardId,
}: {
  suit: Suit;
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  resourcesAfterAction: ResourcePool;
  excludedCardId?: CardId;
}): boolean {
  if (resourcesAfterAction[suit] > 0) {
    return true;
  }
  if ((crownSuitCounts(activePlayer)[suit] ?? 0) > 0) {
    return true;
  }
  if (playerHandHasSellableSuit(activePlayer, suit, excludedCardId)) {
    return true;
  }
  return playerBoardHasSuit(state, activePlayerId, suit);
}

function playerHandHasSellableSuit(
  player: PlayerState,
  suit: Suit,
  excludedCardId?: CardId
): boolean {
  return player.hand.some((cardId) => {
    if (cardId === excludedCardId) {
      return false;
    }
    const card = propertyCard(cardId);
    return Boolean(card?.suits.includes(suit));
  });
}

function playerBoardHasSuit(
  state: GameState,
  playerId: PlayerId,
  suit: Suit
): boolean {
  return state.districts.some((district) => {
    const stack = district.stacks[playerId];
    const developedHasSuit = stack.developed.some((cardId) =>
      propertyCard(cardId)?.suits.includes(suit)
    );
    const deedHasSuit = Boolean(
      stack.deed && propertyCard(stack.deed.cardId)?.suits.includes(suit)
    );
    return developedHasSuit || deedHasSuit;
  });
}

function scoreStackDeveloped(cardIds: readonly CardId[]): number {
  const properties = cardIds.map((cardId) => findProperty(cardId)).filter(isDefined);
  const base = properties.reduce((sum, property) => sum + property.rank, 0);
  const aceBonus = properties
    .filter((property) => property.rank === 1 && property.suits.length === 1)
    .reduce((sum, ace) => {
      const suit = ace.suits[0];
      const additionalMatches = properties.filter(
        (property) => property.id !== ace.id && property.suits.includes(suit)
      ).length;
      return sum + additionalMatches;
    }, 0);
  return base + aceBonus;
}

function resourceDeltaForAction(action: GameAction): Partial<Record<Suit, number>> {
  switch (action.type) {
    case 'develop-deed':
      return negateTokens(action.tokens);
    case 'develop-outright':
      return negateTokens(action.payment);
    case 'buy-deed': {
      const card = propertyCard(action.cardId);
      return card ? negateTokens(deedCost(card)) : {};
    }
    case 'trade':
      return {
        [action.give]: -3,
        [action.receive]: 1,
      };
    case 'sell-card':
      return sellCardDelta(action.cardId);
    case 'choose-income-suit':
      return { [action.suit]: 1 };
    case 'end-turn':
      return {};
  }
}

function sellCardDelta(cardId: CardId): Partial<Record<Suit, number>> {
  const card = propertyCard(cardId);
  if (!card) {
    return {};
  }
  if (card.rank === 1) {
    return { [card.suits[0]]: 2 };
  }
  return card.suits.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
    acc[suit] = (acc[suit] ?? 0) + 1;
    return acc;
  }, {});
}

function negateTokens(
  tokens: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const out: Partial<Record<Suit, number>> = {};
  for (const suit of SUITS) {
    const count = tokens[suit] ?? 0;
    if (count !== 0) {
      out[suit] = -count;
    }
  }
  return out;
}

function applyResourceDelta(
  resources: ResourcePool,
  delta: Partial<Record<Suit, number>>
): ResourcePool {
  const out: ResourcePool = { ...resources };
  for (const suit of SUITS) {
    out[suit] = Math.max(0, out[suit] + (delta[suit] ?? 0));
  }
  return out;
}

function taxExposure(resources: ResourcePool): number {
  return SUITS.reduce(
    (total, suit) => total + Math.max(0, resources[suit] - 1),
    0
  );
}

function surplusTokenCount(
  tokens: Partial<Record<Suit, number>>,
  resources: ResourcePool
): number {
  return SUITS.reduce(
    (total, suit) => total + Math.min(tokens[suit] ?? 0, Math.max(0, resources[suit] - 1)),
    0
  );
}

function crownSuitCounts(player: PlayerState): Partial<Record<Suit, number>> {
  return player.crowns.reduce<Partial<Record<Suit, number>>>((acc, cardId) => {
    const card = CARD_BY_ID[cardId];
    if (card?.kind === 'Crown') {
      const suit = card.suits[0];
      acc[suit] = (acc[suit] ?? 0) + 1;
    }
    return acc;
  }, {});
}

function isDistrictInvestment(
  action: GameAction
): action is Extract<
  GameAction,
  { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
> {
  return (
    action.type === 'buy-deed' ||
    action.type === 'develop-deed' ||
    action.type === 'develop-outright'
  );
}

function isLateGame(state: GameState | undefined): boolean {
  if (!state) {
    return false;
  }
  return (
    (state.finalTurnsRemaining ?? 0) > 0 ||
    (state.deck.reshuffles >= 1 && state.deck.draw.length <= LATE_GAME_DRAW_COUNT)
  );
}

function propertyCard(cardId: string): PropertyCard | undefined {
  const card = CARD_BY_ID[cardId as keyof typeof CARD_BY_ID];
  return card && card.kind === 'Property' ? card : undefined;
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}
