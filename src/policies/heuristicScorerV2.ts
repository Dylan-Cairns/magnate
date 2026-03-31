import { toKeyedActions, type KeyedAction } from '../engine/actionSurface';
import type { CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import { developmentCost, findProperty, SUITS } from '../engine/stateHelpers';
import type {
  DistrictStack,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PlayerView,
  PropertyCard,
} from '../engine/types';
import {
  clamp,
  isDistrictAction,
  otherPlayerId,
  projectDistrictAction,
  projectStateDistrictAction,
  smoothstep,
} from './policyProjection';
import {
  createHeuristicV2PositionContext,
  suitAccessBySuitForPlayerV2,
  type HeuristicV2PositionContext,
  type SuitValueMap,
} from './heuristicV2PositionContext';
import {
  createTokenValueContextV2,
  tokenDeltaForActionV2,
  type TokenValueContextV2,
} from './tokenValueV2';

export interface HeuristicV2SelectionContext {
  state: GameState;
  view: PlayerView;
  legalActions: readonly GameAction[];
}

export interface HeuristicV2ScoredAction extends KeyedAction {
  score: number;
  prior: number;
  rank: number;
}

type HeuristicV2EvaluationContext = Partial<
  Pick<HeuristicV2SelectionContext, 'state' | 'view' | 'legalActions'>
>;

interface ResolvedHeuristicV2Context {
  state: GameState;
  activePlayerId: PlayerId;
  activePlayer: PlayerState;
  positionContext: HeuristicV2PositionContext;
  tokenContext: TokenValueContextV2;
  earningPotential: number;
}

interface CachedHeuristicV2Score extends KeyedAction {
  score: number;
}

const EXPECTED_GAME_TURNS = 42;
const SCORING_SCALE = 5;
const SMALL_ACTION_BASELINE = 0.05;
const TOKEN_VALUE_WEIGHT = 0.02;

export function selectHeuristicV2Action(
  context: HeuristicV2SelectionContext
): GameAction | undefined {
  const ranked = rankHeuristicV2Actions(context.legalActions, context);
  return ranked[0]?.action;
}

export function rankHeuristicV2Actions(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext = {}
): KeyedAction[] {
  return scoreKeyedHeuristicV2Actions(candidateActions, context);
}

export function scoreHeuristicV2Actions(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext = {}
): HeuristicV2ScoredAction[] {
  const scored = scoreKeyedHeuristicV2Actions(candidateActions, context);
  const priors = heuristicV2PriorsByScoredActions(scored);
  return scored.map((candidate, index) => ({
      ...candidate,
      prior: priors.get(candidate.actionKey) ?? 0,
      rank: index,
    }));
}

export function heuristicV2PriorsByKey(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext = {}
): Map<string, number> {
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length === 0) {
    return new Map<string, number>();
  }

  const resolved = resolveContext(context);
  const scores = keyed.map((candidate) =>
    scoreHeuristicV2ActionWithContext(candidate.action, resolved)
  );
  return heuristicV2PriorsByScoredValues(keyed, scores);
}

function heuristicV2PriorsByScoredActions(
  scored: readonly CachedHeuristicV2Score[]
): Map<string, number> {
  return heuristicV2PriorsByScoredValues(
    scored,
    scored.map((candidate) => candidate.score)
  );
}

function heuristicV2PriorsByScoredValues(
  keyed: readonly KeyedAction[],
  scores: readonly number[]
): Map<string, number> {
  const maxScore = Math.max(...scores);
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

export function scoreHeuristicV2Action(
  action: GameAction,
  context: HeuristicV2EvaluationContext = {}
): number {
  return scoreHeuristicV2ActionWithContext(action, resolveContext(context));
}

function scoreHeuristicV2ActionWithContext(
  action: GameAction,
  resolved: ResolvedHeuristicV2Context | undefined
): number {
  if (!resolved) {
    return actionBaseline(action);
  }

  const { state, activePlayerId } = resolved;
  const phase = gamePhase(state);
  const scoringWeight = scoringWeightForPhase(phase);
  const earningWeight = earningWeightForPhase(phase);

  return (
    scoringWeight * scoringDeltaForAction(action, state, activePlayerId) +
    earningWeight * earningDeltaForAction(action, resolved) +
    TOKEN_VALUE_WEIGHT *
      tokenDeltaForActionV2(action, state, activePlayerId, resolved.tokenContext) +
    actionBaseline(action)
  );
}

export function earningPotentialValueForPlayerV2(
  state: GameState,
  playerId: PlayerId
): number {
  const player = state.players.find((entry) => entry.id === playerId);
  if (!player) {
    return 0;
  }
  const access = suitAccessBySuitForPlayerV2(
    createHeuristicV2PositionContext(state, playerId),
    playerId
  );

  return SUITS.reduce((total, suit) => total + Math.log1p(access[suit]), 0);
}

function earningPotentialValueFromAccess(
  access: SuitValueMap<number>
): number {
  return SUITS.reduce((total, suit) => total + Math.log1p(access[suit]), 0);
}

function compareKeyedActionsByHeuristicV2(
  left: KeyedAction,
  right: KeyedAction,
  scoreByKey: ReadonlyMap<string, number>
): number {
  const scoreDelta =
    (scoreByKey.get(right.actionKey) ?? 0) -
    (scoreByKey.get(left.actionKey) ?? 0);
  if (!approximatelyEqual(scoreDelta, 0)) {
    return scoreDelta > 0 ? 1 : -1;
  }
  return left.actionKey.localeCompare(right.actionKey);
}

function scoreKeyedHeuristicV2Actions(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext
): CachedHeuristicV2Score[] {
  const keyed = toKeyedActions(candidateActions);
  const resolved = resolveContext(context);
  const scored = keyed.map((candidate) => ({
    ...candidate,
    score: scoreHeuristicV2ActionWithContext(candidate.action, resolved),
  }));
  const scoreByKey = new Map(
    scored.map((candidate) => [candidate.actionKey, candidate.score])
  );
  return [...scored].sort((left, right) =>
    compareKeyedActionsByHeuristicV2(left, right, scoreByKey)
  );
}

function resolveContext(context: HeuristicV2EvaluationContext):
  | ResolvedHeuristicV2Context
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
  const activePlayer = state.players.find(
    (player) => player.id === activePlayerId
  );
  if (!activePlayer) {
    return undefined;
  }
  const positionContext = createHeuristicV2PositionContext(
    state,
    activePlayerId
  );
  const access = suitAccessBySuitForPlayerV2(positionContext, activePlayerId);
  return {
    state,
    activePlayerId,
    activePlayer,
    positionContext,
    tokenContext: createTokenValueContextV2(
      state,
      activePlayerId,
      activePlayer.resources,
      positionContext
    ),
    earningPotential: earningPotentialValueFromAccess(access),
  };
}

function scoringDeltaForAction(
  action: GameAction,
  state: GameState,
  playerId: PlayerId
): number {
  if (!isDistrictAction(action)) {
    return 0;
  }
  const district = state.districts.find(
    (candidate) => candidate.id === action.districtId
  );
  if (!district) {
    return 0;
  }
  const opponentId = otherPlayerId(playerId);
  const beforeMargin = potentialDistrictMargin(district, playerId, opponentId);
  const afterDistrict = projectDistrictAction(district, action, playerId);
  const afterMargin = potentialDistrictMargin(
    afterDistrict,
    playerId,
    opponentId
  );
  return (
    Math.tanh(afterMargin / SCORING_SCALE) -
    Math.tanh(beforeMargin / SCORING_SCALE)
  );
}

function earningDeltaForAction(
  action: GameAction,
  resolved: ResolvedHeuristicV2Context
): number {
  if (!isDistrictAction(action)) {
    return 0;
  }
  const { state, activePlayerId: playerId, earningPotential } = resolved;
  const afterState = projectStateDistrictAction(action, state, playerId);
  const afterAccess = suitAccessBySuitForPlayerV2(
    createHeuristicV2PositionContext(afterState, playerId),
    playerId
  );
  const after = earningPotentialValueFromAccess(afterAccess);
  return after - earningPotential;
}

function potentialDistrictMargin(
  district: DistrictState,
  playerId: PlayerId,
  opponentId: PlayerId
): number {
  return (
    potentialStackScore(district.stacks[playerId]) -
    potentialStackScore(district.stacks[opponentId])
  );
}

function potentialStackScore(stack: DistrictStack): number {
  const developedScore = districtScore(stack);
  if (!stack.deed) {
    return developedScore;
  }
  const card = propertyCard(stack.deed.cardId);
  if (!card) {
    return developedScore;
  }
  const target = developmentCost(card);
  if (target <= 0) {
    return developedScore;
  }
  const completionRatio = clamp(stack.deed.progress / target, 0, 1);
  const completionWeight = completionRatio * completionRatio;
  return (
    developedScore +
    incrementalDevelopedScore(stack.developed, card.id) * completionWeight
  );
}

function incrementalDevelopedScore(
  developed: readonly CardId[],
  cardId: CardId
): number {
  const before = districtScore({ developed: [...developed] });
  const after = districtScore({ developed: [...developed, cardId] });
  return Math.max(0, after - before);
}

function scoringWeightForPhase(phase: number): number {
  const t = smoothstep(phase);
  return 7 + 17 * t;
}

function earningWeightForPhase(phase: number): number {
  const t = smoothstep(phase);
  return 16 - 10 * t;
}

function gamePhase(state: GameState): number {
  return clamp(state.turn / EXPECTED_GAME_TURNS, 0, 1);
}

function actionBaseline(action: GameAction): number {
  return action.type === 'end-turn' ? 0 : SMALL_ACTION_BASELINE;
}

function propertyCard(cardId: string): PropertyCard | undefined {
  return findProperty(cardId as CardId);
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}
