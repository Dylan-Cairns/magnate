import { toKeyedActions, type KeyedAction } from '../engine/actionSurface';
import { CARD_BY_ID, type CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import { developmentCost, findProperty, SUITS, sumTokens } from '../engine/stateHelpers';
import type {
  DistrictStack,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PlayerView,
  PropertyCard,
  Suit,
} from '../engine/types';

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

type AccessBySuit = Record<Suit, number>;

const EXPECTED_GAME_TURNS = 42;
const SCORING_SCALE = 5;
const DEVELOPED_ACCESS_WEIGHT = 1.0;
const INCOMPLETE_DEED_ACCESS_WEIGHT = 0.8;
const CROWN_ACCESS_BASELINE = 1.0;
const SMALL_ACTION_BASELINE = 0.05;

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
  const keyed = toKeyedActions(candidateActions);
  return [...keyed].sort((left, right) =>
    compareKeyedActionsByHeuristicV2(left, right, context)
  );
}

export function scoreHeuristicV2Actions(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext = {}
): HeuristicV2ScoredAction[] {
  const priors = heuristicV2PriorsByKey(candidateActions, context);
  return rankHeuristicV2Actions(candidateActions, context).map(
    (candidate, index) => ({
      ...candidate,
      score: scoreHeuristicV2Action(candidate.action, context),
      prior: priors.get(candidate.actionKey) ?? 0,
      rank: index,
    })
  );
}

export function heuristicV2PriorsByKey(
  candidateActions: readonly GameAction[],
  context: HeuristicV2EvaluationContext = {}
): Map<string, number> {
  const keyed = toKeyedActions(candidateActions);
  if (keyed.length === 0) {
    return new Map<string, number>();
  }

  const scores = keyed.map((candidate) =>
    scoreHeuristicV2Action(candidate.action, context)
  );
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
  const resolved = resolveContext(context);
  if (!resolved) {
    return actionBaseline(action);
  }

  const { state, activePlayerId } = resolved;
  const phase = gamePhase(state);
  const scoringWeight = scoringWeightForPhase(phase);
  const earningWeight = earningWeightForPhase(phase);

  return (
    scoringWeight * scoringDeltaForAction(action, state, activePlayerId) +
    earningWeight * earningDeltaForAction(action, state, activePlayerId) +
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
  const access = emptyAccessBySuit();

  for (const crownId of player.crowns) {
    const card = CARD_BY_ID[crownId];
    if (card?.kind !== 'Crown') {
      continue;
    }
    access[card.suits[0]] += CROWN_ACCESS_BASELINE;
  }

  for (const district of state.districts) {
    addStackAccess(access, district.stacks[playerId]);
  }

  return SUITS.reduce((total, suit) => total + Math.log1p(access[suit]), 0);
}

function compareKeyedActionsByHeuristicV2(
  left: KeyedAction,
  right: KeyedAction,
  context: HeuristicV2EvaluationContext
): number {
  const scoreDelta =
    scoreHeuristicV2Action(right.action, context) -
    scoreHeuristicV2Action(left.action, context);
  if (!approximatelyEqual(scoreDelta, 0)) {
    return scoreDelta > 0 ? 1 : -1;
  }
  return left.actionKey.localeCompare(right.actionKey);
}

function resolveContext(context: HeuristicV2EvaluationContext):
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
  const activePlayer = state.players.find(
    (player) => player.id === activePlayerId
  );
  if (!activePlayer) {
    return undefined;
  }
  return {
    state,
    activePlayerId,
    activePlayer,
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
  state: GameState,
  playerId: PlayerId
): number {
  const before = earningPotentialValueForPlayerV2(state, playerId);
  const afterState = projectStateForAccess(action, state, playerId);
  const after = earningPotentialValueForPlayerV2(afterState, playerId);
  return after - before;
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

function projectStateForAccess(
  action: GameAction,
  state: GameState,
  playerId: PlayerId
): GameState {
  if (!isDistrictAction(action)) {
    return state;
  }
  return {
    ...state,
    districts: state.districts.map((district) =>
      district.id === action.districtId
        ? projectDistrictAction(district, action, playerId)
        : district
    ),
  };
}

function projectDistrictAction(
  district: DistrictState,
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >,
  playerId: PlayerId
): DistrictState {
  const currentStack = district.stacks[playerId];
  const projectedStack = projectStackAction(currentStack, action);
  return {
    ...district,
    stacks: {
      ...district.stacks,
      [playerId]: projectedStack,
    },
  };
}

function projectStackAction(
  stack: DistrictStack,
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >
): DistrictStack {
  if (action.type === 'develop-outright') {
    return {
      ...stack,
      developed: [...stack.developed, action.cardId],
    };
  }
  if (action.type === 'buy-deed') {
    return {
      ...stack,
      deed: {
        cardId: action.cardId,
        progress: 0,
        tokens: {},
      },
    };
  }

  const deed = stack.deed;
  if (!deed) {
    return stack;
  }
  const card = propertyCard(deed.cardId);
  const progress = deed.progress + sumTokens(action.tokens);
  const target = card ? developmentCost(card) : 0;
  if (target > 0 && progress >= target) {
    return {
      ...stack,
      developed: [...stack.developed, deed.cardId],
      deed: undefined,
    };
  }
  return {
    ...stack,
    deed: {
      ...deed,
      progress,
      tokens: mergeTokens(deed.tokens, action.tokens),
    },
  };
}

function addStackAccess(access: AccessBySuit, stack: DistrictStack): void {
  for (const cardId of stack.developed) {
    const card = propertyCard(cardId);
    if (!card) {
      continue;
    }
    addCardAccess(access, card, DEVELOPED_ACCESS_WEIGHT);
  }
  if (stack.deed) {
    const card = propertyCard(stack.deed.cardId);
    if (card) {
      addCardAccess(access, card, INCOMPLETE_DEED_ACCESS_WEIGHT);
    }
  }
}

function addCardAccess(
  access: AccessBySuit,
  card: PropertyCard,
  sourceWeight: number
): void {
  const value = incomeProbabilityForRank(card.rank) * sourceWeight;
  for (const suit of card.suits) {
    access[suit] += value;
  }
}

function incomeProbabilityForRank(rank: PropertyCard['rank']): number {
  if (rank === 1) {
    return 0.01;
  }
  return (2 * rank - 1) / 100;
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

function emptyAccessBySuit(): AccessBySuit {
  return {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
}

function mergeTokens(
  existing: Partial<Record<Suit, number>>,
  added: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const out: Partial<Record<Suit, number>> = { ...existing };
  for (const suit of SUITS) {
    const count = added[suit] ?? 0;
    if (count > 0) {
      out[suit] = (out[suit] ?? 0) + count;
    }
  }
  return out;
}

function propertyCard(cardId: string): PropertyCard | undefined {
  return findProperty(cardId as CardId);
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function isDistrictAction(
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

function smoothstep(value: number): number {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function approximatelyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-9;
}
