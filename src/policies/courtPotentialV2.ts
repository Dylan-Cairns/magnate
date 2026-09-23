import type { CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import {
  applyDelta,
  deedCost,
  developmentCost,
  findDevelopableCard,
  sumTokens,
} from '../engine/stateHelpers';
import type {
  CourtCard,
  DevelopableCard,
  DistrictStack,
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  ResourcePool,
  Suit,
} from '../engine/types';
import {
  suitAccessBySuitForPlayerV2,
  type HeuristicV2PositionContext,
} from './heuristicV2PositionContext';
import { otherPlayerId, projectDistrictAction } from './policyProjection';

/**
 * Heuristic v2 court valuation.
 *
 * Courts are the only developable cards with zero income potential, so the
 * generic earning term cannot price an incomplete court. This module owns the
 * incomplete-court states (buy-deed and develop-deed on a Court) and values
 * them as a feasibility-discounted district swing. Completed courts stay with
 * the generic developed-card scoring path, and standard-ruleset games never
 * reach this code because the standard deck has no Courts.
 */

export const COURT_FEASIBILITY_HAIRCUT = 0.7;
export const COURT_HORIZON_CAP = 24;
const EXPECTED_GAME_TURNS = 42;
const SCORING_SCALE = 5;

export interface CourtActionBreakdown {
  readonly actionType: 'buy-deed' | 'develop-deed';
  readonly cardId: CardId;
  readonly completes: boolean;
  readonly swing: number;
  readonly entryCost: number;
  readonly turnsLeft: number;
  readonly remainingCost: number;
  readonly availableTokens: number;
  readonly feasibility: number;
  readonly delta: number;
}

interface CourtDeedValuation {
  readonly swing: number;
  readonly feasibility: number;
  readonly remainingCost: number;
  readonly availableTokens: number;
  readonly turnsLeft: number;
  readonly value: number;
}

export function isCourtCard(
  card: DevelopableCard | undefined
): card is CourtCard {
  return card?.kind === 'Court';
}

/**
 * State-based value of an incomplete Court for the leaf evaluator: the same
 * feasibility-discounted completion swing the action term uses, without any
 * action spend. Returns undefined for non-Court, complete, or standard states.
 */
export function courtPotentialValueForPlayerV2(
  state: GameState,
  playerId: PlayerId,
  district: DistrictState,
  positionContext: HeuristicV2PositionContext,
  courtValueScale: number
): number | undefined {
  if (state.ruleset !== 'extended') {
    return undefined;
  }
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    return undefined;
  }
  const valuation = courtDeedValuation(
    district,
    district.stacks[playerId],
    player.resources,
    playerId,
    positionContext
  );
  if (!valuation) {
    return undefined;
  }
  return valuation.value * courtValueScale;
}

export function courtActionBreakdown(
  action: GameAction,
  state: GameState,
  playerId: PlayerId,
  positionContext: HeuristicV2PositionContext,
  courtValueScale: number
): CourtActionBreakdown | undefined {
  if (action.type !== 'buy-deed' && action.type !== 'develop-deed') {
    return undefined;
  }
  if (state.ruleset !== 'extended') {
    return undefined;
  }
  const card = findDevelopableCard(action.cardId);
  if (!isCourtCard(card)) {
    return undefined;
  }
  const district = state.districts.find(
    (candidate) => candidate.id === action.districtId
  );
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!district || !player) {
    return undefined;
  }

  const spend = action.type === 'buy-deed' ? deedCost(card) : action.tokens;
  const before = courtDeedValuation(
    district,
    district.stacks[playerId],
    player.resources,
    playerId,
    positionContext
  );
  const afterDistrict = projectDistrictAction(district, action, playerId);
  const after = courtDeedValuation(
    afterDistrict,
    afterDistrict.stacks[playerId],
    applyDelta(player.resources, negateTokens(spend)),
    playerId,
    positionContext
  );
  const reference = after ?? before;
  if (!reference) {
    return undefined;
  }

  return {
    actionType: action.type,
    cardId: action.cardId,
    completes: after === undefined,
    swing: reference.swing,
    entryCost: action.type === 'buy-deed' ? sumTokens(spend) : 0,
    turnsLeft: reference.turnsLeft,
    remainingCost: reference.remainingCost,
    availableTokens: reference.availableTokens,
    feasibility: reference.feasibility,
    delta: ((after?.value ?? 0) - (before?.value ?? 0)) * courtValueScale,
  };
}

function courtDeedValuation(
  district: DistrictState,
  stack: DistrictStack,
  resources: ResourcePool,
  playerId: PlayerId,
  positionContext: HeuristicV2PositionContext
): CourtDeedValuation | undefined {
  const deed = stack.deed;
  if (!deed) {
    return undefined;
  }
  const card = findDevelopableCard(deed.cardId);
  if (!isCourtCard(card)) {
    return undefined;
  }
  const target = developmentCost(card);
  const remainingCost = target - deed.progress;
  if (remainingCost <= 0) {
    return undefined;
  }

  const access = suitAccessBySuitForPlayerV2(positionContext, playerId);
  let stock = 0;
  let flow = 0;
  for (const suit of card.suits) {
    stock += resources[suit];
    flow += access[suit];
  }

  const turnsLeft = turnsLeftForState(positionContext.state);
  const availableTokens =
    stock + flow * turnsLeft * COURT_FEASIBILITY_HAIRCUT;
  const feasibility =
    availableTokens <= 0
      ? 0
      : availableTokens / (availableTokens + remainingCost);
  const swing = courtCompletionSwing(district, playerId, card);

  return {
    swing,
    feasibility,
    remainingCost,
    availableTokens,
    turnsLeft,
    value: swing * feasibility,
  };
}

function courtCompletionSwing(
  district: DistrictState,
  playerId: PlayerId,
  card: DevelopableCard
): number {
  const own = district.stacks[playerId];
  const opponent = district.stacks[otherPlayerId(playerId)];
  const ownScore = districtScore(own);
  const opponentScore = districtScore(opponent);
  const increment = Math.max(
    0,
    districtScore({ developed: [...own.developed, card.id] }) - ownScore
  );
  const before = Math.tanh((ownScore - opponentScore) / SCORING_SCALE);
  const after = Math.tanh(
    (ownScore + increment - opponentScore) / SCORING_SCALE
  );
  return after - before;
}

function turnsLeftForState(state: GameState): number {
  const exact = state.finalTurnsRemaining;
  const raw =
    exact !== undefined
      ? exact
      : Math.max(0, EXPECTED_GAME_TURNS - state.turn);
  return Math.min(COURT_HORIZON_CAP, Math.max(0, raw));
}

function negateTokens(
  tokens: Partial<Record<Suit, number>>
): Partial<Record<Suit, number>> {
  const out: Partial<Record<Suit, number>> = {};
  for (const suit of Object.keys(tokens) as Suit[]) {
    const count = tokens[suit] ?? 0;
    if (count !== 0) {
      out[suit] = -count;
    }
  }
  return out;
}
