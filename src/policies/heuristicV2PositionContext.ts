import { CARD_BY_ID, PROPERTY_CARDS, type CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import { findProperty, placementAllowed } from '../engine/stateHelpers';
import type {
  DistrictState,
  GameState,
  PlayerId,
  PlayerState,
  PropertyCard,
  Suit,
} from '../engine/types';
import { clamp, otherPlayerId, smoothstep } from './policyProjection';

export type SuitValueMap<T> = Record<Suit, T>;

export interface HeuristicV2PositionContext {
  state: GameState;
  playerId: PlayerId;
  player: PlayerState;
  phase: number;
  knownPropertyIdsByPlayer: Map<PlayerId, Set<CardId>>;
  districtScoresById: Map<string, Record<PlayerId, number>>;
  suitAccessByPlayer: Map<PlayerId, SuitValueMap<number>>;
  cardEarningDemandById: Map<CardId, number>;
  districtCardScoringDemandByKey: Map<string, number>;
  bestCardScoringDemandByKey: Map<string, number>;
  placementAllowedByKey: Map<string, boolean>;
}

const EXPECTED_GAME_TURNS = 42;
const SCORING_SCALE = 5;
const CROWN_ACCESS_WEIGHT = 1.0;
const DEVELOPED_ACCESS_WEIGHT = 1.0;
const INCOMPLETE_DEED_ACCESS_WEIGHT = 0.8;
const EARNING_DEMAND_SCALE = 5.8;
const SCORING_DEMAND_SCALE = 1.15;

export function createHeuristicV2PositionContext(
  state: GameState,
  playerId: PlayerId
): HeuristicV2PositionContext {
  return {
    state,
    playerId,
    player: requiredPlayer(state, playerId),
    phase: gamePhaseForTokenDemand(state),
    knownPropertyIdsByPlayer: new Map(),
    districtScoresById: districtScoresById(state),
    suitAccessByPlayer: new Map(),
    cardEarningDemandById: new Map(),
    districtCardScoringDemandByKey: new Map(),
    bestCardScoringDemandByKey: new Map(),
    placementAllowedByKey: new Map(),
  };
}

export function suitAccessBySuitForPlayerV2(
  context: HeuristicV2PositionContext,
  playerId: PlayerId
): SuitValueMap<number> {
  const cached = context.suitAccessByPlayer.get(playerId);
  if (cached) {
    return cached;
  }

  const player = requiredPlayer(context.state, playerId);
  const access = emptySuitValueMap(() => 0);

  for (const crownId of player.crowns) {
    const card = CARD_BY_ID[crownId];
    if (card?.kind === 'Crown') {
      access[card.suits[0]] += CROWN_ACCESS_WEIGHT;
    }
  }

  for (const district of context.state.districts) {
    const stack = district.stacks[playerId];
    for (const cardId of stack.developed) {
      const card = findProperty(cardId);
      if (card) {
        addCardAccess(access, card, DEVELOPED_ACCESS_WEIGHT);
      }
    }
    if (stack.deed) {
      const card = findProperty(stack.deed.cardId);
      if (card) {
        addCardAccess(access, card, INCOMPLETE_DEED_ACCESS_WEIGHT);
      }
    }
  }

  context.suitAccessByPlayer.set(playerId, access);
  return access;
}

export function knownPropertyIdsForPlayerV2(
  context: HeuristicV2PositionContext,
  playerId: PlayerId
): ReadonlySet<CardId> {
  const cached = context.knownPropertyIdsByPlayer.get(playerId);
  if (cached) {
    return cached;
  }
  const known = informationSafeKnownPropertyIds(context.state, playerId);
  context.knownPropertyIdsByPlayer.set(playerId, known);
  return known;
}

export function cardEarningDemandV2(
  context: HeuristicV2PositionContext,
  card: PropertyCard
): number {
  const cached = context.cardEarningDemandById.get(card.id);
  if (cached !== undefined) {
    return cached;
  }
  const rankQuality = card.rank / 9;
  const phaseMultiplier = 1 - 0.7 * smoothstep(context.phase);
  const value =
    incomeProbabilityForRankV2(card.rank) *
    (0.35 + rankQuality) *
    EARNING_DEMAND_SCALE *
    phaseMultiplier;
  context.cardEarningDemandById.set(card.id, value);
  return value;
}

export function bestCardScoringDemandV2(
  context: HeuristicV2PositionContext,
  playerId: PlayerId,
  card: PropertyCard
): number {
  const key = `${playerId}:${card.id}`;
  const cached = context.bestCardScoringDemandByKey.get(key);
  if (cached !== undefined) {
    return cached;
  }

  let best = 0;
  for (const district of context.state.districts) {
    if (!placementAllowedCached(context, playerId, district, card)) {
      continue;
    }
    best = Math.max(
      best,
      cardScoringDemandInDistrictV2(context, playerId, district, card)
    );
  }
  context.bestCardScoringDemandByKey.set(key, best);
  return best;
}

export function cardScoringDemandInDistrictV2(
  context: HeuristicV2PositionContext,
  playerId: PlayerId,
  district: DistrictState,
  card: PropertyCard
): number {
  const key = `${playerId}:${district.id}:${card.id}`;
  const cached = context.districtCardScoringDemandByKey.get(key);
  if (cached !== undefined) {
    return cached;
  }

  const opponentId = otherPlayerId(playerId);
  const stack = district.stacks[playerId];
  const scores = context.districtScoresById.get(district.id);
  const currentOwnScore = scores?.[playerId] ?? districtScore(stack);
  const opponentScore =
    scores?.[opponentId] ?? districtScore(district.stacks[opponentId]);
  const completedOwnScore = districtScore({
    developed: [...stack.developed, card.id],
  });
  const beforeMargin = currentOwnScore - opponentScore;
  const afterMargin = completedOwnScore - opponentScore;
  const saturatedDelta =
    Math.tanh(afterMargin / SCORING_SCALE) -
    Math.tanh(beforeMargin / SCORING_SCALE);
  const controlBonus =
    beforeMargin <= 0 && afterMargin > 0
      ? 0.45
      : beforeMargin < 0 && afterMargin === 0
        ? 0.2
        : beforeMargin > 0 && afterMargin > beforeMargin
          ? 0.08
          : 0;
  const value =
    Math.max(0, saturatedDelta + controlBonus) * SCORING_DEMAND_SCALE;
  context.districtCardScoringDemandByKey.set(key, value);
  return value;
}

export function placementAllowedCached(
  context: HeuristicV2PositionContext,
  playerId: PlayerId,
  district: DistrictState,
  card: PropertyCard
): boolean {
  const key = `${playerId}:${district.id}:${card.id}`;
  const cached = context.placementAllowedByKey.get(key);
  if (cached !== undefined) {
    return cached;
  }
  const allowed = placementAllowed(card, district, playerId);
  context.placementAllowedByKey.set(key, allowed);
  return allowed;
}

export function propertyCardsUnknownToPlayerV2(
  context: HeuristicV2PositionContext,
  playerId: PlayerId
): readonly PropertyCard[] {
  const known = knownPropertyIdsForPlayerV2(context, playerId);
  return PROPERTY_CARDS.filter((card) => !known.has(card.id));
}

export function incomeProbabilityForRankV2(
  rank: PropertyCard['rank']
): number {
  if (rank === 1) {
    return 0.01;
  }
  return (2 * rank - 1) / 100;
}

function addCardAccess(
  access: SuitValueMap<number>,
  card: PropertyCard,
  sourceWeight: number
): void {
  const value = incomeProbabilityForRankV2(card.rank) * sourceWeight;
  for (const suit of card.suits) {
    access[suit] += value;
  }
}

function districtScoresById(
  state: GameState
): Map<string, Record<PlayerId, number>> {
  const scores = new Map<string, Record<PlayerId, number>>();
  for (const district of state.districts) {
    scores.set(district.id, {
      PlayerA: districtScore(district.stacks.PlayerA),
      PlayerB: districtScore(district.stacks.PlayerB),
    });
  }
  return scores;
}

function informationSafeKnownPropertyIds(
  state: GameState,
  playerId: PlayerId
): Set<CardId> {
  const known = new Set<CardId>();
  const player = requiredPlayer(state, playerId);
  for (const cardId of player.hand) {
    known.add(cardId);
  }
  for (const cardId of state.deck.discard) {
    const card = findProperty(cardId);
    if (card) {
      known.add(card.id);
    }
  }
  for (const district of state.districts) {
    for (const candidatePlayerId of ['PlayerA', 'PlayerB'] as const) {
      const stack = district.stacks[candidatePlayerId];
      for (const cardId of stack.developed) {
        known.add(cardId);
      }
      if (stack.deed) {
        known.add(stack.deed.cardId);
      }
    }
  }
  return known;
}

function requiredPlayer(state: GameState, playerId: PlayerId): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Heuristic v2 context missing player ${playerId}.`);
  }
  return player;
}

function gamePhaseForTokenDemand(state: GameState): number {
  if ((state.finalTurnsRemaining ?? 0) > 0) {
    return 1;
  }
  return clamp(state.turn / EXPECTED_GAME_TURNS, 0, 1);
}

function emptySuitValueMap<T>(create: (suit: Suit) => T): SuitValueMap<T> {
  return {
    Moons: create('Moons'),
    Suns: create('Suns'),
    Waves: create('Waves'),
    Leaves: create('Leaves'),
    Wyrms: create('Wyrms'),
    Knots: create('Knots'),
  };
}
