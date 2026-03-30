import { CARD_BY_ID, PROPERTY_CARDS, type CardId } from '../engine/cards';
import { districtScore } from '../engine/scoring';
import {
  applyDelta,
  deedCost,
  developmentCost,
  findProperty,
  mergeTokens,
  placementAllowed,
  SUITS,
  sumTokens,
} from '../engine/stateHelpers';
import type {
  DistrictState,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PropertyCard,
  ResourcePool,
  Suit,
} from '../engine/types';

export type SuitValueMap<T> = Record<Suit, T>;

export interface SuitTokenValueV2 {
  suit: Suit;
  earningDemand: number;
  scoringDemand: number;
  rawDemand: number;
  access: number;
  replaceabilityMultiplier: number;
  value: number;
}

export interface ResourceBankValueBreakdownV2 {
  directValue: number;
  tradeLiquidityValue: number;
  totalValue: number;
  suitValues: SuitValueMap<SuitTokenValueV2>;
  tradeLiquidityBySuit: SuitValueMap<number>;
}

type DemandBySuit = SuitValueMap<{
  earningDemand: number;
  scoringDemand: number;
}>;

const EXPECTED_GAME_TURNS = 42;
const SCORING_SCALE = 5;

const CROWN_ACCESS_WEIGHT = 1.0;
const DEVELOPED_ACCESS_WEIGHT = 1.0;
const INCOMPLETE_DEED_ACCESS_WEIGHT = 0.8;
const HAND_DEMAND_WEIGHT = 0.85;
const INCOMPLETE_DEED_DEMAND_WEIGHT = 1.35;
const UNKNOWN_POOL_DEMAND_WEIGHT = 0.035;

const EARNING_DEMAND_SCALE = 5.8;
const SCORING_DEMAND_SCALE = 1.15;
const DIRECT_VALUE_FLOOR = 0.02;
const TRADE_SET_WEIGHT = 0.28;
const TRADE_REMAINDER_TWO_WEIGHT = 0.12;
const TRADE_REMAINDER_ONE_WEIGHT = 0.03;

export function contextualSuitTokenValuesV2(
  state: GameState,
  playerId: PlayerId
): SuitValueMap<SuitTokenValueV2> {
  const player = requiredPlayer(state, playerId);
  const access = suitAccessBySuitV2(state, playerId);
  const demand = demandBySuitForPlayer(state, player);
  const values = emptySuitValueMap<SuitTokenValueV2>((suit) => {
    const earningDemand = demand[suit].earningDemand;
    const scoringDemand = demand[suit].scoringDemand;
    const rawDemand = earningDemand + scoringDemand;
    const replaceabilityMultiplier = replaceabilityMultiplierForAccess(
      access[suit]
    );
    return {
      suit,
      earningDemand,
      scoringDemand,
      rawDemand,
      access: access[suit],
      replaceabilityMultiplier,
      value:
        rawDemand <= 0
          ? 0
          : Math.max(
              DIRECT_VALUE_FLOOR,
              rawDemand * replaceabilityMultiplier
            ),
    };
  });

  return values;
}

export function suitAccessBySuitV2(
  state: GameState,
  playerId: PlayerId
): SuitValueMap<number> {
  const player = requiredPlayer(state, playerId);
  const access = emptySuitValueMap(() => 0);

  for (const crownId of player.crowns) {
    const card = CARD_BY_ID[crownId];
    if (card?.kind === 'Crown') {
      access[card.suits[0]] += CROWN_ACCESS_WEIGHT;
    }
  }

  for (const district of state.districts) {
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

  return access;
}

export function resourceBankValueV2(
  state: GameState,
  playerId: PlayerId,
  resources = requiredPlayer(state, playerId).resources
): number {
  return resourceBankValueBreakdownV2(state, playerId, resources).totalValue;
}

export function resourceBankValueBreakdownV2(
  state: GameState,
  playerId: PlayerId,
  resources = requiredPlayer(state, playerId).resources
): ResourceBankValueBreakdownV2 {
  const suitValues = contextualSuitTokenValuesV2(state, playerId);
  let directValue = 0;

  for (const suit of SUITS) {
    directValue += directResourceValueForSuit(
      Math.max(0, resources[suit]),
      suitValues[suit].value
    );
  }

  const tradeLiquidityBySuit = emptySuitValueMap((suit) =>
    tradeLiquidityValueForSuitV2(suit, resources, suitValues)
  );
  const tradeLiquidityValue = SUITS.reduce(
    (total, suit) => total + tradeLiquidityBySuit[suit],
    0
  );

  return {
    directValue,
    tradeLiquidityValue,
    totalValue: directValue + tradeLiquidityValue,
    suitValues,
    tradeLiquidityBySuit,
  };
}

export function tokenDeltaForActionV2(
  action: GameAction,
  state: GameState,
  playerId: PlayerId
): number {
  const player = requiredPlayer(state, playerId);
  const before = player.resources;
  const after = applyDelta(before, resourceDeltaForActionV2(action));
  const afterState = projectStateForTokenValueV2(action, state, playerId);
  return (
    resourceBankValueV2(afterState, playerId, after) -
    resourceBankValueV2(state, playerId, before)
  );
}

export function projectStateForTokenValueV2(
  action: GameAction,
  state: GameState,
  playerId: PlayerId
): GameState {
  const player = requiredPlayer(state, playerId);
  const updatedPlayer = projectPlayerForTokenValue(action, player);
  const withPlayer = replacePlayer(state, playerId, updatedPlayer);
  const withDistricts = isDistrictAction(action)
    ? {
        ...withPlayer,
        districts: withPlayer.districts.map((district) =>
          district.id === action.districtId
            ? projectDistrictForTokenValue(district, action, playerId)
            : district
        ),
      }
    : withPlayer;

  if (action.type !== 'sell-card') {
    return withDistricts;
  }

  return {
    ...withDistricts,
    deck: {
      ...withDistricts.deck,
      discard: [action.cardId, ...withDistricts.deck.discard],
    },
  };
}

export function resourceDeltaForActionV2(
  action: GameAction
): Partial<Record<Suit, number>> {
  switch (action.type) {
    case 'buy-deed': {
      const card = findProperty(action.cardId);
      return card ? negateTokens(deedCost(card)) : {};
    }
    case 'choose-income-suit':
      return { [action.suit]: 1 };
    case 'develop-deed':
      return negateTokens(action.tokens);
    case 'develop-outright':
      return negateTokens(action.payment);
    case 'end-turn':
      return {};
    case 'sell-card':
      return sellCardDelta(action.cardId);
    case 'trade':
      return {
        [action.give]: -3,
        [action.receive]: 1,
      };
  }
}

export function directResourceValueForSuit(
  count: number,
  contextualSuitValue: number
): number {
  let value = 0;
  for (let index = 1; index <= Math.max(0, count); index += 1) {
    value += contextualSuitValue * marginalTokenMultiplier(index);
  }
  return value;
}

export function tradeLiquidityValueForSuitV2(
  suit: Suit,
  resources: ResourcePool,
  suitValues: SuitValueMap<SuitTokenValueV2>
): number {
  const count = Math.max(0, resources[suit]);
  if (count < 2) {
    return 0;
  }

  const bestTargetValue = bestTradeTargetValue(suit, resources, suitValues);
  if (bestTargetValue <= 0) {
    return 0;
  }

  const completedSets = Math.floor(count / 3);
  const remainder = count % 3;
  const completedSetBonus =
    completedSets * TRADE_SET_WEIGHT * bestTargetValue;
  const remainderBonus =
    remainder === 2
      ? TRADE_REMAINDER_TWO_WEIGHT * bestTargetValue
      : remainder === 1 && completedSets > 0
        ? TRADE_REMAINDER_ONE_WEIGHT * bestTargetValue
        : 0;
  const sourceExpendability = tradeSourceExpendability(
    suitValues[suit].value,
    bestTargetValue
  );

  return (completedSetBonus + remainderBonus) * sourceExpendability;
}

function demandBySuitForPlayer(
  state: GameState,
  player: PlayerState
): DemandBySuit {
  const demand = emptySuitValueMap(() => ({
    earningDemand: 0,
    scoringDemand: 0,
  }));
  const phase = gamePhase(state);

  addIncompleteDeedDemand(demand, state, player.id, phase);
  addHandDemand(demand, state, player, phase);
  addUnknownPoolDemand(demand, state, player, phase);

  return demand;
}

function addIncompleteDeedDemand(
  demand: DemandBySuit,
  state: GameState,
  playerId: PlayerId,
  phase: number
): void {
  for (const district of state.districts) {
    const deed = district.stacks[playerId].deed;
    if (!deed) {
      continue;
    }
    const card = findProperty(deed.cardId);
    if (!card) {
      continue;
    }
    const target = developmentCost(card);
    const remaining = Math.max(0, target - deed.progress);
    if (target <= 0 || remaining <= 0) {
      continue;
    }

    const progressRatio = clamp(deed.progress / target, 0, 1);
    const completionPressure =
      0.75 + progressRatio * 0.65 + (remaining <= 2 ? 0.3 : 0);
    addCardDemandToSuits(demand, {
      card,
      earningDemand:
        cardEarningDemand(card, phase) *
        INCOMPLETE_DEED_DEMAND_WEIGHT *
        completionPressure,
      scoringDemand:
        cardScoringDemandInDistrict(state, district, playerId, card) *
        INCOMPLETE_DEED_DEMAND_WEIGHT *
        completionPressure,
    });
  }
}

function addHandDemand(
  demand: DemandBySuit,
  state: GameState,
  player: PlayerState,
  phase: number
): void {
  for (const cardId of player.hand) {
    const card = findProperty(cardId);
    if (!card) {
      continue;
    }
    const placementWeight = playerHasLegalPlacement(state, player.id, card)
      ? 1
      : 0.25;
    addCardDemandToSuits(demand, {
      card,
      earningDemand:
        cardEarningDemand(card, phase) *
        HAND_DEMAND_WEIGHT *
        placementWeight,
      scoringDemand:
        bestHandCardScoringDemand(state, player.id, card) *
        HAND_DEMAND_WEIGHT *
        placementWeight,
    });
  }
}

function addUnknownPoolDemand(
  demand: DemandBySuit,
  state: GameState,
  player: PlayerState,
  phase: number
): void {
  const knownCardIds = informationSafeKnownPropertyIds(state, player.id);
  for (const card of PROPERTY_CARDS) {
    if (knownCardIds.has(card.id)) {
      continue;
    }
    addCardDemandToSuits(demand, {
      card,
      earningDemand:
        cardEarningDemand(card, phase) * UNKNOWN_POOL_DEMAND_WEIGHT,
      scoringDemand:
        bestHandCardScoringDemand(state, player.id, card) *
        UNKNOWN_POOL_DEMAND_WEIGHT,
    });
  }
}

function addCardDemandToSuits(
  demand: DemandBySuit,
  {
    card,
    earningDemand,
    scoringDemand,
  }: {
    card: PropertyCard;
    earningDemand: number;
    scoringDemand: number;
  }
): void {
  const suitShare = 1 / Math.sqrt(Math.max(1, card.suits.length));
  for (const suit of card.suits) {
    demand[suit].earningDemand += earningDemand * suitShare;
    demand[suit].scoringDemand += scoringDemand * suitShare;
  }
}

function cardEarningDemand(card: PropertyCard, phase: number): number {
  const rankQuality = card.rank / 9;
  const phaseMultiplier = 1 - 0.7 * smoothstep(phase);
  return (
    incomeProbabilityForRankV2(card.rank) *
    (0.35 + rankQuality) *
    EARNING_DEMAND_SCALE *
    phaseMultiplier
  );
}

function bestHandCardScoringDemand(
  state: GameState,
  playerId: PlayerId,
  card: PropertyCard
): number {
  let best = 0;
  for (const district of state.districts) {
    if (!placementAllowed(card, district, playerId)) {
      continue;
    }
    best = Math.max(
      best,
      cardScoringDemandInDistrict(state, district, playerId, card)
    );
  }
  return best;
}

function cardScoringDemandInDistrict(
  state: GameState,
  district: DistrictState,
  playerId: PlayerId,
  card: PropertyCard
): number {
  const opponentId = otherPlayerId(playerId);
  const stack = district.stacks[playerId];
  const currentOwnScore = districtScore(stack);
  const opponentScore = districtScore(district.stacks[opponentId]);
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

  return (
    Math.max(0, saturatedDelta + controlBonus) * SCORING_DEMAND_SCALE
  );
}

function playerHasLegalPlacement(
  state: GameState,
  playerId: PlayerId,
  card: PropertyCard
): boolean {
  return state.districts.some((district) =>
    placementAllowed(card, district, playerId)
  );
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

function directValueMultiplierForSurplusToken(index: number): number {
  if (index <= 1) {
    return 1;
  }
  if (index === 2) {
    return 0.55;
  }
  if (index === 3) {
    return 0.28;
  }
  return 0.12;
}

function marginalTokenMultiplier(index: number): number {
  return directValueMultiplierForSurplusToken(index);
}

function bestTradeTargetValue(
  sourceSuit: Suit,
  resources: ResourcePool,
  suitValues: SuitValueMap<SuitTokenValueV2>
): number {
  let best = 0;
  for (const targetSuit of SUITS) {
    if (targetSuit === sourceSuit) {
      continue;
    }
    const targetCount = Math.max(0, resources[targetSuit]);
    const scarcity =
      targetCount === 0 ? 1 : targetCount === 1 ? 0.55 : 0.2;
    best = Math.max(best, suitValues[targetSuit].value * scarcity);
  }
  return best;
}

function tradeSourceExpendability(
  sourceValue: number,
  targetValue: number
): number {
  if (targetValue <= 0) {
    return 0;
  }
  const relativeDemand = sourceValue / targetValue;
  return clamp(1.1 - relativeDemand * 0.35, 0.25, 1);
}

function projectPlayerForTokenValue(
  action: GameAction,
  player: PlayerState
): PlayerState {
  if (
    action.type !== 'buy-deed' &&
    action.type !== 'develop-outright' &&
    action.type !== 'sell-card'
  ) {
    return player;
  }
  return {
    ...player,
    hand: player.hand.filter((cardId) => cardId !== action.cardId),
  };
}

function projectDistrictForTokenValue(
  district: DistrictState,
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >,
  playerId: PlayerId
): DistrictState {
  const stack = district.stacks[playerId];
  const projectedStack = projectStackForTokenValue(stack, action);
  return {
    ...district,
    stacks: {
      ...district.stacks,
      [playerId]: projectedStack,
    },
  };
}

function projectStackForTokenValue(
  stack: DistrictState['stacks'][PlayerId],
  action: Extract<
    GameAction,
    { type: 'buy-deed' | 'develop-deed' | 'develop-outright' }
  >
): DistrictState['stacks'][PlayerId] {
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
  if (action.type === 'develop-outright') {
    return {
      ...stack,
      developed: [...stack.developed, action.cardId],
    };
  }

  const deed = stack.deed;
  if (!deed) {
    return stack;
  }
  const card = findProperty(deed.cardId);
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

function replacePlayer(
  state: GameState,
  playerId: PlayerId,
  updatedPlayer: PlayerState
): GameState {
  return {
    ...state,
    players: state.players.map((player) =>
      player.id === playerId ? updatedPlayer : player
    ),
  };
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

function replaceabilityMultiplierForAccess(access: number): number {
  return clamp(1.45 / (1 + 0.45 * access), 0.55, 1.45);
}

function sellCardDelta(cardId: CardId): Partial<Record<Suit, number>> {
  const card = findProperty(cardId);
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

export function incomeProbabilityForRankV2(
  rank: PropertyCard['rank']
): number {
  if (rank === 1) {
    return 0.01;
  }
  return (2 * rank - 1) / 100;
}

function requiredPlayer(state: GameState, playerId: PlayerId): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Token value v2 missing player ${playerId}.`);
  }
  return player;
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function gamePhase(state: GameState): number {
  if ((state.finalTurnsRemaining ?? 0) > 0) {
    return 1;
  }
  return clamp(state.turn / EXPECTED_GAME_TURNS, 0, 1);
}

function smoothstep(value: number): number {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
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
