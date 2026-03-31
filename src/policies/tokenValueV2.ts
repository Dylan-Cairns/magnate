import type { CardId } from '../engine/cards';
import {
  applyDelta,
  deedCost,
  developmentCost,
  findProperty,
  SUITS,
} from '../engine/stateHelpers';
import type {
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PropertyCard,
  ResourcePool,
  Suit,
} from '../engine/types';
import {
  bestCardScoringDemandV2,
  cardEarningDemandV2,
  cardScoringDemandInDistrictV2,
  createHeuristicV2PositionContext,
  knownPropertyIdsForPlayerV2,
  placementAllowedCached,
  propertyCardsUnknownToPlayerV2,
  suitAccessBySuitForPlayerV2,
  type HeuristicV2PositionContext,
  type SuitValueMap,
} from './heuristicV2PositionContext';
import {
  clamp,
  isDistrictAction,
  projectDistrictAction,
} from './policyProjection';

export type { SuitValueMap } from './heuristicV2PositionContext';

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

export interface TokenValueContextV2 extends ResourceBankValueBreakdownV2 {
  state: GameState;
  playerId: PlayerId;
  player: PlayerState;
  resources: ResourcePool;
  knownPropertyIds: ReadonlySet<CardId>;
  districtScores: ReadonlyMap<string, Record<PlayerId, number>>;
  positionContext: HeuristicV2PositionContext;
}

export interface ProjectedTokenValueContextV2 {
  state: GameState;
  context: TokenValueContextV2;
}

type DemandBySuit = SuitValueMap<{
  earningDemand: number;
  scoringDemand: number;
}>;

interface TokenDemandContextV2 {
  playerId: PlayerId;
  player: PlayerState;
  positionContext: HeuristicV2PositionContext;
  knownPropertyIds: ReadonlySet<CardId>;
}

const HAND_DEMAND_WEIGHT = 0.85;
const INCOMPLETE_DEED_DEMAND_WEIGHT = 1.35;
const UNKNOWN_POOL_DEMAND_WEIGHT = 0.035;

const DIRECT_VALUE_FLOOR = 0.02;
const TRADE_SET_WEIGHT = 0.28;
const TRADE_REMAINDER_TWO_WEIGHT = 0.12;
const TRADE_REMAINDER_ONE_WEIGHT = 0.03;

export function contextualSuitTokenValuesV2(
  state: GameState,
  playerId: PlayerId
): SuitValueMap<SuitTokenValueV2> {
  const positionContext = createHeuristicV2PositionContext(state, playerId);
  return contextualSuitTokenValuesFromContextV2(
    createTokenDemandContextV2(positionContext, playerId)
  );
}

export function suitAccessBySuitV2(
  state: GameState,
  playerId: PlayerId
): SuitValueMap<number> {
  return suitAccessBySuitForPlayerV2(
    createHeuristicV2PositionContext(state, playerId),
    playerId
  );
}

export function resourceBankValueV2(
  state: GameState,
  playerId: PlayerId,
  resources = requiredPlayer(state, playerId).resources
): number {
  return createTokenValueContextV2(state, playerId, resources).totalValue;
}

export function resourceBankValueBreakdownV2(
  state: GameState,
  playerId: PlayerId,
  resources = requiredPlayer(state, playerId).resources
): ResourceBankValueBreakdownV2 {
  const context = createTokenValueContextV2(state, playerId, resources);

  return {
    directValue: context.directValue,
    tradeLiquidityValue: context.tradeLiquidityValue,
    totalValue: context.totalValue,
    suitValues: context.suitValues,
    tradeLiquidityBySuit: context.tradeLiquidityBySuit,
  };
}

export function createTokenValueContextV2(
  state: GameState,
  playerId: PlayerId,
  resources = requiredPlayer(state, playerId).resources,
  positionContext = createHeuristicV2PositionContext(state, playerId)
): TokenValueContextV2 {
  const player = requiredPlayer(state, playerId);
  const demandContext = createTokenDemandContextV2(
    positionContext,
    playerId,
    player
  );
  const suitValues = contextualSuitTokenValuesFromContextV2(demandContext);
  const bank = resourceBankValueFromSuitValuesV2(resources, suitValues);

  return {
    state,
    playerId,
    player,
    resources,
    knownPropertyIds: demandContext.knownPropertyIds,
    districtScores: positionContext.districtScoresById,
    positionContext,
    ...bank,
    suitValues,
  };
}

export function tokenDeltaForActionV2(
  action: GameAction,
  state: GameState,
  playerId: PlayerId,
  context = createTokenValueContextV2(state, playerId),
  projected?: ProjectedTokenValueContextV2
): number {
  if (action.type === 'end-turn') {
    return 0;
  }
  if (action.type === 'trade' || action.type === 'choose-income-suit') {
    const after = applyDelta(context.resources, resourceDeltaForActionV2(action));
    return (
      resourceBankValueFromSuitValuesV2(after, context.suitValues).totalValue -
      context.totalValue
    );
  }
  const afterContext =
    projected?.context ??
    projectTokenValueContextForActionV2(action, state, playerId, context)
      .context;
  return afterContext.totalValue - context.totalValue;
}

export function projectTokenValueContextForActionV2(
  action: GameAction,
  state: GameState,
  playerId: PlayerId,
  context = createTokenValueContextV2(state, playerId)
): ProjectedTokenValueContextV2 {
  const after = applyDelta(context.resources, resourceDeltaForActionV2(action));
  const afterState = projectStateForTokenValueV2(action, state, playerId);
  const positionContext = createHeuristicV2PositionContext(
    afterState,
    playerId
  );
  return {
    state: afterState,
    context: createTokenValueContextV2(
      afterState,
      playerId,
      after,
      positionContext
    ),
  };
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
            ? projectDistrictAction(district, action, playerId)
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
  const completedSetBonus = completedSets * TRADE_SET_WEIGHT * bestTargetValue;
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
  context: TokenDemandContextV2
): DemandBySuit {
  const demand = emptySuitValueMap(() => ({
    earningDemand: 0,
    scoringDemand: 0,
  }));

  addIncompleteDeedDemand(demand, context);
  addHandDemand(demand, context);
  addUnknownPoolDemand(demand, context);

  return demand;
}

function addIncompleteDeedDemand(
  demand: DemandBySuit,
  context: TokenDemandContextV2
): void {
  const { positionContext, playerId } = context;
  const { state } = positionContext;
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
        cardEarningDemandV2(positionContext, card) *
        INCOMPLETE_DEED_DEMAND_WEIGHT *
        completionPressure,
      scoringDemand:
        cardScoringDemandInDistrictV2(
          positionContext,
          playerId,
          district,
          card
        ) *
        INCOMPLETE_DEED_DEMAND_WEIGHT *
        completionPressure,
    });
  }
}

function addHandDemand(
  demand: DemandBySuit,
  context: TokenDemandContextV2
): void {
  const { player, playerId, positionContext } = context;
  for (const cardId of player.hand) {
    const card = findProperty(cardId);
    if (!card) {
      continue;
    }
    const placementWeight = playerHasLegalPlacement(positionContext, playerId, card)
      ? 1
      : 0.25;
    addCardDemandToSuits(demand, {
      card,
      earningDemand:
        cardEarningDemandV2(positionContext, card) *
        HAND_DEMAND_WEIGHT *
        placementWeight,
      scoringDemand:
        bestCardScoringDemandV2(positionContext, playerId, card) *
        HAND_DEMAND_WEIGHT *
        placementWeight,
    });
  }
}

function addUnknownPoolDemand(
  demand: DemandBySuit,
  context: TokenDemandContextV2
): void {
  const { playerId, positionContext } = context;
  for (const card of propertyCardsUnknownToPlayerV2(
    positionContext,
    playerId
  )) {
    addCardDemandToSuits(demand, {
      card,
      earningDemand:
        cardEarningDemandV2(positionContext, card) *
        UNKNOWN_POOL_DEMAND_WEIGHT,
      scoringDemand:
        bestCardScoringDemandV2(positionContext, playerId, card) *
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

function playerHasLegalPlacement(
  positionContext: HeuristicV2PositionContext,
  playerId: PlayerId,
  card: PropertyCard
): boolean {
  return positionContext.state.districts.some((district) =>
    placementAllowedCached(positionContext, playerId, district, card)
  );
}

function createTokenDemandContextV2(
  positionContext: HeuristicV2PositionContext,
  playerId: PlayerId,
  player = requiredPlayer(positionContext.state, playerId)
): TokenDemandContextV2 {
  return {
    playerId,
    player,
    positionContext,
    knownPropertyIds: knownPropertyIdsForPlayerV2(positionContext, playerId),
  };
}

function contextualSuitTokenValuesFromContextV2(
  context: TokenDemandContextV2
): SuitValueMap<SuitTokenValueV2> {
  const access = suitAccessBySuitForPlayerV2(
    context.positionContext,
    context.playerId
  );
  const demand = demandBySuitForPlayer(context);
  return emptySuitValueMap<SuitTokenValueV2>((suit) => {
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
          : Math.max(DIRECT_VALUE_FLOOR, rawDemand * replaceabilityMultiplier),
    };
  });
}

function resourceBankValueFromSuitValuesV2(
  resources: ResourcePool,
  suitValues: SuitValueMap<SuitTokenValueV2>
): Pick<
  ResourceBankValueBreakdownV2,
  'directValue' | 'tradeLiquidityValue' | 'totalValue' | 'tradeLiquidityBySuit'
> {
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
    tradeLiquidityBySuit,
  };
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
    const scarcity = targetCount === 0 ? 1 : targetCount === 1 ? 0.55 : 0.2;
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

function requiredPlayer(state: GameState, playerId: PlayerId): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Token value v2 missing player ${playerId}.`);
  }
  return player;
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
