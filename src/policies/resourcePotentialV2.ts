import { applyDelta } from '../engine/stateHelpers';
import {
  developmentCost,
  findDevelopableCard,
  SUITS,
} from '../engine/stateHelpers';
import {
  type GameAction,
  type GameState,
  type PlayerId,
  type ResourcePool,
  Suit,
} from '../engine/types';
import { isCourtCard } from './courtPotentialV2';
import {
  cardEarningDemandV2,
  cardScoringDemandInDistrictV2,
  createHeuristicV2PositionContext,
  type HeuristicV2PositionContext,
  type SuitValueMap,
} from './heuristicV2PositionContext';
import { resourceDeltaForActionV2 } from './tokenValueV2';

/**
 * Value of one token that no unfinished deed can use. It is deliberately small:
 * surplus is nearly worthless, but a 3:1 trade still destroys two of them, so
 * shedding surplus at bank rates is a strict loss unless the received token
 * closes a concrete deficit.
 */
export const SURPLUS_TOKEN_VALUE = 0.05;

interface SuitNeed {
  /** Marginal value of one token that would progress the best deed wanting it. */
  perTokenValue: number;
  /** How many tokens of this suit that deed can still consume. */
  remaining: number;
}

/**
 * Target-anchored resource potential (heuristic v2). Resources are valued only
 * by how much they close concrete deficits: tokens that progress an unfinished
 * deed on the board score by that deed's per-token swing, and everything else
 * scores the small surplus value. It is monotone non-decreasing in every suit
 * count, and a 3:1 trade strictly decreases it unless the received token is
 * worth more than three surplus tokens.
 *
 * The same potential is used by the action term for resource-converting actions
 * (`trade`, `choose-income-suit`) and by the search leaf's resource term, so the
 * two valuations cannot disagree.
 */
export function resourcePotentialV2(
  state: GameState,
  playerId: PlayerId,
  context: HeuristicV2PositionContext = createHeuristicV2PositionContext(
    state,
    playerId
  )
): number {
  return potentialFromNeedMap(
    requiredResources(state, playerId),
    suitNeedMapV2(state, playerId, context)
  );
}

/**
 * Change in `resourcePotentialV2` caused by an action's resource delta. The
 * target set (unfinished deeds) is unchanged by a pure resource conversion, so
 * only the pool is projected.
 */
export function resourcePotentialDeltaForActionV2(
  action: GameAction,
  state: GameState,
  playerId: PlayerId,
  context: HeuristicV2PositionContext = createHeuristicV2PositionContext(
    state,
    playerId
  )
): number {
  const needMap = suitNeedMapV2(state, playerId, context);
  const resources = requiredResources(state, playerId);
  const after = applyDelta(resources, resourceDeltaForActionV2(action));
  return (
    potentialFromNeedMap(after, needMap) -
    potentialFromNeedMap(resources, needMap)
  );
}

function suitNeedMapV2(
  state: GameState,
  playerId: PlayerId,
  context: HeuristicV2PositionContext
): SuitValueMap<SuitNeed> {
  const map = emptySuitValueMap<SuitNeed>(() => ({
    perTokenValue: 0,
    remaining: 0,
  }));

  for (const district of state.districts) {
    const deed = district.stacks[playerId].deed;
    if (!deed) {
      continue;
    }
    const card = findDevelopableCard(deed.cardId);
    // Courts are owned by the court valuation term; leave them alone here.
    if (!card || isCourtCard(card)) {
      continue;
    }
    const target = developmentCost(card);
    const remaining = Math.max(0, target - deed.progress);
    if (remaining <= 0) {
      continue;
    }
    const perTokenValue =
      (cardScoringDemandInDistrictV2(context, playerId, district, card) +
        cardEarningDemandV2(context, card)) /
      remaining;
    for (const suit of card.suits) {
      const entry = map[suit];
      if (perTokenValue > entry.perTokenValue) {
        entry.perTokenValue = perTokenValue;
        entry.remaining = remaining;
      } else if (perTokenValue === entry.perTokenValue) {
        entry.remaining = Math.max(entry.remaining, remaining);
      }
    }
  }

  return map;
}

function potentialFromNeedMap(
  resources: ResourcePool,
  needMap: SuitValueMap<SuitNeed>
): number {
  let total = 0;
  for (const suit of SUITS) {
    const count = Math.max(0, resources[suit]);
    const need = needMap[suit];
    const coveredValue = Math.max(need.perTokenValue, SURPLUS_TOKEN_VALUE);
    const covered = Math.min(count, need.remaining);
    total += covered * coveredValue + (count - covered) * SURPLUS_TOKEN_VALUE;
  }
  return total;
}

function requiredResources(state: GameState, playerId: PlayerId): ResourcePool {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Resource potential missing player ${playerId}.`);
  }
  return player.resources;
}

function emptySuitValueMap<T>(create: (suit: Suit) => T): SuitValueMap<T> {
  return {
    [Suit.Moons]: create(Suit.Moons),
    [Suit.Suns]: create(Suit.Suns),
    [Suit.Waves]: create(Suit.Waves),
    [Suit.Leaves]: create(Suit.Leaves),
    [Suit.Wyrms]: create(Suit.Wyrms),
    [Suit.Knots]: create(Suit.Knots),
  };
}
