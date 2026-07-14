import { toKeyedActions } from '../engine/actionSurface';
import {
  ALL_CARDS,
  CARD_BY_ID,
  PROPERTY_CARDS,
  type CardId,
} from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
} from '../engine/decisionActor';
import { incomeForResult } from '../engine/income';
import { applyKnownLegalAction } from '../engine/reducer';
import { districtScore, scoreGame } from '../engine/scoring';
import {
  SUITS,
  developmentCost,
  findProperty,
  placementAllowed,
  sumTokens,
} from '../engine/stateHelpers';
import type {
  DistrictId,
  DistrictStack,
  GameAction,
  GamePhase,
  GameState,
  PlayerId,
  Rank,
  ResourcePool,
  Suit,
  WinnerDecider,
} from '../engine/types';
import { toPlayerView } from '../engine/view';

export const STRATEGIC_STATE_SUMMARY_CONTRACT =
  'magnate.strategic-state-summary' as const;
export const STRATEGIC_STATE_SUMMARY_VERSION = 0 as const;

const INCOME_RESULTS: readonly Rank[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const CARD_INDEX = new Map(ALL_CARDS.map((card, index) => [card.id, index]));

export type RelativeOutcomeV0 = 'ahead' | 'behind' | 'tied';
export type RelativeDistrictControlV0 = 'self' | 'opponent' | 'tied';
export type SuitCountsV0 = Readonly<Record<Suit, number>>;

export interface RelativeIntegerMetricV0 {
  readonly self: number;
  readonly opponent: number;
  readonly margin: number;
}

export interface StrategicStateSummaryV0 {
  readonly contract: typeof STRATEGIC_STATE_SUMMARY_CONTRACT;
  readonly version: typeof STRATEGIC_STATE_SUMMARY_VERSION;
  readonly sourceStateSchemaVersion: number;
  readonly visibility: 'player-view';
  readonly perspectivePlayerId: PlayerId;
  readonly opponentPlayerId: PlayerId;
  readonly turn: StrategicTurnFactsV0;
  readonly clock: StrategicClockFactsV0;
  readonly score: StrategicLiveScoreFactsV0;
  readonly players: {
    readonly self: StrategicPlayerFactsV0;
    readonly opponent: StrategicPlayerFactsV0;
  };
  readonly districts: readonly StrategicDistrictFactsV0[];
  readonly cards: StrategicCardKnowledgeFactsV0;
}

export interface StrategicTurnFactsV0 {
  readonly turnNumber: number;
  readonly phase: GamePhase;
  readonly turnOwnerId: PlayerId;
  readonly cardPlayedThisTurn: boolean;
  readonly isTerminal: boolean;
}

export interface StrategicClockFactsV0 {
  readonly drawCount: number;
  readonly discardCount: number;
  readonly reshuffles: 0 | 1 | 2;
  readonly finalTurnsRemaining: number | null;
}

export interface StrategicLiveScoreFactsV0 {
  readonly districts: RelativeIntegerMetricV0 & { readonly tied: number };
  readonly developedRankTotal: RelativeIntegerMetricV0;
  readonly resources: RelativeIntegerMetricV0;
  readonly currentLexicographicOutcome: RelativeOutcomeV0;
  readonly currentDecider: WinnerDecider;
}

export interface StrategicPlayerFactsV0 {
  readonly playerId: PlayerId;
  readonly crownCardIds: readonly CardId[];
  readonly crownSuitCounts: SuitCountsV0;
  readonly resourcesBySuit: SuitCountsV0;
  readonly resourceTotal: number;
  readonly taxLossIfSuitSelected: SuitCountsV0;
  readonly developedPropertyCount: number;
  readonly incompleteDeedCount: number;
  readonly incomeByResult: readonly StrategicIncomeResultFactsV0[];
}

export interface StrategicIncomeResultFactsV0 {
  readonly result: Rank;
  readonly fixedTokensBySuit: SuitCountsV0;
  readonly choiceSources: readonly StrategicIncomeChoiceSourceV0[];
}

export interface StrategicIncomeChoiceSourceV0 {
  readonly districtId: DistrictId;
  readonly cardId: CardId;
  readonly allowedSuits: readonly Suit[];
}

export interface StrategicDistrictFactsV0 {
  readonly districtId: DistrictId;
  readonly boardIndex: number;
  readonly markerSuitMask: readonly Suit[];
  readonly score: {
    readonly self: number;
    readonly opponent: number;
    readonly margin: number;
    readonly control: RelativeDistrictControlV0;
  };
  readonly self: StrategicStackFactsV0;
  readonly opponent: StrategicStackFactsV0;
  readonly placementSupport: {
    readonly ownHandForSelf: readonly CardId[];
    readonly unknownPoolForSelf: readonly CardId[];
    readonly unknownPoolForOpponent: readonly CardId[];
  };
}

export interface StrategicStackFactsV0 {
  readonly developedCardIds: readonly CardId[];
  readonly developedPropertyCount: number;
  readonly developedRankTotal: number;
  readonly aceBonus: number;
  readonly score: number;
  readonly deed: StrategicDeedFactsV0 | null;
  readonly placementConstraint: StrategicPlacementConstraintV0;
}

export interface StrategicDeedFactsV0 {
  readonly cardId: CardId;
  readonly rank: Exclude<Rank, 10>;
  readonly suits: readonly Suit[];
  readonly progress: number;
  readonly target: number;
  readonly remaining: number;
  readonly tokensBySuit: SuitCountsV0;
  readonly matchingLooseResources: number;
  readonly spendableMatchingResources: number;
  readonly resourceCompletionShortfall: number;
  readonly hasResourcesToComplete: boolean;
}

export type StrategicPlacementConstraintV0 =
  | {
      readonly kind: 'blocked-by-deed';
      readonly deedCardId: CardId;
    }
  | { readonly kind: 'unrestricted' }
  | {
      readonly kind: 'shares-any-suit';
      readonly source: 'district-marker';
      readonly suits: readonly Suit[];
    }
  | {
      readonly kind: 'shares-any-suit';
      readonly source: 'top-developed';
      readonly sourceCardId: CardId;
      readonly suits: readonly Suit[];
    };

export interface StrategicCardKnowledgeFactsV0 {
  readonly ownHandCardIds: readonly CardId[];
  readonly opponentHandCount: number;
  readonly discardPropertyCardIds: readonly CardId[];
  readonly unknownPropertyCardIds: readonly CardId[];
}

export type PlayedCardDestinationV0 =
  | 'developed'
  | 'deed'
  | 'first-reshuffle-discard'
  | 'dead-discard'
  | null;

export interface StrategicActionDeltaV0 {
  readonly actionKey: string;
  readonly actionType: GameAction['type'];
  readonly cardId: CardId | null;
  readonly districtId: DistrictId | null;
  readonly districtPointMarginDelta: number;
  readonly developedRankMarginDelta: number;
  readonly resourceMarginDelta: number;
  readonly targetDistrictScoreMarginDelta: number;
  readonly currentOutcomeBefore: RelativeOutcomeV0;
  readonly currentOutcomeAfter: RelativeOutcomeV0;
  readonly cardPlayAvailableAfterAction: boolean;
  readonly playedCardDestination: PlayedCardDestinationV0;
}

/**
 * Creates detached, JSON-safe facts through the perspective player's visibility
 * boundary. No probability, normalization, strategic label, or action value is
 * included.
 */
export function strategicStateSummaryV0(
  state: GameState,
  perspectivePlayerId: PlayerId
): StrategicStateSummaryV0 {
  const view = toPlayerView(state, perspectivePlayerId);
  const opponentPlayerId = otherPlayerId(perspectivePlayerId);
  const turnOwnerId = state.players[state.activePlayerIndex]?.id;
  if (!turnOwnerId) {
    throw new Error('Strategic summary could not resolve the turn owner.');
  }

  const selfView = requiredPlayerView(view.players, perspectivePlayerId);
  const opponentView = requiredPlayerView(view.players, opponentPlayerId);
  const cards = cardKnowledgeFacts({
    state,
    perspectivePlayerId,
    ownHandCardIds: selfView.hand,
    opponentHandCount: opponentView.handCount,
    drawCount: view.deck.drawCount,
    discardCardIds: view.deck.discard,
  });
  const liveScore = scoreGame(state);
  assertTerminalScoreConsistency(state, liveScore);

  const districts = view.districts.map((district, boardIndex) => {
    const selfStack = district.stacks[perspectivePlayerId];
    const opponentStack = district.stacks[opponentPlayerId];
    const selfScore = districtScore(selfStack);
    const opponentScore = districtScore(opponentStack);
    const margin = selfScore - opponentScore;
    return {
      districtId: district.id,
      boardIndex,
      markerSuitMask: sortSuits(district.markerSuitMask),
      score: {
        self: selfScore,
        opponent: opponentScore,
        margin,
        control: relativeControl(margin),
      },
      self: stackFacts(selfStack, district.markerSuitMask, selfView.resources),
      opponent: stackFacts(
        opponentStack,
        district.markerSuitMask,
        opponentView.resources
      ),
      placementSupport: {
        ownHandForSelf: compatibleCards(
          cards.ownHandCardIds,
          district,
          perspectivePlayerId
        ),
        unknownPoolForSelf: compatibleCards(
          cards.unknownPropertyCardIds,
          district,
          perspectivePlayerId
        ),
        unknownPoolForOpponent: compatibleCards(
          cards.unknownPropertyCardIds,
          district,
          opponentPlayerId
        ),
      },
    } satisfies StrategicDistrictFactsV0;
  });

  const districtPoints = relativeMetric(
    liveScore.districtPoints[perspectivePlayerId],
    liveScore.districtPoints[opponentPlayerId]
  );
  const developedRankTotal = relativeMetric(
    liveScore.rankTotals[perspectivePlayerId],
    liveScore.rankTotals[opponentPlayerId]
  );
  const resources = relativeMetric(
    liveScore.resourceTotals[perspectivePlayerId],
    liveScore.resourceTotals[opponentPlayerId]
  );

  return {
    contract: STRATEGIC_STATE_SUMMARY_CONTRACT,
    version: STRATEGIC_STATE_SUMMARY_VERSION,
    sourceStateSchemaVersion: state.schemaVersion,
    visibility: 'player-view',
    perspectivePlayerId,
    opponentPlayerId,
    turn: {
      turnNumber: view.turn,
      phase: view.phase,
      turnOwnerId,
      cardPlayedThisTurn: view.cardPlayedThisTurn,
      isTerminal: view.phase === 'GameOver',
    },
    clock: {
      drawCount: view.deck.drawCount,
      discardCount: view.deck.discard.length,
      reshuffles: view.deck.reshuffles,
      finalTurnsRemaining: view.finalTurnsRemaining ?? null,
    },
    score: {
      districts: {
        ...districtPoints,
        tied:
          state.districts.length -
          districtPoints.self -
          districtPoints.opponent,
      },
      developedRankTotal,
      resources,
      currentLexicographicOutcome: relativeOutcome(
        liveScore.winner,
        perspectivePlayerId
      ),
      currentDecider: liveScore.decidedBy,
    },
    players: {
      self: playerFacts(state, perspectivePlayerId, selfView),
      opponent: playerFacts(state, opponentPlayerId, opponentView),
    },
    districts,
    cards,
  };
}

/**
 * Describes exact immediate consequences of canonical legal ActionWindow
 * actions. These are deltas of factual summaries, not action scores.
 */
export function strategicActionDeltasV0(
  state: GameState,
  perspectivePlayerId: PlayerId
): readonly StrategicActionDeltaV0[] {
  if (state.phase !== 'ActionWindow') {
    throw new Error(
      `Strategic action deltas require ActionWindow; received ${state.phase}.`
    );
  }
  const actor = decisionPlayerIdForState(state);
  if (actor !== perspectivePlayerId) {
    throw new Error(
      `Strategic action delta perspective ${perspectivePlayerId} is not decision actor ${String(actor)}.`
    );
  }

  const before = strategicStateSummaryV0(state, perspectivePlayerId);
  return toKeyedActions(
    legalActionsForDecisionPlayer(state, perspectivePlayerId)
  ).map((candidate) => {
    const nextState = applyKnownLegalAction(state, candidate.action, {
      recordLog: false,
    });
    const after = strategicStateSummaryV0(nextState, perspectivePlayerId);
    const targetBefore = districtMarginForAction(before, candidate.action);
    const targetAfter = districtMarginForAction(after, candidate.action);
    return {
      actionKey: candidate.actionKey,
      actionType: candidate.action.type,
      cardId: actionCardId(candidate.action),
      districtId: actionDistrictId(candidate.action),
      districtPointMarginDelta:
        after.score.districts.margin - before.score.districts.margin,
      developedRankMarginDelta:
        after.score.developedRankTotal.margin -
        before.score.developedRankTotal.margin,
      resourceMarginDelta:
        after.score.resources.margin - before.score.resources.margin,
      targetDistrictScoreMarginDelta: targetAfter - targetBefore,
      currentOutcomeBefore: before.score.currentLexicographicOutcome,
      currentOutcomeAfter: after.score.currentLexicographicOutcome,
      cardPlayAvailableAfterAction:
        nextState.phase === 'ActionWindow' && !nextState.cardPlayedThisTurn,
      playedCardDestination: playedCardDestination(
        candidate.action,
        state.deck.reshuffles
      ),
    };
  });
}

function playerFacts(
  state: GameState,
  playerId: PlayerId,
  player: ReturnType<typeof toPlayerView>['players'][number]
): StrategicPlayerFactsV0 {
  let developedPropertyCount = 0;
  let incompleteDeedCount = 0;
  for (const district of state.districts) {
    developedPropertyCount += district.stacks[playerId].developed.length;
    if (district.stacks[playerId].deed) {
      incompleteDeedCount += 1;
    }
  }

  const resourcesBySuit = suitCounts(player.resources);
  return {
    playerId,
    crownCardIds: sortCardIds(player.crowns),
    crownSuitCounts: crownSuitCounts(player.crowns),
    resourcesBySuit,
    resourceTotal: totalSuitCounts(resourcesBySuit),
    taxLossIfSuitSelected: suitCounts(
      Object.fromEntries(
        SUITS.map((suit) => [suit, Math.max(0, resourcesBySuit[suit] - 1)])
      )
    ),
    developedPropertyCount,
    incompleteDeedCount,
    incomeByResult: INCOME_RESULTS.map((result) => {
      const income = incomeForResult(state, playerId, result);
      return {
        result,
        fixedTokensBySuit: suitCounts(income.fixedDelta),
        choiceSources: income.pendingChoices.map((choice) => ({
          districtId: choice.districtId,
          cardId: choice.cardId,
          allowedSuits: sortSuits(choice.suits),
        })),
      };
    }),
  };
}

function stackFacts(
  stack: DistrictStack,
  markerSuitMask: readonly Suit[],
  resources: ResourcePool
): StrategicStackFactsV0 {
  const developed = stack.developed.map((cardId) => requiredProperty(cardId));
  const developedRankTotal = developed.reduce(
    (total, card) => total + card.rank,
    0
  );
  const score = districtScore(stack);
  const deed = stack.deed ? deedFacts(stack.deed, resources) : null;

  return {
    developedCardIds: [...stack.developed],
    developedPropertyCount: developed.length,
    developedRankTotal,
    aceBonus: score - developedRankTotal,
    score,
    deed,
    placementConstraint: placementConstraint(stack, markerSuitMask),
  };
}

function deedFacts(
  deed: NonNullable<DistrictStack['deed']>,
  resources: ResourcePool
): StrategicDeedFactsV0 {
  const card = requiredProperty(deed.cardId);
  const target = developmentCost(card);
  const tokensBySuit = suitCounts(deed.tokens);
  if (
    !Number.isSafeInteger(deed.progress) ||
    deed.progress < 0 ||
    deed.progress >= target
  ) {
    throw new Error(
      `Invalid incomplete deed progress for ${deed.cardId}: ${String(deed.progress)}/${String(target)}.`
    );
  }
  if (sumTokens(tokensBySuit) !== deed.progress) {
    throw new Error(
      `Deed ${deed.cardId} token total does not match progress ${String(deed.progress)}.`
    );
  }
  for (const suit of SUITS) {
    if (tokensBySuit[suit] > 0 && !card.suits.includes(suit)) {
      throw new Error(`Deed ${deed.cardId} has an invalid ${suit} token.`);
    }
  }

  const remaining = target - deed.progress;
  const matchingLooseResources = card.suits.reduce(
    (total, suit) => total + resources[suit],
    0
  );
  const spendableMatchingResources = Math.min(
    remaining,
    matchingLooseResources
  );
  return {
    cardId: card.id,
    rank: card.rank,
    suits: sortSuits(card.suits),
    progress: deed.progress,
    target,
    remaining,
    tokensBySuit,
    matchingLooseResources,
    spendableMatchingResources,
    resourceCompletionShortfall: remaining - spendableMatchingResources,
    hasResourcesToComplete: spendableMatchingResources === remaining,
  };
}

function placementConstraint(
  stack: DistrictStack,
  markerSuitMask: readonly Suit[]
): StrategicPlacementConstraintV0 {
  if (stack.deed) {
    return { kind: 'blocked-by-deed', deedCardId: stack.deed.cardId };
  }
  const topCardId = stack.developed.at(-1);
  if (topCardId) {
    const card = requiredProperty(topCardId);
    return {
      kind: 'shares-any-suit',
      source: 'top-developed',
      sourceCardId: card.id,
      suits: sortSuits(card.suits),
    };
  }
  if (markerSuitMask.length === 0) {
    return { kind: 'unrestricted' };
  }
  return {
    kind: 'shares-any-suit',
    source: 'district-marker',
    suits: sortSuits(markerSuitMask),
  };
}

function cardKnowledgeFacts({
  state,
  perspectivePlayerId,
  ownHandCardIds,
  opponentHandCount,
  drawCount,
  discardCardIds,
}: {
  state: GameState;
  perspectivePlayerId: PlayerId;
  ownHandCardIds: readonly CardId[];
  opponentHandCount: number;
  drawCount: number;
  discardCardIds: readonly CardId[];
}): StrategicCardKnowledgeFactsV0 {
  const ownHand = ownHandCardIds.map(requiredProperty).map((card) => card.id);
  const discard = discardCardIds
    .map((cardId) => findProperty(cardId))
    .filter(isDefined)
    .map((card) => card.id);
  const board = state.districts.flatMap((district) =>
    (['PlayerA', 'PlayerB'] as const).flatMap((playerId) => {
      const stack = district.stacks[playerId];
      return [...stack.developed, ...(stack.deed ? [stack.deed.cardId] : [])];
    })
  );

  const known = new Set<CardId>();
  for (const [zone, cardIds] of [
    ['own hand', ownHand],
    ['public discard', discard],
    ['public board', board],
  ] as const) {
    for (const cardId of cardIds) {
      requiredProperty(cardId);
      if (known.has(cardId)) {
        throw new Error(
          `Strategic summary card ${cardId} appears in multiple known zones (latest: ${zone}).`
        );
      }
      known.add(cardId);
    }
  }

  const unknownPropertyCardIds = PROPERTY_CARDS.filter(
    (card) => !known.has(card.id)
  ).map((card) => card.id);
  const expectedUnknownCount = drawCount + opponentHandCount;
  if (unknownPropertyCardIds.length !== expectedUnknownCount) {
    throw new Error(
      `Strategic summary card partition mismatch for ${perspectivePlayerId}: unknown=${String(unknownPropertyCardIds.length)} draw+opponentHand=${String(expectedUnknownCount)}.`
    );
  }

  return {
    ownHandCardIds: sortCardIds(ownHand),
    opponentHandCount,
    discardPropertyCardIds: sortCardIds(discard),
    unknownPropertyCardIds,
  };
}

function compatibleCards(
  cardIds: readonly CardId[],
  district: GameState['districts'][number],
  playerId: PlayerId
): CardId[] {
  return cardIds.filter((cardId) =>
    placementAllowed(requiredProperty(cardId), district, playerId)
  );
}

function crownSuitCounts(cardIds: readonly CardId[]): SuitCountsV0 {
  const counts = mutableSuitCounts();
  for (const cardId of cardIds) {
    // Crown identities are public, but malformed non-Crowns should not create
    // invented income. incomeForResult remains the canonical behavior.
    const crown = CARD_BY_ID[cardId];
    if (crown?.kind === 'Crown') {
      counts[crown.suits[0]] += 1;
    }
  }
  return counts;
}

function suitCounts(
  values: Partial<Record<Suit, number>> | Record<string, unknown>
): SuitCountsV0 {
  const counts = mutableSuitCounts();
  for (const suit of SUITS) {
    const value = values[suit];
    if (value === undefined) {
      continue;
    }
    if (
      typeof value !== 'number' ||
      !Number.isSafeInteger(value) ||
      value < 0
    ) {
      throw new Error(
        `Strategic summary expected ${suit} to be a nonnegative safe integer; received ${String(value)}.`
      );
    }
    counts[suit] = value;
  }
  return counts;
}

function mutableSuitCounts(): Record<Suit, number> {
  return {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
}

function sortSuits(suits: readonly Suit[]): Suit[] {
  const included = new Set(suits);
  return SUITS.filter((suit) => included.has(suit));
}

function sortCardIds(cardIds: readonly CardId[]): CardId[] {
  return [...cardIds].sort(
    (left, right) =>
      (CARD_INDEX.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (CARD_INDEX.get(right) ?? Number.MAX_SAFE_INTEGER)
  );
}

function relativeMetric(
  self: number,
  opponent: number
): RelativeIntegerMetricV0 {
  return { self, opponent, margin: self - opponent };
}

function relativeControl(margin: number): RelativeDistrictControlV0 {
  return margin > 0 ? 'self' : margin < 0 ? 'opponent' : 'tied';
}

function relativeOutcome(
  winner: ReturnType<typeof scoreGame>['winner'],
  perspectivePlayerId: PlayerId
): RelativeOutcomeV0 {
  if (winner === 'Draw') {
    return 'tied';
  }
  return winner === perspectivePlayerId ? 'ahead' : 'behind';
}

function requiredPlayerView(
  players: ReturnType<typeof toPlayerView>['players'],
  playerId: PlayerId
): ReturnType<typeof toPlayerView>['players'][number] {
  const player = players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Strategic summary is missing player ${playerId}.`);
  }
  return player;
}

function requiredProperty(cardId: CardId) {
  const card = findProperty(cardId);
  if (!card) {
    throw new Error(`Strategic summary expected property card ${cardId}.`);
  }
  return card;
}

function assertTerminalScoreConsistency(
  state: GameState,
  liveScore: ReturnType<typeof scoreGame>
): void {
  if (
    state.phase === 'GameOver' &&
    (!state.finalScore || !sameFinalScore(state.finalScore, liveScore))
  ) {
    throw new Error(
      'Terminal state finalScore does not match canonical scoreGame.'
    );
  }
}

function sameFinalScore(
  left: ReturnType<typeof scoreGame>,
  right: ReturnType<typeof scoreGame>
): boolean {
  return (
    left.winner === right.winner &&
    left.decidedBy === right.decidedBy &&
    left.districtPoints.PlayerA === right.districtPoints.PlayerA &&
    left.districtPoints.PlayerB === right.districtPoints.PlayerB &&
    left.rankTotals.PlayerA === right.rankTotals.PlayerA &&
    left.rankTotals.PlayerB === right.rankTotals.PlayerB &&
    left.resourceTotals.PlayerA === right.resourceTotals.PlayerA &&
    left.resourceTotals.PlayerB === right.resourceTotals.PlayerB
  );
}

function totalSuitCounts(counts: SuitCountsV0): number {
  return SUITS.reduce((total, suit) => total + counts[suit], 0);
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function districtMarginForAction(
  summary: StrategicStateSummaryV0,
  action: GameAction
): number {
  const districtId = actionDistrictId(action);
  if (!districtId) {
    return 0;
  }
  return (
    summary.districts.find((district) => district.districtId === districtId)
      ?.score.margin ?? 0
  );
}

function actionCardId(action: GameAction): CardId | null {
  return 'cardId' in action ? action.cardId : null;
}

function actionDistrictId(action: GameAction): DistrictId | null {
  return 'districtId' in action ? action.districtId : null;
}

function playedCardDestination(
  action: GameAction,
  reshuffles: 0 | 1 | 2
): PlayedCardDestinationV0 {
  switch (action.type) {
    case 'develop-outright':
      return 'developed';
    case 'buy-deed':
      return 'deed';
    case 'sell-card':
      return reshuffles === 0 ? 'first-reshuffle-discard' : 'dead-discard';
    default:
      return null;
  }
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}
