import { CARD_BY_ID } from '../engine/cards';
import {
  districtScore,
  scoreGame,
  scoreMarginsForPlayer,
} from '../engine/scoring';
import { developmentCost, findProperty, SUITS } from '../engine/stateHelpers';
import type {
  DistrictState,
  DistrictStack,
  GameState,
  PlayerId,
  PlayerState,
  ResourcePool,
  Suit,
} from '../engine/types';
import { resourceBankValueV2 } from './tokenValueV2';

const LATE_GAME_DRAW_COUNT = 6;
const TERMINAL_OUTCOME_BASE_VALUE = 0.72;
const TERMINAL_MARGIN_VALUE = 1 - TERMINAL_OUTCOME_BASE_VALUE;

export function evaluateSearchLeafState(
  state: GameState,
  rootPlayer: PlayerId
): number {
  const opponent = otherPlayerId(rootPlayer);
  const lateGame = isLateGame(state);

  const districtTerm = districtControlTerm(state, rootPlayer, opponent);
  const rankTerm = developedRankTerm(state, rootPlayer, opponent);
  const deedTerm = deedPotentialTerm(state, rootPlayer, opponent);
  const resourceTerm = resourceQualityTerm(state, rootPlayer, opponent);

  const score = lateGame
    ? 0.7 * districtTerm +
      0.18 * rankTerm +
      0.05 * deedTerm +
      0.07 * resourceTerm
    : 0.56 * districtTerm +
      0.16 * rankTerm +
      0.18 * deedTerm +
      0.1 * resourceTerm;

  return clamp(score, -1, 1);
}

export function evaluateSearchTerminalState(
  state: GameState,
  rootPlayer: PlayerId
): number {
  const finalScore = state.finalScore ?? scoreGame(state);
  if (finalScore.winner === 'Draw') {
    return 0;
  }

  const outcomeSign = finalScore.winner === rootPlayer ? 1 : -1;
  const margins = scoreMarginsForPlayer(state, rootPlayer, finalScore);
  const marginFromWinnerPerspective = clamp(
    outcomeSign *
      (0.55 * Math.tanh(margins.districtPointMargin / 3) +
        0.22 * Math.tanh(margins.districtScoreMarginTotal / 18) +
        0.15 * Math.tanh(margins.rankTotalMargin / 24) +
        0.08 * Math.tanh(margins.resourceMargin / 12)),
    0,
    1
  );

  return (
    outcomeSign *
    (TERMINAL_OUTCOME_BASE_VALUE +
      TERMINAL_MARGIN_VALUE * marginFromWinnerPerspective)
  );
}

function districtControlTerm(
  state: GameState,
  rootPlayer: PlayerId,
  opponent: PlayerId
): number {
  let total = 0;
  for (const district of state.districts) {
    const rootScore = districtScore(district.stacks[rootPlayer]);
    const opponentScore = districtScore(district.stacks[opponent]);
    total += districtMarginValue(rootScore - opponentScore);
  }
  return total / Math.max(1, state.districts.length);
}

function districtMarginValue(margin: number): number {
  if (margin === 0) {
    return 0;
  }
  const sign = margin > 0 ? 1 : -1;
  const stability = 0.65 + Math.min(0.35, Math.abs(margin) * 0.07);
  return sign * stability;
}

function developedRankTerm(
  state: GameState,
  rootPlayer: PlayerId,
  opponent: PlayerId
): number {
  let diff = 0;
  for (const district of state.districts) {
    diff +=
      developedRankTotal(district.stacks[rootPlayer]) -
      developedRankTotal(district.stacks[opponent]);
  }
  return Math.tanh(diff / 18);
}

function deedPotentialTerm(
  state: GameState,
  rootPlayer: PlayerId,
  opponent: PlayerId
): number {
  const root = requiredPlayerState(state, rootPlayer);
  const opponentState = requiredPlayerState(state, opponent);
  let diff = 0;
  for (const district of state.districts) {
    diff +=
      deedPotentialForPlayer(state, district, rootPlayer, opponent, root) -
      deedPotentialForPlayer(
        state,
        district,
        opponent,
        rootPlayer,
        opponentState
      );
  }
  return clamp(diff / Math.max(1, state.districts.length), -1, 1);
}

function deedPotentialForPlayer(
  state: GameState,
  district: DistrictState,
  playerId: PlayerId,
  opponentId: PlayerId,
  player: PlayerState
): number {
  const stack = district.stacks[playerId];
  const deed = stack.deed;
  if (!deed) {
    return 0;
  }

  const card = findProperty(deed.cardId);
  if (!card) {
    return 0;
  }

  const target = developmentCost(card);
  if (target <= 0) {
    return 0;
  }

  const remaining = Math.max(0, target - deed.progress);
  const progressRatio = clamp(deed.progress / target, 0, 1);
  const progressWeight = deedProgressWeight(progressRatio, remaining);
  const accessMultiplier = deedAccessMultiplier(
    state,
    playerId,
    player,
    card.suits
  );
  const ownCurrentScore = districtScore(stack);
  const opponentScore = districtScore(district.stacks[opponentId]);
  const ownCompletedScore = districtScore({
    developed: [...stack.developed, deed.cardId],
  });
  const currentMargin = ownCurrentScore - opponentScore;
  const completedMargin = ownCompletedScore - opponentScore;
  const controlImpact = deedControlImpact(currentMargin, completedMargin);
  const rankImpact = card.rank / 9;

  return (
    progressWeight *
    accessMultiplier *
    (0.78 * controlImpact + 0.22 * rankImpact)
  );
}

function deedProgressWeight(progressRatio: number, remaining: number): number {
  if (remaining <= 0) {
    return 1;
  }
  if (remaining <= 1) {
    return 0.9;
  }
  if (remaining <= 2) {
    return 0.75;
  }
  if (progressRatio >= 0.75) {
    return 0.5;
  }
  if (progressRatio >= 0.5) {
    return 0.28;
  }
  return 0.04;
}

function deedControlImpact(
  currentMargin: number,
  completedMargin: number
): number {
  if (completedMargin > 0 && currentMargin <= 0) {
    return 1;
  }
  if (completedMargin === 0 && currentMargin < 0) {
    return 0.45;
  }
  if (currentMargin > 0 && completedMargin > currentMargin) {
    return 0.35;
  }
  if (currentMargin > 0) {
    return 0.25;
  }
  if (completedMargin > currentMargin) {
    return 0.15;
  }
  return 0.05;
}

function deedAccessMultiplier(
  state: GameState,
  playerId: PlayerId,
  player: PlayerState,
  suits: readonly Suit[]
): number {
  const crownCounts = crownSuitCounts(player);
  const missingSuits = suits.filter(
    (suit) =>
      player.resources[suit] <= 0 &&
      (crownCounts[suit] ?? 0) <= 0 &&
      !playerHandHasSuit(player, suit) &&
      !playerDevelopedBoardHasSuit(state, playerId, suit)
  ).length;
  return Math.max(0.35, 1 - missingSuits * 0.25);
}

function resourceQualityTerm(
  state: GameState,
  rootPlayer: PlayerId,
  opponent: PlayerId
): number {
  const root = requiredPlayerState(state, rootPlayer).resources;
  const opponentResources = requiredPlayerState(state, opponent).resources;
  const coverageDiff =
    (suitCoverage(root) - suitCoverage(opponentResources)) / SUITS.length;
  const resourceTotalTerm = Math.tanh(
    (resourceTotal(root) - resourceTotal(opponentResources)) / 12
  );
  const taxExposureTerm = Math.tanh(
    (taxExposure(opponentResources) - taxExposure(root)) / 6
  );
  const contextualBankTerm = Math.tanh(
    (resourceBankValueV2(state, rootPlayer) -
      resourceBankValueV2(state, opponent)) /
      3
  );
  return clamp(
    0.35 * coverageDiff +
      0.18 * resourceTotalTerm +
      0.17 * taxExposureTerm +
      0.3 * contextualBankTerm,
    -1,
    1
  );
}

function developedRankTotal(stack: DistrictStack): number {
  return stack.developed.reduce((total, cardId) => {
    const card = findProperty(cardId);
    return total + (card?.rank ?? 0);
  }, 0);
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

function playerHandHasSuit(player: PlayerState, suit: Suit): boolean {
  return player.hand.some((cardId) =>
    findProperty(cardId)?.suits.includes(suit)
  );
}

function playerDevelopedBoardHasSuit(
  state: GameState,
  playerId: PlayerId,
  suit: Suit
): boolean {
  return state.districts.some((district) =>
    district.stacks[playerId].developed.some((cardId) =>
      findProperty(cardId)?.suits.includes(suit)
    )
  );
}

function suitCoverage(resources: ResourcePool): number {
  return SUITS.filter((suit) => resources[suit] > 0).length;
}

function resourceTotal(resources: ResourcePool): number {
  return SUITS.reduce((total, suit) => total + resources[suit], 0);
}

function taxExposure(resources: ResourcePool): number {
  return SUITS.reduce(
    (total, suit) => total + Math.max(0, resources[suit] - 1),
    0
  );
}

function requiredPlayerState(
  state: GameState,
  playerId: PlayerId
): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (!player) {
    throw new Error(`Search state evaluator missing player ${playerId}.`);
  }
  return player;
}

function isLateGame(state: GameState): boolean {
  return (
    (state.finalTurnsRemaining ?? 0) > 0 ||
    (state.deck.reshuffles >= 1 &&
      state.deck.draw.length <= LATE_GAME_DRAW_COUNT)
  );
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
