import { GamePhase } from './values';
import { findDevelopableCard } from './stateHelpers';
import {
  type DistrictId,
  type DistrictStack,
  type FinalScore,
  type GameState,
  PlayerId,
  Winner,
  WinnerDecider,
} from './types';

export interface DistrictScoreMargin {
  districtId: DistrictId;
  margin: number;
}

export interface PlayerScoreMargins {
  districtPointMargin: number;
  districtScoreMargins: readonly DistrictScoreMargin[];
  districtScoreMarginTotal: number;
  rankTotalMargin: number;
  resourceMargin: number;
}

export function isTerminal(state: GameState): boolean {
  return state.phase === GamePhase.GameOver;
}

export function scoreGame(state: GameState): FinalScore {
  const districtWinners = districtWinnersByPlayer(state);
  const districtPoints = {
    [PlayerId.PlayerA]: districtWinners[PlayerId.PlayerA].length,
    [PlayerId.PlayerB]: districtWinners[PlayerId.PlayerB].length,
  };
  const rankTotals = createPlayerCounter();

  state.districts.forEach((district) => {
    const districtRankTotals = {
      [PlayerId.PlayerA]: rankTotal(district.stacks[PlayerId.PlayerA]),
      [PlayerId.PlayerB]: rankTotal(district.stacks[PlayerId.PlayerB]),
    };

    rankTotals[PlayerId.PlayerA] += districtRankTotals[PlayerId.PlayerA];
    rankTotals[PlayerId.PlayerB] += districtRankTotals[PlayerId.PlayerB];
  });

  const resourceTotals = {
    [PlayerId.PlayerA]: resourceTotal(state, PlayerId.PlayerA),
    [PlayerId.PlayerB]: resourceTotal(state, PlayerId.PlayerB),
  };

  const winner = decideWinner(districtPoints, rankTotals, resourceTotals);
  const decidedBy = winnerReason(districtPoints, rankTotals, resourceTotals);

  return {
    districtPoints,
    rankTotals,
    resourceTotals,
    winner,
    decidedBy,
  };
}

export function scoreLive(state: GameState): FinalScore {
  return scoreGame(state);
}

export function scoreMarginsForPlayer(
  state: GameState,
  playerId: PlayerId,
  finalScore: FinalScore = state.finalScore ?? scoreGame(state)
): PlayerScoreMargins {
  const opponent = otherPlayerId(playerId);
  const districtScoreMargins = state.districts.map((district) => ({
    districtId: district.id,
    margin:
      districtScore(district.stacks[playerId]) -
      districtScore(district.stacks[opponent]),
  }));

  return {
    districtPointMargin:
      finalScore.districtPoints[playerId] - finalScore.districtPoints[opponent],
    districtScoreMargins,
    districtScoreMarginTotal: districtScoreMargins.reduce(
      (total, entry) => total + entry.margin,
      0
    ),
    rankTotalMargin:
      finalScore.rankTotals[playerId] - finalScore.rankTotals[opponent],
    resourceMargin:
      finalScore.resourceTotals[playerId] - finalScore.resourceTotals[opponent],
  };
}

export function districtWinnersByPlayer(
  state: GameState
): Record<PlayerId, DistrictId[]> {
  const winners = createPlayerDistrictList();

  state.districts.forEach((district) => {
    const districtScores = {
      [PlayerId.PlayerA]: districtScore(district.stacks[PlayerId.PlayerA]),
      [PlayerId.PlayerB]: districtScore(district.stacks[PlayerId.PlayerB]),
    };

    if (districtScores[PlayerId.PlayerA] > districtScores[PlayerId.PlayerB]) {
      winners[PlayerId.PlayerA].push(district.id);
    } else if (
      districtScores[PlayerId.PlayerB] > districtScores[PlayerId.PlayerA]
    ) {
      winners[PlayerId.PlayerB].push(district.id);
    }
  });

  return winners;
}

export function districtScore(stack: DistrictStack): number {
  const properties = developedProperties(stack);
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

function rankTotal(stack: DistrictStack): number {
  return developedProperties(stack).reduce(
    (sum, property) => sum + property.rank,
    0
  );
}

function developedProperties(stack: DistrictStack) {
  return stack.developed.map(findDevelopableCard).filter(isDefined);
}

function resourceTotal(state: GameState, playerId: PlayerId): number {
  const player = state.players.find((entry) => entry.id === playerId);
  if (!player) {
    return 0;
  }
  return Object.values(player.resources).reduce((sum, value) => sum + value, 0);
}

function decideWinner(
  districtPoints: Record<PlayerId, number>,
  rankTotals: Record<PlayerId, number>,
  resourceTotals: Record<PlayerId, number>
): Winner {
  if (districtPoints[PlayerId.PlayerA] !== districtPoints[PlayerId.PlayerB]) {
    return districtPoints[PlayerId.PlayerA] > districtPoints[PlayerId.PlayerB]
      ? PlayerId.PlayerA
      : PlayerId.PlayerB;
  }
  if (rankTotals[PlayerId.PlayerA] !== rankTotals[PlayerId.PlayerB]) {
    return rankTotals[PlayerId.PlayerA] > rankTotals[PlayerId.PlayerB]
      ? PlayerId.PlayerA
      : PlayerId.PlayerB;
  }
  if (resourceTotals[PlayerId.PlayerA] !== resourceTotals[PlayerId.PlayerB]) {
    return resourceTotals[PlayerId.PlayerA] > resourceTotals[PlayerId.PlayerB]
      ? PlayerId.PlayerA
      : PlayerId.PlayerB;
  }
  return Winner.Draw;
}

function winnerReason(
  districtPoints: Record<PlayerId, number>,
  rankTotals: Record<PlayerId, number>,
  resourceTotals: Record<PlayerId, number>
): WinnerDecider {
  if (districtPoints[PlayerId.PlayerA] !== districtPoints[PlayerId.PlayerB]) {
    return WinnerDecider.Districts;
  }
  if (rankTotals[PlayerId.PlayerA] !== rankTotals[PlayerId.PlayerB]) {
    return WinnerDecider.RankTotal;
  }
  if (resourceTotals[PlayerId.PlayerA] !== resourceTotals[PlayerId.PlayerB]) {
    return WinnerDecider.Resources;
  }
  return WinnerDecider.Draw;
}

function createPlayerCounter(): Record<PlayerId, number> {
  return {
    [PlayerId.PlayerA]: 0,
    [PlayerId.PlayerB]: 0,
  };
}

function createPlayerDistrictList(): Record<PlayerId, DistrictId[]> {
  return {
    [PlayerId.PlayerA]: [],
    [PlayerId.PlayerB]: [],
  };
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === PlayerId.PlayerA ? PlayerId.PlayerB : PlayerId.PlayerA;
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}
