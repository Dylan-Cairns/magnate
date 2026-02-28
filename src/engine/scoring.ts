import { findProperty } from './stateHelpers';
import type {
  DistrictStack,
  FinalScore,
  GameState,
  PlayerId,
  Winner,
  WinnerDecider,
} from './types';

export function isTerminal(state: GameState): boolean {
  return state.phase === 'GameOver';
}

export function scoreGame(state: GameState): FinalScore {
  const districtPoints = createPlayerCounter();
  const rankTotals = createPlayerCounter();

  state.districts.forEach((district) => {
    const districtScores = {
      PlayerA: districtScore(district.stacks.PlayerA),
      PlayerB: districtScore(district.stacks.PlayerB),
    };
    const districtRankTotals = {
      PlayerA: rankTotal(district.stacks.PlayerA),
      PlayerB: rankTotal(district.stacks.PlayerB),
    };

    rankTotals.PlayerA += districtRankTotals.PlayerA;
    rankTotals.PlayerB += districtRankTotals.PlayerB;

    if (districtScores.PlayerA > districtScores.PlayerB) {
      districtPoints.PlayerA += 1;
    } else if (districtScores.PlayerB > districtScores.PlayerA) {
      districtPoints.PlayerB += 1;
    }
  });

  const resourceTotals = {
    PlayerA: resourceTotal(state, 'PlayerA'),
    PlayerB: resourceTotal(state, 'PlayerB'),
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

function districtScore(stack: DistrictStack): number {
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
  return developedProperties(stack).reduce((sum, property) => sum + property.rank, 0);
}

function developedProperties(stack: DistrictStack) {
  return stack.developed.map(findProperty).filter(isDefined);
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
  if (districtPoints.PlayerA !== districtPoints.PlayerB) {
    return districtPoints.PlayerA > districtPoints.PlayerB ? 'PlayerA' : 'PlayerB';
  }
  if (rankTotals.PlayerA !== rankTotals.PlayerB) {
    return rankTotals.PlayerA > rankTotals.PlayerB ? 'PlayerA' : 'PlayerB';
  }
  if (resourceTotals.PlayerA !== resourceTotals.PlayerB) {
    return resourceTotals.PlayerA > resourceTotals.PlayerB ? 'PlayerA' : 'PlayerB';
  }
  return 'Draw';
}

function winnerReason(
  districtPoints: Record<PlayerId, number>,
  rankTotals: Record<PlayerId, number>,
  resourceTotals: Record<PlayerId, number>
): WinnerDecider {
  if (districtPoints.PlayerA !== districtPoints.PlayerB) {
    return 'districts';
  }
  if (rankTotals.PlayerA !== rankTotals.PlayerB) {
    return 'rank-total';
  }
  if (resourceTotals.PlayerA !== resourceTotals.PlayerB) {
    return 'resources';
  }
  return 'draw';
}

function createPlayerCounter(): Record<PlayerId, number> {
  return {
    PlayerA: 0,
    PlayerB: 0,
  };
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}
