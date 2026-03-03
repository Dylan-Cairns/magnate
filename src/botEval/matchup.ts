import { performance } from 'node:perf_hooks';

import type { PlayerId } from '../engine/types';
import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import type { ActionPolicy } from '../policies/types';
import { playGame, type RuntimeBot } from './playGame';
import { summarizeLatencies, wilsonInterval } from './stats';
import type {
  HeadToHeadConfig,
  HeadToHeadRun,
  HeadToHeadSummary,
  PlayedGame,
} from './types';

export interface HeadToHeadDependencies {
  createPolicy?: (spec: BotSpec) => ActionPolicy;
  now?: () => number;
}

export async function runHeadToHead(
  config: HeadToHeadConfig,
  dependencies: HeadToHeadDependencies = {}
): Promise<HeadToHeadRun> {
  validateHeadToHeadConfig(config);

  const createPolicy = dependencies.createPolicy ?? createPolicyFromBotSpec;
  const now = dependencies.now ?? (() => performance.now());
  const candidate = createRuntimeBot(config.candidate, createPolicy);
  const opponent = createRuntimeBot(config.opponent, createPolicy);
  const games: PlayedGame[] = [];
  const startedAt = now();

  for (let pairIndex = 0; pairIndex < config.gamesPerSide; pairIndex += 1) {
    const pairNumber = pairIndex + 1;
    const seed = `${config.seedPrefix}-${String(pairNumber).padStart(4, '0')}`;
    const firstPlayer: PlayerId = pairIndex % 2 === 0 ? 'PlayerA' : 'PlayerB';

    games.push(
      await playGame({
        gameId: `pair-${String(pairNumber).padStart(4, '0')}-candidate-as-a`,
        seed,
        firstPlayer,
        botBySeat: {
          PlayerA: candidate,
          PlayerB: opponent,
        },
        maxDecisions: config.maxDecisionsPerGame,
        now,
      })
    );
    games.push(
      await playGame({
        gameId: `pair-${String(pairNumber).padStart(4, '0')}-candidate-as-b`,
        seed,
        firstPlayer,
        botBySeat: {
          PlayerA: opponent,
          PlayerB: candidate,
        },
        maxDecisions: config.maxDecisionsPerGame,
        now,
      })
    );
  }

  return {
    config: structuredClone(config),
    summary: summarizeHeadToHead(config, games, now() - startedAt),
    games,
  };
}

export function validateHeadToHeadConfig(config: HeadToHeadConfig): void {
  if (config.schemaVersion !== 1) {
    throw new Error(
      `Unsupported head-to-head schemaVersion=${String(config.schemaVersion)}.`
    );
  }
  if (config.runLabel.trim() === '') {
    throw new Error('runLabel must be a non-empty string.');
  }
  if (config.seedPrefix.trim() === '') {
    throw new Error('seedPrefix must be a non-empty string.');
  }
  if (!Number.isInteger(config.gamesPerSide) || config.gamesPerSide <= 0) {
    throw new Error('gamesPerSide must be a positive integer.');
  }
  if (
    config.maxDecisionsPerGame !== undefined &&
    (!Number.isInteger(config.maxDecisionsPerGame) ||
      config.maxDecisionsPerGame <= 0)
  ) {
    throw new Error('maxDecisionsPerGame must be a positive integer.');
  }
  if (config.candidate.id === config.opponent.id) {
    throw new Error('candidate and opponent bot ids must be distinct.');
  }
}

function createRuntimeBot(
  spec: BotSpec,
  createPolicy: (spec: BotSpec) => ActionPolicy
): RuntimeBot {
  return {
    spec,
    policy: createPolicy(spec),
  };
}

function summarizeHeadToHead(
  config: HeadToHeadConfig,
  games: readonly PlayedGame[],
  elapsedMs: number
): HeadToHeadSummary {
  const candidateId = config.candidate.id;
  const opponentId = config.opponent.id;
  let candidateWins = 0;
  let opponentWins = 0;
  let draws = 0;
  let candidateWinsAsPlayerA = 0;
  let candidateWinsAsPlayerB = 0;
  let candidateWinsMovingFirst = 0;
  let candidateWinsMovingSecond = 0;
  let candidateGamesAsPlayerA = 0;
  let candidateGamesAsPlayerB = 0;
  let candidateGamesMovingFirst = 0;
  let candidateGamesMovingSecond = 0;
  let turnTotal = 0;
  const finalScoreDeciders: HeadToHeadSummary['finalScoreDeciders'] = {
    districts: 0,
    'rank-total': 0,
    resources: 0,
    draw: 0,
  };
  const latencyValuesByBotId = new Map<string, number[]>();

  for (const game of games) {
    const candidateSeat = seatForBot(game, candidateId);
    const candidateWon = game.finalScore.winner === candidateSeat;
    const candidateMovedFirst = candidateSeat === game.firstPlayer;

    if (game.finalScore.winner === 'Draw') {
      draws += 1;
    } else if (candidateWon) {
      candidateWins += 1;
    } else {
      opponentWins += 1;
    }

    if (candidateSeat === 'PlayerA') {
      candidateGamesAsPlayerA += 1;
      candidateWinsAsPlayerA += candidateWon ? 1 : 0;
    } else {
      candidateGamesAsPlayerB += 1;
      candidateWinsAsPlayerB += candidateWon ? 1 : 0;
    }

    if (candidateMovedFirst) {
      candidateGamesMovingFirst += 1;
      candidateWinsMovingFirst += candidateWon ? 1 : 0;
    } else {
      candidateGamesMovingSecond += 1;
      candidateWinsMovingSecond += candidateWon ? 1 : 0;
    }

    finalScoreDeciders[game.finalScore.decidedBy] += 1;
    turnTotal += game.turns;
    for (const decision of game.transcript) {
      const existing = latencyValuesByBotId.get(decision.botId) ?? [];
      existing.push(decision.latencyMs);
      latencyValuesByBotId.set(decision.botId, existing);
    }
  }

  const totalGames = games.length;
  if (totalGames === 0) {
    throw new Error('Cannot summarize an empty head-to-head run.');
  }

  return {
    gamesPerSide: config.gamesPerSide,
    totalGames,
    candidateId,
    opponentId,
    candidateWins,
    opponentWins,
    draws,
    candidateWinRate: candidateWins / totalGames,
    candidateWinRateCi95: wilsonInterval(candidateWins, totalGames),
    candidateWinRateAsPlayerA: candidateWinsAsPlayerA / candidateGamesAsPlayerA,
    candidateWinRateAsPlayerB: candidateWinsAsPlayerB / candidateGamesAsPlayerB,
    candidateWinRateMovingFirst:
      candidateWinsMovingFirst / candidateGamesMovingFirst,
    candidateWinRateMovingSecond:
      candidateWinsMovingSecond / candidateGamesMovingSecond,
    sideGap: Math.abs(
      candidateWinsAsPlayerA / candidateGamesAsPlayerA -
        candidateWinsAsPlayerB / candidateGamesAsPlayerB
    ),
    averageTurns: turnTotal / totalGames,
    elapsedMs,
    gamesPerMinute: elapsedMs > 0 ? (totalGames * 60_000) / elapsedMs : 0,
    finalScoreDeciders,
    latencyByBotId: Object.fromEntries(
      [candidateId, opponentId].map((botId) => [
        botId,
        summarizeLatencies(latencyValuesByBotId.get(botId) ?? []),
      ])
    ),
  };
}

function seatForBot(game: PlayedGame, botId: string): PlayerId {
  if (game.botBySeat.PlayerA === botId) {
    return 'PlayerA';
  }
  if (game.botBySeat.PlayerB === botId) {
    return 'PlayerB';
  }
  throw new Error(`Game ${game.gameId} does not include bot ${botId}.`);
}
