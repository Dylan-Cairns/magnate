import { performance } from 'node:perf_hooks';

import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import type { GameState, PlayerId } from '../engine/types';
import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import {
  policyRandomForState,
  policyRandomSeedForState,
} from '../policies/policyRandom';
import {
  encodeActionCandidates,
  encodeObservation,
} from '../policies/trainingEncoding';
import type { ActionPolicy } from '../policies/types';
import type { RuntimeBot } from './playGame';
import type {
  CollectedTdReplayGame,
  TdReplayConfig,
  TdReplayOpponentSamplePayload,
  TdReplayRun,
  TdReplayValueTransitionPayload,
} from './types';

const DEFAULT_MAX_DECISIONS_PER_GAME = 10_000;

export type TdReplayProgress =
  | {
      type: 'game-heartbeat';
      gameNumber: number;
      totalGames: number;
      gameId: string;
      turn: number;
      decisions: number;
      elapsedMs: number;
    }
  | {
      type: 'game-completed';
      gameNumber: number;
      totalGames: number;
      game: CollectedTdReplayGame;
      elapsedMs: number;
      gamesPerMinute: number;
    };

export interface TdReplayDependencies {
  createPolicy?: (spec: BotSpec) => ActionPolicy;
  now?: () => number;
  progressIntervalMs?: number;
  onProgress?: (progress: TdReplayProgress) => void;
}

export interface TdReplayGamesResult {
  config: TdReplayConfig;
  elapsedMs: number;
}

export type TdReplayGameHandler = (
  game: CollectedTdReplayGame
) => Promise<void> | void;

interface CollectTdReplayGameOptions {
  gameId: string;
  seed: string;
  firstPlayer: PlayerId;
  botBySeat: Record<PlayerId, RuntimeBot>;
  maxDecisions: number;
  now: () => number;
  progressIntervalMs: number;
  onHeartbeat?: (heartbeat: {
    gameId: string;
    turn: number;
    decisions: number;
    elapsedMs: number;
  }) => void;
}

export async function collectTdReplay(
  config: TdReplayConfig,
  dependencies: TdReplayDependencies = {}
): Promise<TdReplayRun> {
  const games: CollectedTdReplayGame[] = [];
  const valueTransitions: TdReplayValueTransitionPayload[] = [];
  const opponentSamples: TdReplayOpponentSamplePayload[] = [];
  const result = await collectTdReplayGames(config, dependencies, (game) => {
    games.push(game);
    valueTransitions.push(...game.valueTransitions);
    opponentSamples.push(...game.opponentSamples);
  });

  return {
    config: structuredClone(config),
    games,
    valueTransitions,
    opponentSamples,
    elapsedMs: result.elapsedMs,
  };
}

export async function collectTdReplayGames(
  config: TdReplayConfig,
  dependencies: TdReplayDependencies = {},
  onGame: TdReplayGameHandler
): Promise<TdReplayGamesResult> {
  validateTdReplayConfig(config);

  const createPolicy = dependencies.createPolicy ?? createPolicyFromBotSpec;
  const now = dependencies.now ?? (() => performance.now());
  const progressIntervalMs = dependencies.progressIntervalMs ?? 0;
  if (!Number.isFinite(progressIntervalMs) || progressIntervalMs < 0) {
    throw new Error('progressIntervalMs must be a finite number >= 0.');
  }

  const startedAt = now();
  const botBySeat: Record<PlayerId, RuntimeBot> = {
    PlayerA: createRuntimeBot(config.playerA, createPolicy),
    PlayerB: createRuntimeBot(config.playerB, createPolicy),
  };

  for (let gameIndex = 0; gameIndex < config.games; gameIndex += 1) {
    const gameNumber = gameIndex + 1;
    const gameId = `game-${String(gameIndex).padStart(4, '0')}`;
    const seed = `${config.seedPrefix}-${String(gameIndex)}`;
    const game = await collectTdReplayGame({
      gameId,
      seed,
      firstPlayer: gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB',
      botBySeat,
      maxDecisions:
        config.maxDecisionsPerGame ?? DEFAULT_MAX_DECISIONS_PER_GAME,
      now,
      progressIntervalMs,
      onHeartbeat(heartbeat) {
        dependencies.onProgress?.({
          type: 'game-heartbeat',
          gameNumber,
          totalGames: config.games,
          ...heartbeat,
        });
      },
    });
    await onGame(game);

    const elapsedMs = now() - startedAt;
    const gamesPerMinute =
      elapsedMs > 0 ? (gameNumber * 60_000) / elapsedMs : 0;
    dependencies.onProgress?.({
      type: 'game-completed',
      gameNumber,
      totalGames: config.games,
      game,
      elapsedMs,
      gamesPerMinute,
    });
  }

  return {
    config: structuredClone(config),
    elapsedMs: now() - startedAt,
  };
}

export function validateTdReplayConfig(config: TdReplayConfig): void {
  if (config.schemaVersion !== 1) {
    throw new Error(
      `Unsupported TD replay schemaVersion=${String(config.schemaVersion)}.`
    );
  }
  if (config.runLabel.trim() === '') {
    throw new Error('runLabel must be a non-empty string.');
  }
  if (config.seedPrefix.trim() === '') {
    throw new Error('seedPrefix must be a non-empty string.');
  }
  if (!Number.isInteger(config.games) || config.games <= 0) {
    throw new Error('games must be a positive integer.');
  }
  if (
    config.maxDecisionsPerGame !== undefined &&
    (!Number.isInteger(config.maxDecisionsPerGame) ||
      config.maxDecisionsPerGame <= 0)
  ) {
    throw new Error('maxDecisionsPerGame must be a positive integer.');
  }
  rejectUnsupportedBotSpec(config.playerA, 'playerA');
  rejectUnsupportedBotSpec(config.playerB, 'playerB');
}

async function collectTdReplayGame({
  gameId,
  seed,
  firstPlayer,
  botBySeat,
  maxDecisions,
  now,
  progressIntervalMs,
  onHeartbeat,
}: CollectTdReplayGameOptions): Promise<CollectedTdReplayGame> {
  const gameStartedAt = now();
  let nextHeartbeatAt = gameStartedAt + progressIntervalMs;
  let state = createSession(seed, firstPlayer);
  const decisions: CollectedTdReplayGame['decisions'] = [];
  const valueTransitions: TdReplayValueTransitionPayload[] = [];
  const opponentSamples: TdReplayOpponentSamplePayload[] = [];
  const pendingObservationByPlayer: Record<PlayerId, number[] | null> = {
    PlayerA: null,
    PlayerB: null,
  };
  const pendingTimestepByPlayer: Record<PlayerId, number | null> = {
    PlayerA: null,
    PlayerB: null,
  };
  const nextTimestepByPlayer: Record<PlayerId, number> = {
    PlayerA: 0,
    PlayerB: 0,
  };

  function emitHeartbeatIfDue(): void {
    if (!onHeartbeat || progressIntervalMs === 0) {
      return;
    }
    const currentTime = now();
    if (currentTime < nextHeartbeatAt) {
      return;
    }
    onHeartbeat({
      gameId,
      turn: state.turn,
      decisions: decisions.length,
      elapsedMs: currentTime - gameStartedAt,
    });
    nextHeartbeatAt = currentTime + progressIntervalMs;
  }

  while (!isTerminal(state)) {
    if (decisions.length >= maxDecisions) {
      throw new Error(
        `Game ${gameId} exceeded maxDecisions=${String(maxDecisions)}.`
      );
    }

    const activePlayerId = activePlayerIdForState(state, gameId);
    const bot = botBySeat[activePlayerId];
    const actions = toKeyedActions(
      legalActionsForDecisionPlayer(state, activePlayerId)
    ).map(
      (entry) => entry.action
    );
    if (actions.length === 0) {
      throw new Error(
        `Game ${gameId} has no legal actions for ${activePlayerId} during ${state.phase}.`
      );
    }

    const view = toDecisionPlayerView(state, activePlayerId);
    const observation = encodeObservation(view);
    const priorObservation = pendingObservationByPlayer[activePlayerId];
    const priorTimestep = pendingTimestepByPlayer[activePlayerId];
    if (priorObservation !== null) {
      if (priorTimestep === null) {
        throw new Error(
          `Game ${gameId} missing pending timestep for ${activePlayerId}.`
        );
      }
      valueTransitions.push({
        observation: priorObservation,
        reward: 0.0,
        done: false,
        nextObservation: observation,
        playerId: activePlayerId,
        episodeId: seed,
        timestep: priorTimestep,
      });
    }

    const actionFeatures = encodeActionCandidates(actions);
    const selected = await bot.policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: policyRandomForState(state, bot.spec.id),
      randomSeed: policyRandomSeedForState(state, bot.spec.id),
      onProgress: emitHeartbeatIfDue,
    });
    if (!selected) {
      throw new Error(
        `Bot ${bot.spec.id} did not select an action in game ${gameId}.`
      );
    }

    const actionKey = actionStableKey(selected);
    const actionIndex = actions.findIndex(
      (action) => actionStableKey(action) === actionKey
    );
    if (actionIndex < 0) {
      throw new Error(
        `Bot ${bot.spec.id} selected illegal action ${actionKey} in game ${gameId}.`
      );
    }
    const indexedActionKey = actionStableKey(actions[actionIndex]);

    opponentSamples.push({
      observation,
      actionFeatures,
      actionIndex,
      playerId: activePlayerId,
    });
    decisions.push({
      decisionIndex: decisions.length,
      turn: state.turn,
      phase: state.phase,
      activePlayerId,
      botId: bot.spec.id,
      actionKey,
      actionIndex,
      indexedActionKey,
      legalActionCount: actions.length,
    });

    pendingObservationByPlayer[activePlayerId] = observation;
    pendingTimestepByPlayer[activePlayerId] =
      nextTimestepByPlayer[activePlayerId];
    nextTimestepByPlayer[activePlayerId] += 1;
    state = stepToDecision(state, actions[actionIndex]);
    emitHeartbeatIfDue();
  }

  if (!state.finalScore) {
    throw new Error(`Terminal game ${gameId} is missing its final score.`);
  }

  for (const playerId of ['PlayerA', 'PlayerB'] as const) {
    const pendingObservation = pendingObservationByPlayer[playerId];
    if (pendingObservation === null) {
      continue;
    }
    const timestep = pendingTimestepByPlayer[playerId];
    if (timestep === null) {
      throw new Error(
        `Game ${gameId} missing terminal timestep for ${playerId}.`
      );
    }
    valueTransitions.push({
      observation: pendingObservation,
      reward: terminalReward(state.finalScore.winner, playerId),
      done: true,
      nextObservation: null,
      playerId,
      episodeId: seed,
      timestep,
    });
  }

  return {
    gameId,
    seed,
    firstPlayer,
    botBySeat: {
      PlayerA: botBySeat.PlayerA.spec.id,
      PlayerB: botBySeat.PlayerB.spec.id,
    },
    decisions,
    finalScore: structuredClone(state.finalScore),
    turns: state.turn,
    elapsedMs: now() - gameStartedAt,
    valueTransitions,
    opponentSamples,
  };
}

function activePlayerIdForState(state: GameState, gameId: string): PlayerId {
  const activePlayerId = decisionPlayerIdForState(state);
  if (activePlayerId !== 'PlayerA' && activePlayerId !== 'PlayerB') {
    throw new Error(`Game ${gameId} could not resolve its active player.`);
  }
  return activePlayerId;
}

function terminalReward(
  winner: 'PlayerA' | 'PlayerB' | 'Draw',
  playerId: PlayerId
): number {
  if (winner === 'Draw') {
    return 0.0;
  }
  return winner === playerId ? 1.0 : -1.0;
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

function rejectUnsupportedBotSpec(spec: BotSpec, label: string): void {
  if (spec.kind === 'td-search') {
    throw new Error(
      `${label}.kind td-search is not supported by collect-td-replay yet; use random, heuristic, or search.`
    );
  }
}
