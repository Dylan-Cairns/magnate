import { performance } from 'node:perf_hooks';

import type { PlayerId } from '../engine/types';
import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import type { ActionPolicy } from '../policies/types';
import {
  playGame,
  type PlayGameHeartbeat,
  type RuntimeBot,
} from './playGame';
import type { HeadToHeadConfig, PlayedGame } from './types';

export interface PairedSeedJob {
  pairIndex: number;
  pairNumber: number;
  seed: string;
  firstPlayer: PlayerId;
}

export interface PairedSeedResult {
  pairIndex: number;
  games: [PlayedGame, PlayedGame];
}

export interface RuntimePairBots {
  candidate: RuntimeBot;
  opponent: RuntimeBot;
}

export interface PlayPairedSeedOptions {
  config: HeadToHeadConfig;
  job: PairedSeedJob;
  bots: RuntimePairBots;
  now?: () => number;
  progressIntervalMs?: number;
  onHeartbeat?: (heartbeat: PlayGameHeartbeat) => void;
  onGameCompleted?: (game: PlayedGame) => void;
}

export function createPairedSeedJobs(config: HeadToHeadConfig): PairedSeedJob[] {
  return Array.from({ length: config.gamesPerSide }, (_, pairIndex) =>
    createPairedSeedJob(config, pairIndex)
  );
}

export function createRuntimePairBots(
  config: HeadToHeadConfig,
  createPolicy: (spec: BotSpec) => ActionPolicy = createPolicyFromBotSpec
): RuntimePairBots {
  return {
    candidate: createRuntimeBot(config.candidate, createPolicy),
    opponent: createRuntimeBot(config.opponent, createPolicy),
  };
}

export async function playPairedSeed({
  config,
  job,
  bots,
  now = () => performance.now(),
  progressIntervalMs = 0,
  onHeartbeat,
  onGameCompleted,
}: PlayPairedSeedOptions): Promise<PairedSeedResult> {
  const pairId = String(job.pairNumber).padStart(4, '0');
  const candidateAsA = await playGame({
    gameId: `pair-${pairId}-candidate-as-a`,
    seed: job.seed,
    firstPlayer: job.firstPlayer,
    botBySeat: {
      PlayerA: bots.candidate,
      PlayerB: bots.opponent,
    },
    maxDecisions: config.maxDecisionsPerGame,
    now,
    progressIntervalMs,
    onHeartbeat,
  });
  onGameCompleted?.(candidateAsA);
  const candidateAsB = await playGame({
    gameId: `pair-${pairId}-candidate-as-b`,
    seed: job.seed,
    firstPlayer: job.firstPlayer,
    botBySeat: {
      PlayerA: bots.opponent,
      PlayerB: bots.candidate,
    },
    maxDecisions: config.maxDecisionsPerGame,
    now,
    progressIntervalMs,
    onHeartbeat,
  });
  onGameCompleted?.(candidateAsB);
  return {
    pairIndex: job.pairIndex,
    games: [candidateAsA, candidateAsB],
  };
}

function createPairedSeedJob(
  config: HeadToHeadConfig,
  pairIndex: number
): PairedSeedJob {
  const pairNumber = pairIndex + 1;
  return {
    pairIndex,
    pairNumber,
    seed: `${config.seedPrefix}-${String(pairNumber).padStart(4, '0')}`,
    firstPlayer: pairIndex % 2 === 0 ? 'PlayerA' : 'PlayerB',
  };
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
