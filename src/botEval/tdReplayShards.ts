import { randomUUID } from 'node:crypto';
import { fork, type ChildProcess } from 'node:child_process';
import { mkdir, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import type { FinalScore } from '../engine/types';
import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import {
  ACTION_FEATURE_DIM,
  ENCODING_VERSION,
  OBSERVATION_DIM,
} from '../policies/trainingEncoding';
import { collectGitMetadata } from './gitMetadata';
import { validateTdReplayConfig, type TdReplayProgress } from './tdReplay';
import {
  defaultTdReplayOutputDirectory,
  defaultTdReplayRunBaseName,
} from './tdReplayArtifacts';
import type {
  TdReplayShardPlan,
  TdReplayShardResult,
  TdReplayShardWorkerRequest,
  TdReplayShardWorkerResponse,
} from './tdReplayShardWorkerProtocol';
import type {
  GitMetadata,
  TdReplayConfig,
  TdReplaySummaryGame,
} from './types';

export const TD_REPLAY_SHARDED_SUMMARY_SCHEMA_VERSION = 1;
export const TD_REPLAY_SHARDED_ARTIFACT_TYPE = 'ts-td-replay-sharded';

export interface TdReplayShardExecution {
  requestedWorkers: number;
  workers: number;
  parallelUnit: 'contiguous-game-range';
  shards: number;
  shardGames?: number;
}

export interface TdReplayShardSummaryEntry {
  shardIndex: number;
  gameIndexStart: number;
  games: number;
  valuePath: string;
  opponentPath: string;
  summaryPath: string;
  results: {
    games: number;
    decisions: number;
    valueTransitions: number;
    opponentSamples: number;
    elapsedMs: number;
  };
}

export interface TdReplayShardedSummary {
  schemaVersion: typeof TD_REPLAY_SHARDED_SUMMARY_SCHEMA_VERSION;
  artifactType: typeof TD_REPLAY_SHARDED_ARTIFACT_TYPE;
  generatedAtUtc: string;
  policyRandomSchemeVersion: string;
  runtime: {
    nodeVersion: string;
  };
  git: GitMetadata;
  execution: TdReplayShardExecution;
  config: TdReplayConfig;
  encoding: {
    encodingVersion: number;
    observationDim: number;
    actionFeatureDim: number;
  };
  results: {
    games: number;
    winners: Record<FinalScore['winner'], number>;
    averageTurns: number;
    decisions: number;
    valueTransitions: number;
    opponentSamples: number;
    elapsedMs: number;
  };
  artifacts: {
    directory: string;
    shardsDirectory: string;
    summary: string;
  };
  shards: TdReplayShardSummaryEntry[];
  games: TdReplaySummaryGame[];
}

export interface CollectShardedTdReplayOptions {
  workers: number;
  shardGames?: number;
  cwd?: string;
  generatedAtUtc?: string;
  git?: GitMetadata;
  nodeVersion?: string;
  runBaseName?: string;
  progressIntervalMs?: number;
  onProgress?: (progress: TdReplayShardProgress) => void;
}

export type TdReplayShardProgress =
  | {
      type: 'sharded-started';
      requestedWorkers: number;
      workers: number;
      shards: number;
      shardGames?: number;
      games: number;
      runDirectory: string;
    }
  | {
      type: 'shard-started';
      shard: TdReplayShardPlan;
    }
  | {
      type: 'shard-progress';
      shard: TdReplayShardPlan;
      progress: TdReplayProgress;
    }
  | {
      type: 'shard-completed';
      shard: TdReplayShardPlan;
      result: TdReplayShardResult;
    }
  | {
      type: 'sharded-completed';
      summary: TdReplayShardedSummary;
    };

export interface WrittenShardedTdReplayArtifacts {
  summary: TdReplayShardedSummary;
  runDirectory: string;
  shardsDirectory: string;
  summaryPath: string;
}

interface PoolWorker {
  id: number;
  process: ChildProcess;
  activeShardIndex?: number;
}

export interface PlanContiguousTdReplayShardsOptions {
  games: number;
  workers: number;
  shardGames?: number;
}

export function planContiguousTdReplayShards(
  options: PlanContiguousTdReplayShardsOptions
): TdReplayShardPlan[] {
  const { games, workers, shardGames } = options;
  if (!Number.isInteger(games) || games <= 0) {
    throw new Error('games must be a positive integer.');
  }
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new Error('workers must be a positive integer.');
  }
  if (shardGames !== undefined) {
    if (!Number.isInteger(shardGames) || shardGames <= 0) {
      throw new Error('shardGames must be a positive integer.');
    }
    const shards: TdReplayShardPlan[] = [];
    for (
      let gameIndexStart = 0, shardIndex = 0;
      gameIndexStart < games;
      gameIndexStart += shardGames, shardIndex += 1
    ) {
      shards.push({
        shardIndex,
        gameIndexStart,
        games: Math.min(shardGames, games - gameIndexStart),
      });
    }
    return shards;
  }
  const shardCount = Math.min(games, workers);
  const baseGamesPerShard = Math.floor(games / shardCount);
  const remainder = games % shardCount;
  const shards: TdReplayShardPlan[] = [];
  let gameIndexStart = 0;
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    const shardGames = baseGamesPerShard + (shardIndex < remainder ? 1 : 0);
    shards.push({
      shardIndex,
      gameIndexStart,
      games: shardGames,
    });
    gameIndexStart += shardGames;
  }
  return shards;
}

export function defaultShardedTdReplayOutputDirectory(): string {
  return defaultTdReplayOutputDirectory();
}

export async function collectAndWriteShardedTdReplayArtifacts(
  config: TdReplayConfig,
  outputDirectory: string,
  options: CollectShardedTdReplayOptions
): Promise<WrittenShardedTdReplayArtifacts> {
  validateTdReplayConfig(config);
  if (!Number.isInteger(options.workers) || options.workers <= 0) {
    throw new Error('workers must be a positive integer.');
  }
  const generatedAtUtc = options.generatedAtUtc ?? new Date().toISOString();
  const git = options.git ?? collectGitMetadata(options.cwd);
  const nodeVersion = options.nodeVersion ?? process.version;
  const requestedWorkers = options.workers;
  const shards = planContiguousTdReplayShards({
    games: config.games,
    workers: requestedWorkers,
    shardGames: options.shardGames,
  });
  const workerCount = Math.min(requestedWorkers, shards.length);
  const execution: TdReplayShardExecution = {
    requestedWorkers,
    workers: workerCount,
    parallelUnit: 'contiguous-game-range',
    shards: shards.length,
    ...(options.shardGames === undefined
      ? {}
      : { shardGames: options.shardGames }),
  };
  const runBaseName =
    options.runBaseName ??
    defaultTdReplayRunBaseName(config.runLabel, generatedAtUtc);
  const runDirectory = path.join(outputDirectory, runBaseName);
  const shardsDirectory = path.join(runDirectory, 'shards');
  const summaryPath = path.join(runDirectory, 'summary.json');
  const progressIntervalMs = options.progressIntervalMs ?? 0;
  if (!Number.isFinite(progressIntervalMs) || progressIntervalMs < 0) {
    throw new Error('progressIntervalMs must be a finite number >= 0.');
  }

  await mkdir(shardsDirectory, { recursive: true });
  options.onProgress?.({
    type: 'sharded-started',
    requestedWorkers,
    workers: workerCount,
    shards: shards.length,
    ...(options.shardGames === undefined
      ? {}
      : { shardGames: options.shardGames }),
    games: config.games,
    runDirectory,
  });
  const startedAt = performance.now();
  const results = await runTdReplayShardJobsInChildPool({
    config,
    shards,
    workers: workerCount,
    gameIndexTotal: config.games,
    outputDirectory: shardsDirectory,
    progressIntervalMs,
    generatedAtUtc,
    git,
    nodeVersion,
    onShardStarted(shard) {
      options.onProgress?.({ type: 'shard-started', shard });
    },
    onProgress(shard, progress) {
      options.onProgress?.({ type: 'shard-progress', shard, progress });
    },
    onShardCompleted(shard, result) {
      options.onProgress?.({ type: 'shard-completed', shard, result });
    },
  });
  const elapsedMs = performance.now() - startedAt;
  const summary = createShardedTdReplaySummary({
    config,
    execution,
    generatedAtUtc,
    git,
    nodeVersion,
    runDirectory,
    shardsDirectory,
    summaryPath,
    results,
    elapsedMs,
  });
  await writeAtomic(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  options.onProgress?.({ type: 'sharded-completed', summary });

  return {
    summary,
    runDirectory,
    shardsDirectory,
    summaryPath,
  };
}

function createShardedTdReplaySummary({
  config,
  execution,
  generatedAtUtc,
  git,
  nodeVersion,
  runDirectory,
  shardsDirectory,
  summaryPath,
  results,
  elapsedMs,
}: {
  config: TdReplayConfig;
  execution: TdReplayShardExecution;
  generatedAtUtc: string;
  git: GitMetadata;
  nodeVersion: string;
  runDirectory: string;
  shardsDirectory: string;
  summaryPath: string;
  results: readonly TdReplayShardResult[];
  elapsedMs: number;
}): TdReplayShardedSummary {
  const winners: Record<FinalScore['winner'], number> = {
    PlayerA: 0,
    PlayerB: 0,
    Draw: 0,
  };
  const games = results
    .flatMap((result) => result.written.summary.games)
    .sort((left, right) => left.gameId.localeCompare(right.gameId));
  let turnTotal = 0;
  let decisionTotal = 0;
  let valueTransitionTotal = 0;
  let opponentSampleTotal = 0;
  for (const game of games) {
    winners[game.winner] += 1;
    turnTotal += game.turns;
    decisionTotal += game.decisions;
    valueTransitionTotal += game.valueTransitions;
    opponentSampleTotal += game.opponentSamples;
  }

  return {
    schemaVersion: TD_REPLAY_SHARDED_SUMMARY_SCHEMA_VERSION,
    artifactType: TD_REPLAY_SHARDED_ARTIFACT_TYPE,
    generatedAtUtc,
    policyRandomSchemeVersion: POLICY_RANDOM_SCHEME_VERSION,
    runtime: {
      nodeVersion,
    },
    git,
    execution,
    config: structuredClone(config),
    encoding: {
      encodingVersion: ENCODING_VERSION,
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
    },
    results: {
      games: games.length,
      winners,
      averageTurns: games.length > 0 ? turnTotal / games.length : 0,
      decisions: decisionTotal,
      valueTransitions: valueTransitionTotal,
      opponentSamples: opponentSampleTotal,
      elapsedMs,
    },
    artifacts: {
      directory: runDirectory,
      shardsDirectory,
      summary: summaryPath,
    },
    shards: results.map((result) => ({
      shardIndex: result.shard.shardIndex,
      gameIndexStart: result.shard.gameIndexStart,
      games: result.shard.games,
      valuePath: result.written.valuePath,
      opponentPath: result.written.opponentPath,
      summaryPath: result.written.summaryPath,
      results: {
        games: result.written.summary.results.games,
        decisions: result.written.summary.results.decisions,
        valueTransitions: result.written.summary.results.valueTransitions,
        opponentSamples: result.written.summary.results.opponentSamples,
        elapsedMs: result.written.summary.results.elapsedMs,
      },
    })),
    games,
  };
}

function runTdReplayShardJobsInChildPool({
  config,
  shards,
  workers,
  gameIndexTotal,
  outputDirectory,
  progressIntervalMs,
  generatedAtUtc,
  git,
  nodeVersion,
  onShardStarted,
  onProgress,
  onShardCompleted,
}: {
  config: TdReplayConfig;
  shards: readonly TdReplayShardPlan[];
  workers: number;
  gameIndexTotal: number;
  outputDirectory: string;
  progressIntervalMs: number;
  generatedAtUtc: string;
  git: GitMetadata;
  nodeVersion: string;
  onShardStarted?: (shard: TdReplayShardPlan) => void;
  onProgress?: (
    shard: TdReplayShardPlan,
    progress: TdReplayProgress
  ) => void;
  onShardCompleted?: (
    shard: TdReplayShardPlan,
    result: TdReplayShardResult
  ) => void;
}): Promise<TdReplayShardResult[]> {
  if (shards.length === 0) {
    return Promise.resolve([]);
  }
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new Error('workers must be a positive integer.');
  }

  return new Promise((resolve, reject) => {
    const queue = [...shards];
    const results: TdReplayShardResult[] = [];
    const pool: PoolWorker[] = [];
    let shuttingDown = false;
    let settled = false;

    function fail(error: unknown): void {
      if (settled) {
        return;
      }
      settled = true;
      shuttingDown = true;
      for (const worker of pool) {
        worker.process.kill();
      }
      reject(error instanceof Error ? error : new Error(String(error)));
    }

    function send(
      worker: PoolWorker,
      request: TdReplayShardWorkerRequest
    ): void {
      worker.process.send?.(request, (error) => {
        if (error) {
          fail(error);
        }
      });
    }

    function complete(): void {
      if (settled) {
        return;
      }
      settled = true;
      shuttingDown = true;
      for (const worker of pool) {
        send(worker, { type: 'shutdown' });
      }
      resolve(
        results.sort(
          (left, right) => left.shard.shardIndex - right.shard.shardIndex
        )
      );
    }

    function dispatch(worker: PoolWorker): void {
      const shard = queue.shift();
      if (!shard) {
        if (
          results.length === shards.length &&
          pool.every((entry) => entry.activeShardIndex === undefined)
        ) {
          complete();
        }
        return;
      }
      worker.activeShardIndex = shard.shardIndex;
      onShardStarted?.(shard);
      send(worker, {
        type: 'run-shard',
        config,
        shard,
        gameIndexTotal,
        outputDirectory,
        progressIntervalMs,
        generatedAtUtc,
        git,
        nodeVersion,
      });
    }

    function handleMessage(
      worker: PoolWorker,
      response: TdReplayShardWorkerResponse
    ): void {
      switch (response.type) {
        case 'ready':
          dispatch(worker);
          return;
        case 'progress': {
          const shard = shards.find(
            (entry) => entry.shardIndex === response.shardIndex
          );
          if (!shard) {
            fail(
              new Error(
                `TD replay shard worker ${String(worker.id)} reported unknown shard ${String(response.shardIndex)}.`
              )
            );
            return;
          }
          onProgress?.(shard, response.progress);
          return;
        }
        case 'shard-completed':
          if (worker.activeShardIndex !== response.result.shard.shardIndex) {
            fail(
              new Error(
                `TD replay shard worker ${String(worker.id)} completed unexpected shard ${String(response.result.shard.shardIndex)}.`
              )
            );
            return;
          }
          worker.activeShardIndex = undefined;
          results.push(response.result);
          onShardCompleted?.(response.result.shard, response.result);
          dispatch(worker);
          return;
        case 'error':
          fail(
            new Error(
              `TD replay shard worker ${String(worker.id)} failed${response.shardIndex === undefined ? '' : ` on shard ${String(response.shardIndex)}`}: ${response.message}`
            )
          );
          return;
      }
    }

    const poolSize = Math.min(workers, shards.length);
    for (let index = 0; index < poolSize; index += 1) {
      const child = fork(
        fileURLToPath(new URL('./tdReplayShardWorker.ts', import.meta.url)),
        [],
        {
          execArgv: ['--import', 'tsx'],
          stdio: ['ignore', 'ignore', 'inherit', 'ipc'],
        }
      );
      const worker: PoolWorker = {
        id: index + 1,
        process: child,
      };
      pool.push(worker);
      child.on('message', (message: TdReplayShardWorkerResponse) => {
        handleMessage(worker, message);
      });
      child.on('error', fail);
      child.on('exit', (code, signal) => {
        if (!shuttingDown) {
          fail(
            new Error(
              `TD replay shard worker ${String(worker.id)} exited unexpectedly (code=${String(code)}, signal=${String(signal)}).`
            )
          );
        }
      });
    }
  });
}

async function writeAtomic(
  targetPath: string,
  contents: string
): Promise<void> {
  const tempPath = `${targetPath}.${String(process.pid)}.${randomUUID()}.tmp`;
  await writeFile(tempPath, contents, 'utf8');
  await rename(tempPath, targetPath);
}
