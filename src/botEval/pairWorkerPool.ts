import { fork, type ChildProcess } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import type { PlayGameHeartbeat } from './playGame';
import type { PairedSeedJob, PairedSeedResult } from './pair';
import type {
  PairWorkerRequest,
  PairWorkerResponse,
} from './pairWorkerProtocol';
import type { HeadToHeadConfig, PlayedGame } from './types';

export interface RunPairedSeedJobsInChildPoolOptions {
  config: HeadToHeadConfig;
  jobs: readonly PairedSeedJob[];
  workers: number;
  progressIntervalMs: number;
  onHeartbeat?: (
    workerId: number,
    pairIndex: number,
    heartbeat: PlayGameHeartbeat
  ) => void;
  onGameCompleted?: (
    workerId: number,
    pairIndex: number,
    game: PlayedGame
  ) => void;
  onPairCompleted?: (
    workerId: number,
    result: PairedSeedResult
  ) => void;
}

interface PoolWorker {
  id: number;
  process: ChildProcess;
  activePairIndex?: number;
}

export async function runPairedSeedJobsInChildPool({
  config,
  jobs,
  workers,
  progressIntervalMs,
  onHeartbeat,
  onGameCompleted,
  onPairCompleted,
}: RunPairedSeedJobsInChildPoolOptions): Promise<PairedSeedResult[]> {
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new Error('workers must be a positive integer.');
  }
  if (jobs.length === 0) {
    return [];
  }

  return new Promise((resolve, reject) => {
    const queue = [...jobs];
    const results: PairedSeedResult[] = [];
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

    function send(worker: PoolWorker, request: PairWorkerRequest): void {
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
      resolve(results.sort((left, right) => left.pairIndex - right.pairIndex));
    }

    function dispatch(worker: PoolWorker): void {
      const job = queue.shift();
      if (!job) {
        if (
          results.length === jobs.length &&
          pool.every((entry) => entry.activePairIndex === undefined)
        ) {
          complete();
        }
        return;
      }
      worker.activePairIndex = job.pairIndex;
      send(worker, { type: 'run-pair', job });
    }

    function handleMessage(
      worker: PoolWorker,
      response: PairWorkerResponse
    ): void {
      switch (response.type) {
        case 'ready':
          dispatch(worker);
          return;
        case 'heartbeat':
          onHeartbeat?.(worker.id, response.pairIndex, response.heartbeat);
          return;
        case 'game-completed':
          onGameCompleted?.(worker.id, response.pairIndex, response.game);
          return;
        case 'pair-completed':
          if (worker.activePairIndex !== response.result.pairIndex) {
            fail(
              new Error(
                `Pair worker ${String(worker.id)} completed unexpected pair ${String(response.result.pairIndex)}.`
              )
            );
            return;
          }
          worker.activePairIndex = undefined;
          results.push(response.result);
          onPairCompleted?.(worker.id, response.result);
          dispatch(worker);
          return;
        case 'error':
          fail(
            new Error(
              `Pair worker ${String(worker.id)} failed${response.pairIndex === undefined ? '' : ` on pair ${String(response.pairIndex + 1)}`}: ${response.message}`
            )
          );
          return;
      }
    }

    const poolSize = Math.min(workers, jobs.length);
    for (let index = 0; index < poolSize; index += 1) {
      const child = fork(
        fileURLToPath(new URL('./pairWorker.ts', import.meta.url)),
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
      child.on('message', (message: PairWorkerResponse) => {
        handleMessage(worker, message);
      });
      child.on('error', fail);
      child.on('exit', (code, signal) => {
        if (!shuttingDown) {
          fail(
            new Error(
              `Pair worker ${String(worker.id)} exited unexpectedly (code=${String(code)}, signal=${String(signal)}).`
            )
          );
        }
      });
      send(worker, {
        type: 'initialize',
        config,
        progressIntervalMs,
      });
    }
  });
}
