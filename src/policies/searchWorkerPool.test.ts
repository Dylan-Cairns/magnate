import { describe, expect, it } from 'vitest';

import { createSession } from '../engine/session';
import {
  createSearchWorkerPool,
  type SearchWorkerPoolWorker,
} from './searchWorkerPool';
import type {
  SearchWorkerRequest,
  SearchWorkerResponse,
  SearchWorkerResult,
  SearchWorkerTask,
} from './searchWorkerProtocol';

describe('search worker pool', () => {
  it('distributes a batch across workers and resolves returned results', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 2,
      createWorker: () => pushWorker(workers),
    });

    const run = pool.runBatch([task(0), task(1), task(2)]);

    expect(workers).toHaveLength(2);
    expect(
      runBatchRequest(workers[0].messages[0]).tasks.map(
        (item) => item.visitIndex
      )
    ).toEqual([0, 2]);
    expect(
      runBatchRequest(workers[1].messages[0]).tasks.map(
        (item) => item.visitIndex
      )
    ).toEqual([1]);

    workers[1].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[1].messages[0]).requestId,
      results: [result(1)],
    });
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[0]).requestId,
      results: [result(0), result(2)],
    });

    await expect(run).resolves.toEqual([result(0), result(2), result(1)]);
  });

  it('rejects active work and closes workers when a worker fails', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 1,
      createWorker: () => pushWorker(workers),
    });

    const run = pool.runBatch([task(0)]);
    workers[0].fail('worker exploded');

    await expect(run).rejects.toThrow('worker exploded');
    expect(workers[0].terminated).toBe(true);
  });

  it('terminates workers when closed', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 2,
      createWorker: () => pushWorker(workers),
    });

    pool.close();

    expect(workers.every((worker) => worker.terminated)).toBe(true);
    await expect(pool.runBatch([task(0)])).rejects.toThrow('closed');
  });
});

class FakeSearchWorker implements SearchWorkerPoolWorker {
  onmessage: ((event: { data: SearchWorkerResponse }) => void) | null = null;
  onerror: ((event: { message?: string; error?: unknown }) => void) | null =
    null;
  messages: SearchWorkerRequest[] = [];
  terminated = false;

  postMessage(message: SearchWorkerRequest): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
    this.onmessage = null;
    this.onerror = null;
  }

  emit(response: SearchWorkerResponse): void {
    this.onmessage?.({ data: response });
  }

  fail(message: string): void {
    this.onerror?.({ message });
  }
}

function pushWorker(workers: FakeSearchWorker[]): FakeSearchWorker {
  const worker = new FakeSearchWorker();
  workers.push(worker);
  return worker;
}

function runBatchRequest(request: SearchWorkerRequest) {
  if (request.type !== 'run-batch') {
    throw new Error('Expected a run-batch request.');
  }
  return request;
}

function task(visitIndex: number): SearchWorkerTask {
  return {
    kind: 'rollout-search',
    visitIndex,
    world: createSession(`search-worker-pool-${String(visitIndex)}`, 'PlayerA'),
    rootPlayer: 'PlayerA',
    rootActionKey: `action-${String(visitIndex)}`,
    config: {
      worlds: 1,
      rollouts: 1,
      depth: 1,
      maxRootActions: 1,
      rolloutEpsilon: 0,
    },
    randomSeed: `task-${String(visitIndex)}`,
  };
}

function result(visitIndex: number): SearchWorkerResult {
  return {
    kind: 'rollout-search',
    visitIndex,
    actionKey: `action-${String(visitIndex)}`,
    score: visitIndex,
    simulatedActionSteps: 1,
    terminatedBeforeDepthLimit: false,
  };
}
