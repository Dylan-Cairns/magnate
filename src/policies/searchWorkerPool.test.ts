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

const TEST_CONTEXT = {
  contextId: 'search-worker-pool-test-context',
  worldStates: [createSession('search-worker-pool-context', 'PlayerA')],
};

describe('search worker pool', () => {
  it('distributes a batch across workers and resolves returned results', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 2,
      createWorker: () => pushWorker(workers),
    });

    const run = pool.runBatch([task(0), task(1), task(2)], TEST_CONTEXT);

    expect(workers).toHaveLength(2);
    expect(initRequest(workers[0].messages[0]).context.contextId).toBe(
      TEST_CONTEXT.contextId
    );
    expect(initRequest(workers[1].messages[0]).context.contextId).toBe(
      TEST_CONTEXT.contextId
    );
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[0]).requestId,
    });
    workers[1].emit({
      type: 'initialized',
      requestId: initRequest(workers[1].messages[0]).requestId,
    });
    await flushAsyncWork();

    expect(
      runBatchRequest(workers[0].messages[1]).tasks.map(
        (item) => item.visitIndex
      )
    ).toEqual([0, 2]);
    expect(
      runBatchRequest(workers[1].messages[1]).tasks.map(
        (item) => item.visitIndex
      )
    ).toEqual([1]);

    workers[1].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[1].messages[1]).requestId,
      results: [result(1)],
    });
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[1]).requestId,
      results: [result(0), result(2)],
    });

    await expect(run).resolves.toEqual([result(0), result(2), result(1)]);
  });

  it('reuses initialized rollout search worlds for the same context object', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 1,
      createWorker: () => pushWorker(workers),
    });

    const first = pool.runBatch([task(0)], TEST_CONTEXT);
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[0]).requestId,
    });
    await flushAsyncWork();
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[1]).requestId,
      results: [result(0)],
    });
    await expect(first).resolves.toEqual([result(0)]);

    const second = pool.runBatch([task(1)], TEST_CONTEXT);
    await flushAsyncWork();

    expect(runBatchRequest(workers[0].messages[2]).tasks).toHaveLength(1);
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[2]).requestId,
      results: [result(1)],
    });
    await expect(second).resolves.toEqual([result(1)]);
  });

  it('forwards an evaluation execution mode to every worker chunk', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 2,
      executionMode: 'resumable-paired-td',
      createWorker: () => pushWorker(workers),
    });

    const run = pool.runBatch([task(0), task(1)], TEST_CONTEXT);
    for (const worker of workers) {
      worker.emit({
        type: 'initialized',
        requestId: initRequest(worker.messages[0]).requestId,
      });
    }
    await flushAsyncWork();

    for (const worker of workers) {
      expect(runBatchRequest(worker.messages[1]).executionMode).toBe(
        'resumable-paired-td'
      );
    }
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[1]).requestId,
      results: [result(0)],
    });
    workers[1].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[1].messages[1]).requestId,
      results: [result(1)],
    });
    await expect(run).resolves.toEqual([result(0), result(1)]);
  });

  it('reinitializes rollout search worlds when the same context id carries new worlds', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 1,
      createWorker: () => pushWorker(workers),
    });
    const nextContext = {
      contextId: TEST_CONTEXT.contextId,
      worldStates: [
        createSession('search-worker-pool-next-context', 'PlayerA'),
      ],
    };

    const first = pool.runBatch([task(0)], TEST_CONTEXT);
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[0]).requestId,
    });
    await flushAsyncWork();
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[1]).requestId,
      results: [result(0)],
    });
    await expect(first).resolves.toEqual([result(0)]);

    const second = pool.runBatch([task(1)], nextContext);

    expect(initRequest(workers[0].messages[2]).context).toBe(nextContext);
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[2]).requestId,
    });
    await flushAsyncWork();
    expect(runBatchRequest(workers[0].messages[3]).tasks).toHaveLength(1);
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[3]).requestId,
      results: [result(1)],
    });
    await expect(second).resolves.toEqual([result(1)]);
  });

  it('reinitializes rollout search context when guidance changes', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 1,
      createWorker: () => pushWorker(workers),
    });
    const firstContext = {
      ...TEST_CONTEXT,
      guidance: {
        kind: 'td-root' as const,
        modelIndexPath: 'model-packs/first-index.json',
      },
    };
    const secondContext = {
      ...TEST_CONTEXT,
      guidance: {
        kind: 'td-root' as const,
        modelIndexPath: 'model-packs/second-index.json',
      },
    };

    const first = pool.runBatch([task(0)], firstContext);
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[0]).requestId,
    });
    await flushAsyncWork();
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[1]).requestId,
      results: [result(0)],
    });
    await expect(first).resolves.toEqual([result(0)]);

    const second = pool.runBatch([task(1)], secondContext);

    expect(initRequest(workers[0].messages[2]).context).toBe(secondContext);
    workers[0].emit({
      type: 'initialized',
      requestId: initRequest(workers[0].messages[2]).requestId,
    });
    await flushAsyncWork();
    workers[0].emit({
      type: 'batch-result',
      requestId: runBatchRequest(workers[0].messages[3]).requestId,
      results: [result(1)],
    });
    await expect(second).resolves.toEqual([result(1)]);
  });

  it('rejects active work and closes workers when a worker fails', async () => {
    const workers: FakeSearchWorker[] = [];
    const pool = createSearchWorkerPool({
      workerCount: 1,
      createWorker: () => pushWorker(workers),
    });

    const run = pool.runBatch([task(0)], TEST_CONTEXT);
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
    await expect(pool.runBatch([task(0)], TEST_CONTEXT)).rejects.toThrow(
      'closed'
    );
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

function initRequest(request: SearchWorkerRequest) {
  if (request.type !== 'initialize-rollout-search') {
    throw new Error('Expected an initialize-rollout-search request.');
  }
  return request;
}

function task(visitIndex: number): SearchWorkerTask {
  return {
    kind: 'rollout-search',
    contextId: TEST_CONTEXT.contextId,
    visitIndex,
    actionVisitIndex: visitIndex,
    scenarioIndex: visitIndex,
    worldIndex: 0,
    engineSeed: `engine-${String(visitIndex)}`,
    rootPlayer: 'PlayerA',
    rootAction: { type: 'end-turn' },
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

function flushAsyncWork(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
