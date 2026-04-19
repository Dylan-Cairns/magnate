import type { RolloutSearchWorkerContext } from './rolloutSearchCore';
import type {
  SearchWorkerRequest,
  SearchWorkerResponse,
  SearchWorkerResult,
  SearchWorkerTask,
} from './searchWorkerProtocol';

export interface SearchWorkerPoolWorker {
  onmessage: ((event: { data: SearchWorkerResponse }) => void) | null;
  onerror: ((event: { message?: string; error?: unknown }) => void) | null;
  postMessage(message: SearchWorkerRequest): void;
  terminate(): void;
}

export interface SearchWorkerPool {
  runBatch(
    tasks: readonly SearchWorkerTask[],
    context: RolloutSearchWorkerContext
  ): Promise<readonly SearchWorkerResult[]>;
  close(): void;
}

export interface SearchWorkerPoolOptions {
  workerCount: number;
  createWorker?: () => SearchWorkerPoolWorker;
}

interface PoolWorker {
  id: number;
  worker: SearchWorkerPoolWorker;
}

interface PendingChunk {
  kind: 'batch';
  resolve(results: readonly SearchWorkerResult[]): void;
  reject(error: Error): void;
}

interface PendingInitialize {
  kind: 'initialize';
  resolve(): void;
  reject(error: Error): void;
}

type PendingRequest = PendingChunk | PendingInitialize;

export function createSearchWorkerPool({
  workerCount,
  createWorker = createDefaultSearchWorker,
}: SearchWorkerPoolOptions): SearchWorkerPool {
  if (!Number.isInteger(workerCount) || workerCount <= 0) {
    throw new Error('Search worker pool workerCount must be positive.');
  }

  let closed = false;
  let activeRequest = false;
  let nextRequestId = 1;
  let initializedRolloutSearchContextId: string | null = null;
  let initializedRolloutSearchWorldStates: RolloutSearchWorkerContext['worldStates'] | null =
    null;
  const pendingByRequestId = new Map<number, PendingRequest>();
  const workers: PoolWorker[] = Array.from(
    { length: workerCount },
    (_, index) => createPoolWorker(index + 1, createWorker)
  );

  function createPoolWorker(
    id: number,
    workerFactory: () => SearchWorkerPoolWorker
  ): PoolWorker {
    const worker = workerFactory();
    const poolWorker: PoolWorker = { id, worker };
    worker.onmessage = (event) => {
      handleMessage(poolWorker, event.data);
    };
    worker.onerror = (event) => {
      failAll(
        event.error instanceof Error
          ? event.error
          : new Error(event.message ?? `Search worker ${String(id)} failed.`)
      );
    };
    return poolWorker;
  }

  async function runBatch(
    tasks: readonly SearchWorkerTask[],
    context: RolloutSearchWorkerContext
  ): Promise<readonly SearchWorkerResult[]> {
    if (closed) {
      throw new Error('Search worker pool is closed.');
    }
    if (activeRequest) {
      throw new Error(
        'Search worker pool does not support concurrent batches.'
      );
    }
    if (tasks.length === 0) {
      return [];
    }

    activeRequest = true;
    try {
      await ensureRolloutSearchContext(context);
      const chunks = chunkTasks(tasks, workers.length);
      const resultsByChunk = await Promise.all(
        chunks.map((chunk, index) => runChunk(workers[index], chunk))
      );
      return resultsByChunk.flat();
    } finally {
      activeRequest = false;
    }
  }

  async function ensureRolloutSearchContext(
    context: RolloutSearchWorkerContext
  ): Promise<void> {
    if (
      initializedRolloutSearchContextId === context.contextId &&
      initializedRolloutSearchWorldStates === context.worldStates
    ) {
      return;
    }
    await Promise.all(
      workers.map((poolWorker) => initializeWorker(poolWorker, context))
    );
    initializedRolloutSearchContextId = context.contextId;
    initializedRolloutSearchWorldStates = context.worldStates;
  }

  function initializeWorker(
    poolWorker: PoolWorker,
    context: RolloutSearchWorkerContext
  ): Promise<void> {
    const requestId = nextRequestId;
    nextRequestId += 1;

    return new Promise((resolve, reject) => {
      pendingByRequestId.set(requestId, { kind: 'initialize', resolve, reject });
      try {
        poolWorker.worker.postMessage({
          type: 'initialize-rollout-search',
          requestId,
          context,
        });
      } catch (error) {
        pendingByRequestId.delete(requestId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  function runChunk(
    poolWorker: PoolWorker,
    tasks: readonly SearchWorkerTask[]
  ): Promise<readonly SearchWorkerResult[]> {
    const requestId = nextRequestId;
    nextRequestId += 1;

    return new Promise((resolve, reject) => {
      pendingByRequestId.set(requestId, { kind: 'batch', resolve, reject });
      try {
        poolWorker.worker.postMessage({
          type: 'run-batch',
          requestId,
          tasks: [...tasks],
        });
      } catch (error) {
        pendingByRequestId.delete(requestId);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  function handleMessage(
    poolWorker: PoolWorker,
    response: SearchWorkerResponse
  ): void {
    if (response.type === 'error') {
      const label =
        response.requestId === undefined
          ? ''
          : ` request ${String(response.requestId)}`;
      failAll(
        new Error(
          `Search worker ${String(poolWorker.id)} failed${label}: ${response.message}`
        )
      );
      return;
    }

    const pending = pendingByRequestId.get(response.requestId);
    if (!pending) {
      return;
    }
    pendingByRequestId.delete(response.requestId);
    if (response.type === 'initialized') {
      if (pending.kind !== 'initialize') {
        failAll(
          new Error(
            `Search worker ${String(poolWorker.id)} returned initialization for a batch request.`
          )
        );
        return;
      }
      pending.resolve();
      return;
    }
    if (pending.kind !== 'batch') {
      failAll(
        new Error(
          `Search worker ${String(poolWorker.id)} returned batch results for an initialization request.`
        )
      );
      return;
    }
    pending.resolve(response.results);
  }

  function failAll(error: Error): void {
    for (const pending of pendingByRequestId.values()) {
      pending.reject(error);
    }
    pendingByRequestId.clear();
    close();
  }

  function close(): void {
    if (closed) {
      return;
    }
    closed = true;
    for (const pending of pendingByRequestId.values()) {
      pending.reject(new Error('Search worker pool closed.'));
    }
    pendingByRequestId.clear();
    for (const poolWorker of workers) {
      poolWorker.worker.onmessage = null;
      poolWorker.worker.onerror = null;
      poolWorker.worker.terminate();
    }
  }

  return {
    runBatch,
    close,
  };
}

function chunkTasks<T>(items: readonly T[], maxChunks: number): T[][] {
  const chunkCount = Math.max(1, Math.min(maxChunks, items.length));
  const chunks: T[][] = Array.from({ length: chunkCount }, () => []);
  for (let index = 0; index < items.length; index += 1) {
    chunks[index % chunkCount].push(items[index]);
  }
  return chunks;
}

function createDefaultSearchWorker(): SearchWorkerPoolWorker {
  return new Worker(new URL('./searchWorker.ts', import.meta.url), {
    type: 'module',
  }) as unknown as SearchWorkerPoolWorker;
}
