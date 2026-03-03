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
    tasks: readonly SearchWorkerTask[]
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
  resolve(results: readonly SearchWorkerResult[]): void;
  reject(error: Error): void;
}

export function createSearchWorkerPool({
  workerCount,
  createWorker = createDefaultSearchWorker,
}: SearchWorkerPoolOptions): SearchWorkerPool {
  if (!Number.isInteger(workerCount) || workerCount <= 0) {
    throw new Error('Search worker pool workerCount must be positive.');
  }

  let closed = false;
  let activeBatch = false;
  let nextRequestId = 1;
  const pendingByRequestId = new Map<number, PendingChunk>();
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
    tasks: readonly SearchWorkerTask[]
  ): Promise<readonly SearchWorkerResult[]> {
    if (closed) {
      throw new Error('Search worker pool is closed.');
    }
    if (activeBatch) {
      throw new Error(
        'Search worker pool does not support concurrent batches.'
      );
    }
    if (tasks.length === 0) {
      return [];
    }

    activeBatch = true;
    try {
      const chunks = chunkTasks(tasks, workers.length);
      const resultsByChunk = await Promise.all(
        chunks.map((chunk, index) => runChunk(workers[index], chunk))
      );
      return resultsByChunk.flat();
    } finally {
      activeBatch = false;
    }
  }

  function runChunk(
    poolWorker: PoolWorker,
    tasks: readonly SearchWorkerTask[]
  ): Promise<readonly SearchWorkerResult[]> {
    const requestId = nextRequestId;
    nextRequestId += 1;

    return new Promise((resolve, reject) => {
      pendingByRequestId.set(requestId, { resolve, reject });
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
