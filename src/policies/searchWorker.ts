import { runRolloutSearchTask } from './rolloutSearchCore';
import type {
  SearchWorkerRequest,
  SearchWorkerResponse,
  SearchWorkerRunBatchRequest,
} from './searchWorkerProtocol';

interface SearchWorkerGlobalScope {
  onmessage: ((event: MessageEvent<SearchWorkerRequest>) => void) | null;
  postMessage(message: SearchWorkerResponse): void;
  close(): void;
}

const workerScope = globalThis as unknown as SearchWorkerGlobalScope;

workerScope.onmessage = (event) => {
  try {
    handleRequest(event.data);
  } catch (error: unknown) {
    const requestId =
      event.data.type === 'run-batch' ? event.data.requestId : undefined;
    postError(requestId, error);
  }
};

function handleRequest(request: SearchWorkerRequest): void {
  switch (request.type) {
    case 'shutdown':
      workerScope.close();
      return;
    case 'run-batch':
      runBatch(request);
      return;
  }
}

function runBatch(request: SearchWorkerRunBatchRequest): void {
  const results = request.tasks.map((task) => {
    switch (task.kind) {
      case 'rollout-search':
        return runRolloutSearchTask(task);
    }
  });
  workerScope.postMessage({
    type: 'batch-result',
    requestId: request.requestId,
    results,
  });
}

function postError(requestId: number | undefined, error: unknown): void {
  const normalized = error instanceof Error ? error : new Error(String(error));
  workerScope.postMessage({
    type: 'error',
    ...(requestId !== undefined ? { requestId } : {}),
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
