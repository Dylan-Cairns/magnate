import { runRolloutSearchTask } from './rolloutSearchCore';
import type { GameState } from '../engine/types';
import type {
  SearchWorkerRequest,
  SearchWorkerResponse,
  SearchWorkerInitializeRolloutSearchRequest,
  SearchWorkerRunBatchRequest,
} from './searchWorkerProtocol';

interface SearchWorkerGlobalScope {
  onmessage: ((event: MessageEvent<SearchWorkerRequest>) => void) | null;
  postMessage(message: SearchWorkerResponse): void;
  close(): void;
}

const workerScope = globalThis as unknown as SearchWorkerGlobalScope;
let rolloutSearchContextId: string | null = null;
let rolloutSearchWorldStates: readonly GameState[] = [];

workerScope.onmessage = (event) => {
  try {
    handleRequest(event.data);
  } catch (error: unknown) {
    const requestId =
      event.data.type === 'run-batch' ||
      event.data.type === 'initialize-rollout-search'
        ? event.data.requestId
        : undefined;
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
    case 'initialize-rollout-search':
      initializeRolloutSearch(request);
      return;
  }
}

function initializeRolloutSearch(
  request: SearchWorkerInitializeRolloutSearchRequest
): void {
  rolloutSearchContextId = request.context.contextId;
  rolloutSearchWorldStates = request.context.worldStates;
  workerScope.postMessage({
    type: 'initialized',
    requestId: request.requestId,
  });
}

function runBatch(request: SearchWorkerRunBatchRequest): void {
  const results = request.tasks.map((task) => {
    switch (task.kind) {
      case 'rollout-search':
        if (task.contextId !== rolloutSearchContextId) {
          throw new Error(
            `Rollout search worker missing context ${task.contextId}.`
          );
        }
        return runRolloutSearchTask(task, rolloutSearchWorldStates);
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
