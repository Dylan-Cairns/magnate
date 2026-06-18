import {
  runRolloutSearchTask,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchWorkerGuidance,
} from './rolloutSearchCore';
import type { GameState } from '../engine/types';
import { preloadTdRootBrowserModel } from './modelRuntimeCache';
import { createTdRootSearchRolloutGuidance } from './tdRootSearchPolicy';
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
let rolloutSearchGuidance: RolloutSearchRuntimeGuidance | undefined;

workerScope.onmessage = (event) => {
  void handleRequest(event.data).catch((error: unknown) => {
    const requestId =
      event.data.type === 'run-batch' ||
      event.data.type === 'initialize-rollout-search'
        ? event.data.requestId
        : undefined;
    postError(requestId, error);
  });
};

async function handleRequest(request: SearchWorkerRequest): Promise<void> {
  switch (request.type) {
    case 'shutdown':
      workerScope.close();
      return;
    case 'run-batch':
      runBatch(request);
      return;
    case 'initialize-rollout-search':
      await initializeRolloutSearch(request);
      return;
  }
}

async function initializeRolloutSearch(
  request: SearchWorkerInitializeRolloutSearchRequest
): Promise<void> {
  rolloutSearchContextId = request.context.contextId;
  rolloutSearchWorldStates = request.context.worldStates;
  rolloutSearchGuidance = await createRuntimeGuidance(request.context.guidance);
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
        return runRolloutSearchTask(
          task,
          rolloutSearchWorldStates,
          undefined,
          rolloutSearchGuidance
        );
    }
  });
  workerScope.postMessage({
    type: 'batch-result',
    requestId: request.requestId,
    results,
  });
}

async function createRuntimeGuidance(
  guidance: RolloutSearchWorkerGuidance | undefined
): Promise<RolloutSearchRuntimeGuidance | undefined> {
  if (!guidance) {
    return undefined;
  }
  switch (guidance.kind) {
    case 'td-root': {
      const model = await preloadTdRootBrowserModel(guidance.modelIndexPath);
      return createTdRootSearchRolloutGuidance({ model });
    }
  }
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
