import {
  SearchWorkerMessageType,
  SearchWorkerExecutionMode,
} from './workerValues';
import {
  runRolloutSearchTask,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchWorkerGuidance,
} from './rolloutSearchCore';
import type { GameState } from '../engine/types';
import { preloadTdRootBrowserModel } from './modelRuntimeCache';
import { createTdRootSearchRolloutGuidance } from './tdRootSearchPolicy';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import {
  assertPairedTdRolloutAvailable,
  runRolloutSearchTaskBatchResumable,
} from './rolloutSearchPairedTd';
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
let rolloutSearchModel: LoadedTdGuidanceModel | undefined;
let rolloutSearchPairTdActions = false;

workerScope.onmessage = (event) => {
  void handleRequest(event.data).catch((error: unknown) => {
    const requestId =
      event.data.type === SearchWorkerMessageType.RunBatch ||
      event.data.type === SearchWorkerMessageType.InitializeRolloutSearch
        ? event.data.requestId
        : undefined;
    postError(requestId, error);
  });
};

async function handleRequest(request: SearchWorkerRequest): Promise<void> {
  switch (request.type) {
    case SearchWorkerMessageType.Shutdown:
      workerScope.close();
      return;
    case SearchWorkerMessageType.RunBatch:
      runBatch(request);
      return;
    case SearchWorkerMessageType.InitializeRolloutSearch:
      await initializeRolloutSearch(request);
      return;
  }
}

async function initializeRolloutSearch(
  request: SearchWorkerInitializeRolloutSearchRequest
): Promise<void> {
  rolloutSearchContextId = request.context.contextId;
  rolloutSearchWorldStates = request.context.worldStates;
  const runtime = await createRuntimeGuidance(request.context.guidance);
  rolloutSearchGuidance = runtime.guidance;
  rolloutSearchModel = runtime.model;
  rolloutSearchPairTdActions = runtime.pairTdActions;
  workerScope.postMessage({
    type: SearchWorkerMessageType.Initialized,
    requestId: request.requestId,
  });
}

function runBatch(request: SearchWorkerRunBatchRequest): void {
  for (const task of request.tasks) {
    if (task.contextId !== rolloutSearchContextId) {
      throw new Error(
        `Rollout search worker missing context ${task.contextId}.`
      );
    }
  }
  if (request.executionMode === SearchWorkerExecutionMode.ResumablePairedTd) {
    assertPairedTdRolloutAvailable(
      rolloutSearchModel,
      rolloutSearchPairTdActions
    );
  }
  const results =
    request.executionMode === SearchWorkerExecutionMode.ResumablePairedTd ||
    request.executionMode === SearchWorkerExecutionMode.ResumableScalar
      ? runRolloutSearchTaskBatchResumable(
          request.tasks,
          rolloutSearchWorldStates,
          rolloutSearchGuidance,
          rolloutSearchModel,
          request.executionMode ===
            SearchWorkerExecutionMode.ResumablePairedTd &&
            rolloutSearchPairTdActions
        ).results
      : request.tasks.map((task) =>
          runRolloutSearchTask(
            task,
            rolloutSearchWorldStates,
            undefined,
            rolloutSearchGuidance
          )
        );
  workerScope.postMessage({
    type: SearchWorkerMessageType.BatchResult,
    requestId: request.requestId,
    results: [...results],
  });
}

async function createRuntimeGuidance(
  guidance: RolloutSearchWorkerGuidance | undefined
): Promise<{
  guidance: RolloutSearchRuntimeGuidance | undefined;
  model: LoadedTdGuidanceModel | undefined;
  pairTdActions: boolean;
}> {
  if (!guidance) {
    return {
      guidance: undefined,
      model: undefined,
      pairTdActions: false,
    };
  }
  const model = await preloadTdRootBrowserModel(guidance.modelIndexPath);
  return {
    guidance: createTdRootSearchRolloutGuidance({ model }),
    model,
    pairTdActions: true,
  };
}

function postError(requestId: number | undefined, error: unknown): void {
  const normalized = error instanceof Error ? error : new Error(String(error));
  workerScope.postMessage({
    type: SearchWorkerMessageType.Error,
    ...(requestId !== undefined ? { requestId } : {}),
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
