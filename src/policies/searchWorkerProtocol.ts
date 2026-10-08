import {
  SearchWorkerExecutionMode,
  SearchWorkerMessageType,
} from './workerValues';
import type {
  RolloutSearchWorkerContext,
  RolloutSearchVisitResult,
  RolloutSearchWorkerTask,
} from './rolloutSearchCore';

export type SearchWorkerTask = RolloutSearchWorkerTask;
export type SearchWorkerResult = RolloutSearchVisitResult;

export interface SearchWorkerRunBatchRequest {
  type: typeof SearchWorkerMessageType.RunBatch;
  requestId: number;
  tasks: SearchWorkerTask[];
  executionMode?: SearchWorkerExecutionMode;
}

export interface SearchWorkerInitializeRolloutSearchRequest {
  type: typeof SearchWorkerMessageType.InitializeRolloutSearch;
  requestId: number;
  context: RolloutSearchWorkerContext;
}

export interface SearchWorkerShutdownRequest {
  type: typeof SearchWorkerMessageType.Shutdown;
}

export type SearchWorkerRequest =
  | SearchWorkerRunBatchRequest
  | SearchWorkerInitializeRolloutSearchRequest
  | SearchWorkerShutdownRequest;

export interface SearchWorkerInitializedResponse {
  type: typeof SearchWorkerMessageType.Initialized;
  requestId: number;
}

export interface SearchWorkerBatchResultResponse {
  type: typeof SearchWorkerMessageType.BatchResult;
  requestId: number;
  results: SearchWorkerResult[];
}

export interface SearchWorkerErrorResponse {
  type: typeof SearchWorkerMessageType.Error;
  requestId?: number;
  message: string;
  stack?: string;
}

export type SearchWorkerResponse =
  | SearchWorkerInitializedResponse
  | SearchWorkerBatchResultResponse
  | SearchWorkerErrorResponse;

export { SearchWorkerExecutionMode } from './workerValues';
