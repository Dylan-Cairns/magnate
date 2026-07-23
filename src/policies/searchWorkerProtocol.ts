import type {
  RolloutSearchWorkerContext,
  RolloutSearchVisitResult,
  RolloutSearchWorkerTask,
} from './rolloutSearchCore';

export type SearchWorkerTask = RolloutSearchWorkerTask;
export type SearchWorkerResult = RolloutSearchVisitResult;
export type SearchWorkerExecutionMode =
  | 'legacy'
  | 'resumable-scalar'
  | 'resumable-paired-td';

export interface SearchWorkerRunBatchRequest {
  type: 'run-batch';
  requestId: number;
  tasks: SearchWorkerTask[];
  executionMode?: SearchWorkerExecutionMode;
}

export interface SearchWorkerInitializeRolloutSearchRequest {
  type: 'initialize-rollout-search';
  requestId: number;
  context: RolloutSearchWorkerContext;
}

export interface SearchWorkerShutdownRequest {
  type: 'shutdown';
}

export type SearchWorkerRequest =
  | SearchWorkerRunBatchRequest
  | SearchWorkerInitializeRolloutSearchRequest
  | SearchWorkerShutdownRequest;

export interface SearchWorkerInitializedResponse {
  type: 'initialized';
  requestId: number;
}

export interface SearchWorkerBatchResultResponse {
  type: 'batch-result';
  requestId: number;
  results: SearchWorkerResult[];
}

export interface SearchWorkerErrorResponse {
  type: 'error';
  requestId?: number;
  message: string;
  stack?: string;
}

export type SearchWorkerResponse =
  | SearchWorkerInitializedResponse
  | SearchWorkerBatchResultResponse
  | SearchWorkerErrorResponse;
