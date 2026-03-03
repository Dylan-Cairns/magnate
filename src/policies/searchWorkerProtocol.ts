import type {
  RolloutSearchVisitResult,
  RolloutSearchWorkerTask,
} from './rolloutSearchCore';

export type SearchWorkerTask = RolloutSearchWorkerTask;
export type SearchWorkerResult = RolloutSearchVisitResult;

export interface SearchWorkerRunBatchRequest {
  type: 'run-batch';
  requestId: number;
  tasks: SearchWorkerTask[];
}

export interface SearchWorkerShutdownRequest {
  type: 'shutdown';
}

export type SearchWorkerRequest =
  | SearchWorkerRunBatchRequest
  | SearchWorkerShutdownRequest;

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
  | SearchWorkerBatchResultResponse
  | SearchWorkerErrorResponse;
