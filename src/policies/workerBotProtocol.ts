import type { GameAction, GameState, PlayerView } from '../engine/types';
import type { BotSpec } from './botSpec';
import type { SearchWorkerExecutionMode } from './searchWorkerProtocol';
import type { SearchDecisionDiagnostics } from './types';

export interface BotWorkerSelectActionRequest {
  type: 'select-action';
  requestId: number;
  spec: BotSpec;
  state: GameState;
  view: PlayerView;
  legalActions: GameAction[];
  randomSeed: string;
  /** Request full root-search diagnostics for explicit browser debugging. */
  collectDiagnostics?: boolean;
  searchExecutionMode?: SearchWorkerExecutionMode;
}

/**
 * Cooperative cancellation for work the caller is abandoning. The worker stops
 * scheduling new rollout batches but keeps its pool warm. Do not use this to
 * shorten a decision that will still be applied; bot strength comes from the
 * fixed per-profile visit budget.
 */
export interface BotWorkerCancelRequest {
  type: 'cancel';
  requestId: number;
}

/**
 * Cooperative teardown: the worker closes its nested search pool and then
 * closes itself. The owning policy falls back to `terminate()` if the worker
 * does not process this request promptly.
 */
export interface BotWorkerShutdownRequest {
  type: 'shutdown';
}

export type BotWorkerRequest =
  | BotWorkerSelectActionRequest
  | BotWorkerCancelRequest
  | BotWorkerShutdownRequest;

export interface BotWorkerSelectedActionResponse {
  type: 'selected-action';
  requestId: number;
  actionKey?: string;
  diagnostics?: SearchDecisionDiagnostics;
  searchExecutionMode?: SearchWorkerExecutionMode | 'synchronous';
}

export interface BotWorkerErrorResponse {
  type: 'error';
  requestId: number;
  message: string;
  stack?: string;
}

export type BotWorkerResponse =
  | BotWorkerSelectedActionResponse
  | BotWorkerErrorResponse;
