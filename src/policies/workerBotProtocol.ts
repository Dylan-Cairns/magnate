import type { GameAction, GameState, PlayerView } from '../engine/types';
import type { BotSpec } from './botSpec';
import type { SearchDecisionDiagnostics } from './types';

export interface BotWorkerSelectActionRequest {
  type: 'select-action';
  requestId: number;
  spec: BotSpec;
  state: GameState;
  view: PlayerView;
  legalActions: GameAction[];
  randomSeed: string;
}

export interface BotWorkerCancelRequest {
  type: 'cancel';
  requestId: number;
}

export type BotWorkerRequest =
  | BotWorkerSelectActionRequest
  | BotWorkerCancelRequest;

export interface BotWorkerSelectedActionResponse {
  type: 'selected-action';
  requestId: number;
  actionKey?: string;
  diagnostics?: SearchDecisionDiagnostics;
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
