import type { ActionId, GameAction, GameState, PlayerId, PlayerView } from '../engine/types';
import type { KeyedAction } from '../engine/actionSurface';

export const BRIDGE_CONTRACT_NAME = 'magnate_bridge' as const;
export const BRIDGE_CONTRACT_VERSION = 'v1' as const;

export const BRIDGE_COMMANDS = [
  'metadata',
  'reset',
  'step',
  'legalActions',
  'observation',
  'serialize',
] as const;

export type BridgeCommand = (typeof BRIDGE_COMMANDS)[number];

export const BRIDGE_ERROR_CODES = [
  'INVALID_COMMAND',
  'INVALID_PAYLOAD',
  'ILLEGAL_ACTION',
  'STATE_DESERIALIZATION_FAILED',
  'INTERNAL_ENGINE_ERROR',
] as const;

export type BridgeErrorCode = (typeof BRIDGE_ERROR_CODES)[number];

export interface BridgeRequestEnvelope {
  requestId: string;
  command: string;
  payload?: unknown;
}

export interface BridgeSuccessEnvelope<TResult = unknown> {
  requestId: string;
  ok: true;
  result: TResult;
}

export interface BridgeFailureEnvelope {
  requestId: string;
  ok: false;
  error: {
    code: BridgeErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

export type BridgeResponseEnvelope<TResult = unknown> =
  | BridgeSuccessEnvelope<TResult>
  | BridgeFailureEnvelope;

export interface BridgeMetadataResult {
  contractName: typeof BRIDGE_CONTRACT_NAME;
  contractVersion: typeof BRIDGE_CONTRACT_VERSION;
  schemaVersion: number;
  commands: readonly BridgeCommand[];
  actionIds: readonly ActionId[];
  actionSurface: {
    stableKey: 'actionKey';
    canonicalOrder: 'ascending_lexicographic_action_key';
  };
  observationSpec: {
    name: 'player_view_v1';
    defaultViewer: 'active-player';
    optionalMask: 'legal action keys';
  };
  modelIO: {
    inputs: {
      observation: 'observation';
      actionMask: 'action_mask';
    };
    outputs: {
      maskedLogits: 'masked_logits';
      value: 'value';
    };
  };
}

export interface BridgeStateResult {
  state: GameState;
  view: PlayerView;
  terminal: boolean;
}

export interface BridgeResetPayload {
  seed?: string;
  firstPlayer?: PlayerId;
  serializedState?: unknown;
  skipAdvanceToDecision?: boolean;
}

export interface BridgeLegalActionsResult {
  actions: readonly KeyedAction[];
  activePlayerId: PlayerId;
  phase: GameState['phase'];
}

export interface BridgeObservationPayload {
  viewerId?: PlayerId;
  includeLegalActionMask?: boolean;
}

export interface BridgeObservationResult {
  view: PlayerView;
  legalActionMask?: readonly string[];
}

export interface BridgeStepPayload {
  action?: GameAction;
  actionKey?: string;
}

