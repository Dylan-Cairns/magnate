export const BridgeCommand = {
  Metadata: 'metadata',
  Reset: 'reset',
  Step: 'step',
  LegalActions: 'legalActions',
  Observation: 'observation',
  Serialize: 'serialize',
} as const;
export type BridgeCommand = (typeof BridgeCommand)[keyof typeof BridgeCommand];

export const BridgeErrorCode = {
  InvalidCommand: 'INVALID_COMMAND',
  InvalidPayload: 'INVALID_PAYLOAD',
  IllegalAction: 'ILLEGAL_ACTION',
  StateDeserializationFailed: 'STATE_DESERIALIZATION_FAILED',
  InternalEngineError: 'INTERNAL_ENGINE_ERROR',
} as const;
export type BridgeErrorCode =
  (typeof BridgeErrorCode)[keyof typeof BridgeErrorCode];

export const BRIDGE_COMMANDS = [
  BridgeCommand.Metadata,
  BridgeCommand.Reset,
  BridgeCommand.Step,
  BridgeCommand.LegalActions,
  BridgeCommand.Observation,
  BridgeCommand.Serialize,
] as const;

export const BRIDGE_ERROR_CODES = [
  BridgeErrorCode.InvalidCommand,
  BridgeErrorCode.InvalidPayload,
  BridgeErrorCode.IllegalAction,
  BridgeErrorCode.StateDeserializationFailed,
  BridgeErrorCode.InternalEngineError,
] as const;
