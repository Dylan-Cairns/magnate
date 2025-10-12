import { ACTION_IDS, actionStableKey, legalActionsCanonical } from '../engine/actionSurface';
import { createSession } from '../engine/session';
import { applyAction } from '../engine/reducer';
import { isTerminal } from '../engine/scoring';
import { advanceToDecision } from '../engine/turnFlow';
import type { GameAction, GameState, PlayerId } from '../engine/types';
import { toActivePlayerView, toPlayerView } from '../engine/view';
import type {
  BridgeCommand,
  BridgeErrorCode,
  BridgeFailureEnvelope,
  BridgeLegalActionsResult,
  BridgeMetadataResult,
  BridgeObservationPayload,
  BridgeObservationResult,
  BridgeResetPayload,
  BridgeResponseEnvelope,
  BridgeStateResult,
  BridgeStepPayload,
  BridgeSuccessEnvelope,
} from './protocol';
import {
  BRIDGE_COMMANDS,
  BRIDGE_CONTRACT_NAME,
  BRIDGE_CONTRACT_VERSION,
} from './protocol';

const DEFAULT_RESET_SEED = 'bridge-default-seed';
const DEFAULT_FIRST_PLAYER: PlayerId = 'PlayerA';
const SUPPORTED_SCHEMA_VERSION = 1;

class RuntimeBridgeError extends Error {
  code: BridgeErrorCode;
  details?: Record<string, unknown>;

  constructor(
    code: BridgeErrorCode,
    message: string,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.code = code;
    this.details = details;
  }
}

export class MagnateBridgeRuntime {
  private state: GameState;

  constructor() {
    this.state = createSession(DEFAULT_RESET_SEED, DEFAULT_FIRST_PLAYER);
  }

  handleRequest(raw: unknown): BridgeResponseEnvelope {
    const requestId = extractRequestId(raw) ?? 'unknown';

    try {
      const request = parseEnvelope(raw);
      const result = this.execute(request.command, request.payload);
      return success(request.requestId, result);
    } catch (error) {
      return failure(requestId, toBridgeError(error));
    }
  }

  private execute(command: BridgeCommand, payload: unknown): unknown {
    switch (command) {
      case 'metadata':
        return this.metadata();
      case 'reset':
        return this.reset(payload);
      case 'legalActions':
        return this.legalActions(payload);
      case 'observation':
        return this.observation(payload);
      case 'step':
        return this.step(payload);
      case 'serialize':
        return this.serialize(payload);
    }
  }

  private metadata(): BridgeMetadataResult {
    return {
      contractName: BRIDGE_CONTRACT_NAME,
      contractVersion: BRIDGE_CONTRACT_VERSION,
      schemaVersion: SUPPORTED_SCHEMA_VERSION,
      commands: BRIDGE_COMMANDS,
      actionIds: ACTION_IDS,
      actionSurface: {
        stableKey: 'actionKey',
        canonicalOrder: 'ascending_lexicographic_action_key',
      },
      observationSpec: {
        name: 'player_view_v1',
        defaultViewer: 'active-player',
        optionalMask: 'legal action keys',
      },
      modelIO: {
        inputs: {
          observation: 'observation',
          actionMask: 'action_mask',
        },
        outputs: {
          maskedLogits: 'masked_logits',
          value: 'value',
        },
      },
    };
  }

  private reset(payload: unknown): BridgeStateResult {
    const parsed = parseResetPayload(payload);
    if (parsed.serializedState !== undefined) {
      const deserialized = parseSerializedState(parsed.serializedState);
      this.state = parsed.skipAdvanceToDecision ? deserialized : advanceToDecision(deserialized);
      return this.stateResult();
    }

    const seed = parsed.seed ?? DEFAULT_RESET_SEED;
    const firstPlayer = parsed.firstPlayer ?? DEFAULT_FIRST_PLAYER;
    this.state = createSession(seed, firstPlayer);
    return this.stateResult();
  }

  private legalActions(payload: unknown): BridgeLegalActionsResult {
    if (payload !== undefined && !isObject(payload)) {
      throw new RuntimeBridgeError(
        'INVALID_PAYLOAD',
        'legalActions payload must be an object when provided.'
      );
    }

    return {
      actions: cloneForWire(legalActionsCanonical(this.state)),
      activePlayerId: this.activePlayerId(),
      phase: this.state.phase,
    };
  }

  private observation(payload: unknown): BridgeObservationResult {
    const parsed = parseObservationPayload(payload);
    const viewerId = parsed.viewerId ?? this.activePlayerId();
    const view = toPlayerView(this.state, viewerId);

    if (!parsed.includeLegalActionMask) {
      return { view: cloneForWire(view) };
    }

    const legalActionMask =
      viewerId === this.activePlayerId()
        ? legalActionsCanonical(this.state).map((entry) => entry.actionKey)
        : [];

    return {
      view: cloneForWire(view),
      legalActionMask,
    };
  }

  private step(payload: unknown): BridgeStateResult {
    const parsed = parseStepPayload(payload);
    const action = this.resolveStepAction(parsed);

    try {
      this.state = advanceToDecision(applyAction(this.state, action));
    } catch (error) {
      if (error instanceof Error && error.message.includes('Illegal action')) {
        throw new RuntimeBridgeError('ILLEGAL_ACTION', error.message);
      }
      throw error;
    }

    return this.stateResult();
  }

  private serialize(payload: unknown): { state: GameState } {
    if (payload !== undefined && !isObject(payload)) {
      throw new RuntimeBridgeError(
        'INVALID_PAYLOAD',
        'serialize payload must be an object when provided.'
      );
    }

    return {
      state: cloneForWire(this.state),
    };
  }

  private resolveStepAction(payload: BridgeStepPayload): GameAction {
    if (payload.actionKey !== undefined) {
      const key = payload.actionKey;
      const match = legalActionsCanonical(this.state).find(
        (candidate) => candidate.actionKey === key
      );
      if (!match) {
        throw new RuntimeBridgeError('ILLEGAL_ACTION', `Unknown legal action key: ${key}`);
      }

      if (payload.action) {
        const payloadKey = actionStableKey(payload.action);
        if (payloadKey !== key) {
          throw new RuntimeBridgeError(
            'INVALID_PAYLOAD',
            'step payload action and actionKey must refer to the same action.',
            { actionKey: key, payloadActionKey: payloadKey }
          );
        }
      }

      return match.action;
    }

    if (payload.action) {
      return payload.action;
    }

    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'step payload requires either action or actionKey.'
    );
  }

  private stateResult(): BridgeStateResult {
    return {
      state: cloneForWire(this.state),
      view: cloneForWire(toActivePlayerView(this.state)),
      terminal: isTerminal(this.state),
    };
  }

  private activePlayerId(): PlayerId {
    const activePlayer = this.state.players[this.state.activePlayerIndex];
    if (!activePlayer) {
      throw new RuntimeBridgeError(
        'INTERNAL_ENGINE_ERROR',
        `Active player index is out of bounds: ${this.state.activePlayerIndex}.`
      );
    }
    return activePlayer.id;
  }
}

function parseEnvelope(raw: unknown): {
  requestId: string;
  command: BridgeCommand;
  payload: unknown;
} {
  if (!isObject(raw)) {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'Request must be a JSON object.');
  }

  const requestId = raw.requestId;
  if (typeof requestId !== 'string' || requestId.trim() === '') {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'Request field "requestId" must be a non-empty string.'
    );
  }

  const commandValue = raw.command;
  if (typeof commandValue !== 'string') {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'Request field "command" must be a string.'
    );
  }

  const command = parseCommand(commandValue);
  return {
    requestId,
    command,
    payload: raw.payload,
  };
}

function parseCommand(command: string): BridgeCommand {
  if (BRIDGE_COMMANDS.includes(command as BridgeCommand)) {
    return command as BridgeCommand;
  }

  throw new RuntimeBridgeError('INVALID_COMMAND', `Unsupported command: ${command}`);
}

function parseResetPayload(payload: unknown): BridgeResetPayload {
  if (payload === undefined) {
    return {};
  }

  if (!isObject(payload)) {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'reset payload must be an object.');
  }

  const seed = payload.seed;
  if (seed !== undefined && typeof seed !== 'string') {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'reset.seed must be a string when provided.');
  }

  const firstPlayer = payload.firstPlayer;
  if (
    firstPlayer !== undefined &&
    firstPlayer !== 'PlayerA' &&
    firstPlayer !== 'PlayerB'
  ) {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'reset.firstPlayer must be "PlayerA" or "PlayerB" when provided.'
    );
  }

  const skipAdvanceToDecision = payload.skipAdvanceToDecision;
  if (
    skipAdvanceToDecision !== undefined &&
    typeof skipAdvanceToDecision !== 'boolean'
  ) {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'reset.skipAdvanceToDecision must be a boolean when provided.'
    );
  }

  return {
    seed,
    firstPlayer,
    serializedState: payload.serializedState,
    skipAdvanceToDecision,
  };
}

function parseObservationPayload(payload: unknown): BridgeObservationPayload {
  if (payload === undefined) {
    return {};
  }

  if (!isObject(payload)) {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'observation payload must be an object.');
  }

  const viewerId = payload.viewerId;
  if (viewerId !== undefined && viewerId !== 'PlayerA' && viewerId !== 'PlayerB') {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'observation.viewerId must be "PlayerA" or "PlayerB" when provided.'
    );
  }

  const includeLegalActionMask = payload.includeLegalActionMask;
  if (
    includeLegalActionMask !== undefined &&
    typeof includeLegalActionMask !== 'boolean'
  ) {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'observation.includeLegalActionMask must be a boolean when provided.'
    );
  }

  return {
    viewerId,
    includeLegalActionMask,
  };
}

function parseStepPayload(payload: unknown): BridgeStepPayload {
  if (!isObject(payload)) {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'step payload must be an object.');
  }

  const action = payload.action;
  const actionKey = payload.actionKey;

  if (action !== undefined && !isObject(action)) {
    throw new RuntimeBridgeError('INVALID_PAYLOAD', 'step.action must be an object when provided.');
  }

  if (actionKey !== undefined && typeof actionKey !== 'string') {
    throw new RuntimeBridgeError(
      'INVALID_PAYLOAD',
      'step.actionKey must be a string when provided.'
    );
  }

  return {
    action: action as GameAction | undefined,
    actionKey,
  };
}

function parseSerializedState(candidate: unknown): GameState {
  if (!isObject(candidate)) {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState must be an object.'
    );
  }

  if (candidate.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      `Unsupported schemaVersion: ${String(candidate.schemaVersion)}.`
    );
  }

  if (typeof candidate.seed !== 'string') {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState.seed must be a string.'
    );
  }

  if (!Array.isArray(candidate.players) || candidate.players.length !== 2) {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState.players must contain exactly 2 players.'
    );
  }

  if (!Array.isArray(candidate.districts)) {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState.districts must be an array.'
    );
  }

  if (typeof candidate.phase !== 'string') {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState.phase must be a string.'
    );
  }

  if (typeof candidate.activePlayerIndex !== 'number') {
    throw new RuntimeBridgeError(
      'STATE_DESERIALIZATION_FAILED',
      'serializedState.activePlayerIndex must be a number.'
    );
  }

  return candidate as unknown as GameState;
}

function success<TResult>(
  requestId: string,
  result: TResult
): BridgeSuccessEnvelope<TResult> {
  return {
    requestId,
    ok: true,
    result,
  };
}

function failure(
  requestId: string,
  error: RuntimeBridgeError
): BridgeFailureEnvelope {
  return {
    requestId,
    ok: false,
    error: {
      code: error.code,
      message: error.message,
      details: error.details,
    },
  };
}

function toBridgeError(error: unknown): RuntimeBridgeError {
  if (error instanceof RuntimeBridgeError) {
    return error;
  }

  if (error instanceof Error) {
    if (error.message.includes('Illegal action')) {
      return new RuntimeBridgeError('ILLEGAL_ACTION', error.message);
    }

    return new RuntimeBridgeError('INTERNAL_ENGINE_ERROR', error.message);
  }

  return new RuntimeBridgeError(
    'INTERNAL_ENGINE_ERROR',
    `Unknown error: ${String(error)}`
  );
}

function extractRequestId(value: unknown): string | undefined {
  if (!isObject(value)) {
    return undefined;
  }
  return typeof value.requestId === 'string' ? value.requestId : undefined;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function cloneForWire<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
