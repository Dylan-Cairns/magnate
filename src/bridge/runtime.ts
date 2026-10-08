import { BridgeCommand, BridgeErrorCode } from './values';
import { GamePhase, Ruleset } from '../engine/values';
import {
  ACTION_IDS,
  actionStableKey,
  legalActionsCanonical,
  toKeyedActions,
} from '../engine/actionSurface';
import { createSession } from '../engine/session';
import { applyAction } from '../engine/reducer';
import { isTerminal } from '../engine/scoring';
import { advanceToDecision } from '../engine/turnFlow';
import { type GameAction, type GameState, PlayerId } from '../engine/types';
import { toPlayerView } from '../engine/view';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import type {
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
const DEFAULT_FIRST_PLAYER: PlayerId = PlayerId.PlayerA;
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
      case BridgeCommand.Metadata:
        return this.metadata();
      case BridgeCommand.Reset:
        return this.reset(payload);
      case BridgeCommand.LegalActions:
        return this.legalActions(payload);
      case BridgeCommand.Observation:
        return this.observation(payload);
      case BridgeCommand.Step:
        return this.step(payload);
      case BridgeCommand.Serialize:
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
        defaultViewer: 'decision-player',
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
      this.state = parsed.skipAdvanceToDecision
        ? deserialized
        : advanceToDecision(deserialized);
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
        BridgeErrorCode.InvalidPayload,
        'legalActions payload must be an object when provided.'
      );
    }

    return {
      actions: cloneForWire(this.decisionLegalActionsCanonical()),
      activePlayerId: this.decisionPlayerId(),
      phase: this.state.phase,
    };
  }

  private observation(payload: unknown): BridgeObservationResult {
    const parsed = parseObservationPayload(payload);
    const decisionPlayerId = this.decisionPlayerId();
    const viewerId = parsed.viewerId ?? decisionPlayerId;
    const view =
      viewerId === decisionPlayerId
        ? toDecisionPlayerView(this.state, decisionPlayerId)
        : toPlayerView(this.state, viewerId);

    if (!parsed.includeLegalActionMask) {
      return { view: cloneForWire(view) };
    }

    const legalActionMask =
      viewerId === decisionPlayerId
        ? this.decisionLegalActionsCanonical().map((entry) => entry.actionKey)
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
        throw new RuntimeBridgeError(
          BridgeErrorCode.IllegalAction,
          error.message
        );
      }
      throw error;
    }

    return this.stateResult();
  }

  private serialize(payload: unknown): { state: GameState } {
    if (payload !== undefined && !isObject(payload)) {
      throw new RuntimeBridgeError(
        BridgeErrorCode.InvalidPayload,
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
      const match = this.decisionLegalActionsCanonical().find(
        (candidate) => candidate.actionKey === key
      );
      if (!match) {
        throw new RuntimeBridgeError(
          BridgeErrorCode.IllegalAction,
          `Unknown legal action key: ${key}`
        );
      }

      if (payload.action) {
        const payloadKey = actionStableKey(payload.action);
        if (payloadKey !== key) {
          throw new RuntimeBridgeError(
            BridgeErrorCode.InvalidPayload,
            'step payload action and actionKey must refer to the same action.',
            { actionKey: key, payloadActionKey: payloadKey }
          );
        }
      }

      return match.action;
    }

    if (payload.action) {
      const payloadKey = actionStableKey(payload.action);
      const match = this.decisionLegalActionsCanonical().find(
        (candidate) => candidate.actionKey === payloadKey
      );
      if (!match) {
        throw new RuntimeBridgeError(
          BridgeErrorCode.IllegalAction,
          `Unknown legal action key: ${payloadKey}`
        );
      }
      return match.action;
    }

    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'step payload requires either action or actionKey.'
    );
  }

  private stateResult(): BridgeStateResult {
    return {
      state: cloneForWire(this.state),
      view: cloneForWire(toDecisionPlayerView(this.state)),
      terminal: isTerminal(this.state),
    };
  }

  private decisionPlayerId(): PlayerId {
    const decisionPlayerId = decisionPlayerIdForState(this.state);
    if (
      decisionPlayerId !== PlayerId.PlayerA &&
      decisionPlayerId !== PlayerId.PlayerB
    ) {
      throw new RuntimeBridgeError(
        BridgeErrorCode.InternalEngineError,
        'Could not resolve bridge decision player.'
      );
    }
    return decisionPlayerId;
  }

  private decisionLegalActionsCanonical() {
    const decisionPlayerId = this.decisionPlayerId();
    if (this.state.phase === GamePhase.CollectIncome) {
      return toKeyedActions(
        legalActionsForDecisionPlayer(this.state, decisionPlayerId)
      );
    }
    return legalActionsCanonical(this.state);
  }
}

function parseEnvelope(raw: unknown): {
  requestId: string;
  command: BridgeCommand;
  payload: unknown;
} {
  if (!isObject(raw)) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'Request must be a JSON object.'
    );
  }

  const requestId = raw.requestId;
  if (typeof requestId !== 'string' || requestId.trim() === '') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'Request field "requestId" must be a non-empty string.'
    );
  }

  const commandValue = raw.command;
  if (typeof commandValue !== 'string') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
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

  throw new RuntimeBridgeError(
    BridgeErrorCode.InvalidCommand,
    `Unsupported command: ${command}`
  );
}

function parseResetPayload(payload: unknown): BridgeResetPayload {
  if (payload === undefined) {
    return {};
  }

  if (!isObject(payload)) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'reset payload must be an object.'
    );
  }

  const seed = payload.seed;
  if (seed !== undefined && typeof seed !== 'string') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'reset.seed must be a string when provided.'
    );
  }

  const firstPlayer = payload.firstPlayer;
  if (
    firstPlayer !== undefined &&
    firstPlayer !== PlayerId.PlayerA &&
    firstPlayer !== PlayerId.PlayerB
  ) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'reset.firstPlayer must be "PlayerA" or "PlayerB" when provided.'
    );
  }

  const skipAdvanceToDecision = payload.skipAdvanceToDecision;
  if (
    skipAdvanceToDecision !== undefined &&
    typeof skipAdvanceToDecision !== 'boolean'
  ) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
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
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'observation payload must be an object.'
    );
  }

  const viewerId = payload.viewerId;
  if (
    viewerId !== undefined &&
    viewerId !== PlayerId.PlayerA &&
    viewerId !== PlayerId.PlayerB
  ) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'observation.viewerId must be "PlayerA" or "PlayerB" when provided.'
    );
  }

  const includeLegalActionMask = payload.includeLegalActionMask;
  if (
    includeLegalActionMask !== undefined &&
    typeof includeLegalActionMask !== 'boolean'
  ) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
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
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'step payload must be an object.'
    );
  }

  const action = payload.action;
  const actionKey = payload.actionKey;

  if (action !== undefined && !isObject(action)) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
      'step.action must be an object when provided.'
    );
  }

  if (actionKey !== undefined && typeof actionKey !== 'string') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.InvalidPayload,
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
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState must be an object.'
    );
  }

  if (candidate.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      `Unsupported schemaVersion: ${String(candidate.schemaVersion)}.`
    );
  }

  if (typeof candidate.seed !== 'string') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState.seed must be a string.'
    );
  }

  if (!Array.isArray(candidate.players) || candidate.players.length !== 2) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState.players must contain exactly 2 players.'
    );
  }

  if (!Array.isArray(candidate.districts)) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState.districts must be an array.'
    );
  }

  if (typeof candidate.phase !== 'string') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState.phase must be a string.'
    );
  }

  if (typeof candidate.activePlayerIndex !== 'number') {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'serializedState.activePlayerIndex must be a number.'
    );
  }

  // The Python training/eval bridge stays on the standard ruleset.
  if (
    candidate.ruleset !== undefined &&
    candidate.ruleset !== Ruleset.Standard
  ) {
    throw new RuntimeBridgeError(
      BridgeErrorCode.StateDeserializationFailed,
      'The bridge supports the standard ruleset only.'
    );
  }
  candidate.ruleset = Ruleset.Standard;

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
      return new RuntimeBridgeError(
        BridgeErrorCode.IllegalAction,
        error.message
      );
    }

    return new RuntimeBridgeError(
      BridgeErrorCode.InternalEngineError,
      error.message
    );
  }

  return new RuntimeBridgeError(
    BridgeErrorCode.InternalEngineError,
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
