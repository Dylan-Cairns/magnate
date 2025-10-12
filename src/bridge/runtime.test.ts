import { describe, expect, it } from 'vitest';

import type { KeyedAction } from '../engine/actionSurface';
import {
  BRIDGE_CONTRACT_NAME,
  BRIDGE_CONTRACT_VERSION,
  BRIDGE_COMMANDS,
  type BridgeLegalActionsResult,
  type BridgeMetadataResult,
  type BridgeObservationResult,
  type BridgeStateResult,
} from './protocol';
import { MagnateBridgeRuntime } from './runtime';

interface RequestEnvelope {
  requestId: string;
  command: string;
  payload?: unknown;
}

function request(runtime: MagnateBridgeRuntime, envelope: RequestEnvelope) {
  return runtime.handleRequest(envelope);
}

function expectOk<TResult>(
  response: ReturnType<MagnateBridgeRuntime['handleRequest']>
): TResult {
  expect(response.ok).toBe(true);
  if (!response.ok) {
    throw new Error(`Expected ok response, got ${response.error.code}: ${response.error.message}`);
  }
  return response.result as TResult;
}

function expectErr(response: ReturnType<MagnateBridgeRuntime['handleRequest']>) {
  expect(response.ok).toBe(false);
  if (response.ok) {
    throw new Error('Expected error response.');
  }
  return response.error;
}

describe('MagnateBridgeRuntime', () => {
  it('returns metadata with contract identity and deterministic action surface declarations', () => {
    const runtime = new MagnateBridgeRuntime();
    const result = expectOk<BridgeMetadataResult>(
      request(runtime, { requestId: 'req-1', command: 'metadata' })
    );

    expect(result.contractName).toBe(BRIDGE_CONTRACT_NAME);
    expect(result.contractVersion).toBe(BRIDGE_CONTRACT_VERSION);
    expect(result.commands).toEqual(BRIDGE_COMMANDS);
    expect(result.actionSurface.canonicalOrder).toBe(
      'ascending_lexicographic_action_key'
    );
  });

  it('reset with same seed returns deterministic state snapshots', () => {
    const runtime = new MagnateBridgeRuntime();

    const first = expectOk<BridgeStateResult>(
      request(runtime, {
        requestId: 'req-2a',
        command: 'reset',
        payload: { seed: 'bridge-seed-1', firstPlayer: 'PlayerA' },
      })
    );

    const second = expectOk<BridgeStateResult>(
      request(runtime, {
        requestId: 'req-2b',
        command: 'reset',
        payload: { seed: 'bridge-seed-1', firstPlayer: 'PlayerA' },
      })
    );

    expect(first.state).toEqual(second.state);
    expect(first.view).toEqual(second.view);
  });

  it('legalActions returns canonical key-sorted actions', () => {
    const runtime = new MagnateBridgeRuntime();
    const result = expectOk<BridgeLegalActionsResult>(
      request(runtime, { requestId: 'req-3', command: 'legalActions' })
    );

    const actionKeys = result.actions.map((entry: KeyedAction) => entry.actionKey);
    const sorted = [...actionKeys].sort();
    expect(actionKeys).toEqual(sorted);
  });

  it('step supports actionKey payloads from legalActions', () => {
    const runtime = new MagnateBridgeRuntime();

    const legal = expectOk<BridgeLegalActionsResult>(
      request(runtime, { requestId: 'req-4a', command: 'legalActions' })
    );
    const actionKey = legal.actions[0]?.actionKey;
    expect(actionKey).toBeDefined();

    const stepped = expectOk<BridgeStateResult>(
      request(runtime, {
        requestId: 'req-4b',
        command: 'step',
        payload: { actionKey },
      })
    );

    expect(stepped.state.log.length).toBeGreaterThan(0);
    expect(typeof stepped.terminal).toBe('boolean');
  });

  it('observation can include legal action key mask for active viewer', () => {
    const runtime = new MagnateBridgeRuntime();

    const observation = expectOk<BridgeObservationResult>(
      request(runtime, {
        requestId: 'req-5',
        command: 'observation',
        payload: { includeLegalActionMask: true },
      })
    );

    expect(Array.isArray(observation.legalActionMask)).toBe(true);
    expect(observation.legalActionMask).toBeDefined();
    expect(observation.legalActionMask?.length ?? 0).toBeGreaterThan(0);
  });

  it('observation mask is empty for non-active viewer', () => {
    const runtime = new MagnateBridgeRuntime();

    const reset = expectOk<BridgeStateResult>(
      request(runtime, {
        requestId: 'req-6a',
        command: 'reset',
        payload: { seed: 'bridge-seed-mask', firstPlayer: 'PlayerA' },
      })
    );

    const viewerId = reset.view.activePlayerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
    const observation = expectOk<BridgeObservationResult>(
      request(runtime, {
        requestId: 'req-6b',
        command: 'observation',
        payload: { viewerId, includeLegalActionMask: true },
      })
    );

    expect(observation.legalActionMask).toEqual([]);
  });

  it('serialize returns canonical state with schemaVersion', () => {
    const runtime = new MagnateBridgeRuntime();
    const result = expectOk<{ state: { schemaVersion: number; seed: string } }>(
      request(runtime, { requestId: 'req-7', command: 'serialize' })
    );

    expect(result.state.schemaVersion).toBe(1);
    expect(typeof result.state.seed).toBe('string');
  });

  it('returns INVALID_COMMAND for unsupported commands', () => {
    const runtime = new MagnateBridgeRuntime();
    const error = expectErr(
      request(runtime, { requestId: 'req-8', command: 'unknown-command' })
    );
    expect(error.code).toBe('INVALID_COMMAND');
  });

  it('returns INVALID_PAYLOAD for malformed request envelope', () => {
    const runtime = new MagnateBridgeRuntime();
    const response = runtime.handleRequest({
      requestId: '',
      command: 'metadata',
    });

    const error = expectErr(response);
    expect(error.code).toBe('INVALID_PAYLOAD');
  });

  it('returns ILLEGAL_ACTION for unknown action key', () => {
    const runtime = new MagnateBridgeRuntime();
    const error = expectErr(
      request(runtime, {
        requestId: 'req-9',
        command: 'step',
        payload: { actionKey: 'not-a-real-action-key' },
      })
    );

    expect(error.code).toBe('ILLEGAL_ACTION');
  });

  it('returns STATE_DESERIALIZATION_FAILED for invalid serializedState payload', () => {
    const runtime = new MagnateBridgeRuntime();
    const error = expectErr(
      request(runtime, {
        requestId: 'req-10',
        command: 'reset',
        payload: { serializedState: { schemaVersion: 999 } },
      })
    );

    expect(error.code).toBe('STATE_DESERIALIZATION_FAILED');
  });
});
