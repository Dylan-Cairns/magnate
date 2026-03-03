import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { toPlayerView } from '../engine/view';
import type { BotSpec } from './botSpec';
import { policyRandomSeedForState } from './policyRandom';
import type { ActionSelectionContext } from './types';
import {
  createWorkerBackedPolicy,
  type WorkerBackedPolicyWorker,
} from './workerPolicy';
import type {
  BotWorkerRequest,
  BotWorkerResponse,
} from './workerBotProtocol';

describe('worker-backed policy', () => {
  it('returns single legal actions without creating a worker', async () => {
    const workers: FakeWorker[] = [];
    const { context, actions } = selectionFixture();
    const policy = createWorkerBackedPolicy(searchSpec(), {
      createWorker: () => pushWorker(workers),
    });

    const selected = await Promise.resolve(
      policy.selectAction({
        ...context,
        legalActions: [actions[0]],
      })
    );

    expect(selected).toBe(actions[0]);
    expect(workers).toHaveLength(0);
  });

  it('sends a cloneable request and maps the returned action key', async () => {
    const workers: FakeWorker[] = [];
    const spec = searchSpec();
    const { context, state, actions } = selectionFixture();
    const diagnostics = {
      kind: 'search' as const,
      legalRootActions: actions.length,
      expandedRootActions: 2,
      rootVisitBudget: 4,
      configProxyCost: 16,
      maxSimulatedActionSteps: 20,
      simulatedActionSteps: 18,
      terminalRollouts: 1,
    };
    const emittedDiagnostics: unknown[] = [];
    const policy = createWorkerBackedPolicy(spec, {
      createWorker: () => pushWorker(workers),
    });

    const selectedPromise = Promise.resolve(
      policy.selectAction({
        ...context,
        onSearchDiagnostics(value) {
          emittedDiagnostics.push(value);
        },
      })
    );

    expect(workers).toHaveLength(1);
    const request = workers[0].messages[0];
    expect(request).toMatchObject({
      type: 'select-action',
      requestId: 1,
      spec,
      randomSeed: policyRandomSeedForState(state, spec.id),
    });

    const selectedAction = actions[1];
    workers[0].emit({
      type: 'selected-action',
      requestId: request.requestId,
      actionKey: actionStableKey(selectedAction),
      diagnostics,
    });

    await expect(selectedPromise).resolves.toBe(selectedAction);
    expect(emittedDiagnostics).toEqual([diagnostics]);
  });

  it('rejects worker-selected action keys outside the current legal set', async () => {
    const workers: FakeWorker[] = [];
    const { context } = selectionFixture();
    const policy = createWorkerBackedPolicy(searchSpec(), {
      createWorker: () => pushWorker(workers),
    });

    const selectedPromise = Promise.resolve(policy.selectAction(context));
    workers[0].emit({
      type: 'selected-action',
      requestId: workers[0].messages[0].requestId,
      actionKey: 'not-a-legal-action',
    });

    await expect(selectedPromise).rejects.toThrow('illegal action key');
  });

  it('supersedes pending work without letting stale responses select actions', async () => {
    const workers: FakeWorker[] = [];
    const { context, actions } = selectionFixture();
    const policy = createWorkerBackedPolicy(searchSpec(), {
      createWorker: () => pushWorker(workers),
    });

    const stalePromise = Promise.resolve(policy.selectAction(context));
    const freshPromise = Promise.resolve(policy.selectAction(context));

    expect(workers).toHaveLength(2);
    expect(workers[0].terminated).toBe(true);
    await expect(stalePromise).resolves.toBeUndefined();

    workers[0].emit({
      type: 'selected-action',
      requestId: workers[0].messages[0].requestId,
      actionKey: actionStableKey(actions[0]),
    });
    workers[1].emit({
      type: 'selected-action',
      requestId: workers[1].messages[0].requestId,
      actionKey: actionStableKey(actions[1]),
    });

    await expect(freshPromise).resolves.toBe(actions[1]);
  });
});

class FakeWorker implements WorkerBackedPolicyWorker {
  onmessage: ((event: { data: BotWorkerResponse }) => void) | null = null;
  onerror: ((event: { message?: string; error?: unknown }) => void) | null =
    null;
  messages: BotWorkerRequest[] = [];
  terminated = false;

  postMessage(message: BotWorkerRequest): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
    this.onmessage = null;
    this.onerror = null;
  }

  emit(response: BotWorkerResponse): void {
    this.onmessage?.({ data: response });
  }
}

function pushWorker(workers: FakeWorker[]): FakeWorker {
  const worker = new FakeWorker();
  workers.push(worker);
  return worker;
}

function searchSpec(): BotSpec {
  return {
    id: 'worker-search',
    kind: 'search',
    config: {
      worlds: 1,
      rollouts: 1,
      depth: 2,
      maxRootActions: 2,
      rolloutEpsilon: 0,
    },
  };
}

function selectionFixture(): {
  state: ReturnType<typeof createSession>;
  actions: ReturnType<typeof legalActions>;
  context: ActionSelectionContext;
} {
  const state = createSession('worker-policy-test', 'PlayerB');
  const actions = legalActions(state);
  if (actions.length < 2) {
    throw new Error('workerPolicy test fixture requires multiple legal actions.');
  }
  return {
    state,
    actions,
    context: {
      state,
      view: toPlayerView(state, 'PlayerB'),
      legalActions: actions,
      random: rngFromSeed('worker-policy-unused-rng'),
    },
  };
}
