import { actionStableKey } from '../engine/actionSurface';
import type { GameAction } from '../engine/types';
import type { BotSpec } from './botSpec';
import { policyRandomSeedForState } from './policyRandom';
import type { ActionPolicy, ActionSelectionContext } from './types';
import { browserSearchExecutionModeOverride } from './searchExecutionRuntime';
import type { SearchWorkerExecutionMode } from './searchWorkerProtocol';
import type { BotWorkerRequest, BotWorkerResponse } from './workerBotProtocol';

export interface WorkerBackedPolicyWorker {
  onmessage: ((event: { data: BotWorkerResponse }) => void) | null;
  onerror: ((event: { message?: string; error?: unknown }) => void) | null;
  postMessage(message: BotWorkerRequest): void;
  terminate(): void;
}

export interface WorkerBackedPolicyOptions {
  createWorker?: () => WorkerBackedPolicyWorker;
  randomSeedForContext?: (
    context: ActionSelectionContext,
    spec: BotSpec
  ) => string;
  /**
   * Explicit worker executor override. When omitted, a browser URL override is
   * forwarded if present and the bot worker resolves the eligible default.
   */
  searchExecutionMode?: SearchWorkerExecutionMode;
  onSearchExecutionMode?: (
    mode: SearchWorkerExecutionMode | 'synchronous'
  ) => void;
}

export interface WorkerBackedActionPolicy extends ActionPolicy {
  close(): void;
}

/** Grace period between a cooperative shutdown and a hard `terminate()`. */
export const WORKER_SHUTDOWN_GRACE_MS = 250;
/** Warm-pool lifetime without a bot decision before teardown. */
export const WORKER_IDLE_SHUTDOWN_MS = 5 * 60_000;

interface PendingSelection {
  legalActions: readonly GameAction[];
  onSearchDiagnostics: ActionSelectionContext['onSearchDiagnostics'];
  resolve(action: GameAction | undefined): void;
  reject(error: Error): void;
}

export function createWorkerBackedPolicy(
  spec: BotSpec,
  options: WorkerBackedPolicyOptions = {}
): WorkerBackedActionPolicy {
  let worker: WorkerBackedPolicyWorker | null = null;
  let idleShutdownTimer: ReturnType<typeof setTimeout> | null = null;
  let nextRequestId = 1;
  const pendingByRequestId = new Map<number, PendingSelection>();
  const createWorker = options.createWorker ?? createDefaultWorker;
  const randomSeedForContext =
    options.randomSeedForContext ??
    ((context: ActionSelectionContext, policySpec: BotSpec) =>
      context.randomSeed ??
      policyRandomSeedForState(context.state, policySpec.id));
  const searchExecutionMode =
    options.searchExecutionMode ?? browserSearchExecutionModeOverride();

  function ensureWorker(): WorkerBackedPolicyWorker {
    cancelIdleShutdown();
    if (worker) {
      return worker;
    }
    const created = createWorker();
    created.onmessage = handleMessage;
    created.onerror = handleWorkerError;
    worker = created;
    return created;
  }

  function close(): void {
    settlePendingAsSuperseded();
    shutdownWorker();
  }

  function shutdownWorker(): void {
    cancelIdleShutdown();
    const current = worker;
    if (!current) {
      return;
    }
    worker = null;
    current.onmessage = null;
    current.onerror = null;
    try {
      current.postMessage({ type: 'shutdown' });
    } catch {
      current.terminate();
      return;
    }
    // The shutdown request lets the worker close its nested search pool before
    // it stops; the fallback guarantees teardown even if it is never processed.
    setTimeout(() => {
      current.terminate();
    }, WORKER_SHUTDOWN_GRACE_MS);
  }

  function scheduleIdleShutdown(): void {
    cancelIdleShutdown();
    idleShutdownTimer = setTimeout(() => {
      idleShutdownTimer = null;
      shutdownWorker();
    }, WORKER_IDLE_SHUTDOWN_MS);
  }

  function cancelIdleShutdown(): void {
    if (idleShutdownTimer === null) {
      return;
    }
    clearTimeout(idleShutdownTimer);
    idleShutdownTimer = null;
  }

  function settlePendingAsSuperseded(): void {
    for (const pending of pendingByRequestId.values()) {
      pending.resolve(undefined);
    }
    pendingByRequestId.clear();
  }

  function handleMessage(event: { data: BotWorkerResponse }): void {
    const response = event.data;
    const pending = pendingByRequestId.get(response.requestId);
    if (!pending) {
      return;
    }
    pendingByRequestId.delete(response.requestId);
    scheduleIdleShutdown();

    if (response.type === 'error') {
      pending.reject(new Error(response.message));
      return;
    }

    if (response.diagnostics) {
      pending.onSearchDiagnostics?.(structuredClone(response.diagnostics));
    }
    if (response.searchExecutionMode) {
      options.onSearchExecutionMode?.(response.searchExecutionMode);
    }
    if (!response.actionKey) {
      pending.resolve(undefined);
      return;
    }

    const selected = pending.legalActions.find(
      (action) => actionStableKey(action) === response.actionKey
    );
    if (!selected) {
      pending.reject(
        new Error(
          `Worker-backed policy selected an illegal action key: ${response.actionKey}.`
        )
      );
      return;
    }
    pending.resolve(selected);
  }

  function handleWorkerError(event: {
    message?: string;
    error?: unknown;
  }): void {
    const message = event.message ?? 'Bot worker failed.';
    const error =
      event.error instanceof Error ? event.error : new Error(message);
    shutdownWorker();
    for (const pending of pendingByRequestId.values()) {
      pending.reject(error);
    }
    pendingByRequestId.clear();
  }

  return {
    selectAction(context) {
      if (context.legalActions.length === 0) {
        return undefined;
      }
      if (context.legalActions.length === 1) {
        return context.legalActions[0];
      }

      if (pendingByRequestId.size > 0) {
        settlePendingAsSuperseded();
        shutdownWorker();
      }

      const requestId = nextRequestId;
      nextRequestId += 1;

      return new Promise<GameAction | undefined>((resolve, reject) => {
        pendingByRequestId.set(requestId, {
          legalActions: context.legalActions,
          onSearchDiagnostics: context.onSearchDiagnostics,
          resolve,
          reject,
        });

        try {
          ensureWorker().postMessage({
            type: 'select-action',
            requestId,
            spec,
            state: context.state,
            view: context.view,
            legalActions: [...context.legalActions],
            randomSeed: randomSeedForContext(context, spec),
            ...(context.onSearchDiagnostics
              ? { collectDiagnostics: true }
              : {}),
            ...(searchExecutionMode ? { searchExecutionMode } : {}),
          });
        } catch (error) {
          pendingByRequestId.delete(requestId);
          shutdownWorker();
          reject(error instanceof Error ? error : new Error(String(error)));
        }
      });
    },
    close,
  };
}

function createDefaultWorker(): WorkerBackedPolicyWorker {
  return new Worker(new URL('./botWorker.ts', import.meta.url), {
    type: 'module',
  }) as unknown as WorkerBackedPolicyWorker;
}
