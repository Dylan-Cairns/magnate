import { actionStableKey } from '../engine/actionSurface';
import type { GameAction } from '../engine/types';
import type { BotSpec } from './botSpec';
import { policyRandomSeedForState } from './policyRandom';
import type { ActionPolicy, ActionSelectionContext } from './types';
import type {
  BotWorkerRequest,
  BotWorkerResponse,
} from './workerBotProtocol';

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
}

export interface WorkerBackedActionPolicy extends ActionPolicy {
  close(): void;
}

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
  let nextRequestId = 1;
  const pendingByRequestId = new Map<number, PendingSelection>();
  const createWorker = options.createWorker ?? createDefaultWorker;
  const randomSeedForContext =
    options.randomSeedForContext ??
    ((context: ActionSelectionContext, policySpec: BotSpec) =>
      policyRandomSeedForState(context.state, policySpec.id));

  function ensureWorker(): WorkerBackedPolicyWorker {
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
    terminateWorker();
  }

  function terminateWorker(): void {
    if (!worker) {
      return;
    }
    worker.onmessage = null;
    worker.onerror = null;
    worker.terminate();
    worker = null;
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

    if (response.type === 'error') {
      pending.reject(new Error(response.message));
      return;
    }

    if (response.diagnostics) {
      pending.onSearchDiagnostics?.(structuredClone(response.diagnostics));
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
    const error = event.error instanceof Error ? event.error : new Error(message);
    terminateWorker();
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
        terminateWorker();
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
          });
        } catch (error) {
          pendingByRequestId.delete(requestId);
          terminateWorker();
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
