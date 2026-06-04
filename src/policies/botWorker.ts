import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import { createPolicyFromBotSpec } from './botSpec';
import type { ActionPolicy, SearchDecisionDiagnostics } from './types';
import type {
  BotWorkerRequest,
  BotWorkerResponse,
  BotWorkerSelectActionRequest,
} from './workerBotProtocol';

interface BotWorkerGlobalScope {
  onmessage: ((event: MessageEvent<BotWorkerRequest>) => void) | null;
  postMessage(message: BotWorkerResponse): void;
}

const workerScope = globalThis as unknown as BotWorkerGlobalScope;
const policyBySpecKey = new Map<string, ActionPolicy>();
const cancelledRequestIds = new Set<number>();

workerScope.onmessage = (event) => {
  void handleRequest(event.data).catch((error: unknown) => {
    const requestId =
      typeof event.data === 'object' && event.data !== null
        ? event.data.requestId
        : -1;
    postError(requestId, error);
  });
};

async function handleRequest(request: BotWorkerRequest): Promise<void> {
  switch (request.type) {
    case 'cancel':
      cancelledRequestIds.add(request.requestId);
      return;
    case 'select-action':
      await selectAction(request);
      return;
  }
}

async function selectAction(
  request: BotWorkerSelectActionRequest
): Promise<void> {
  if (cancelledRequestIds.has(request.requestId)) {
    cancelledRequestIds.delete(request.requestId);
    postSelectedAction({
      requestId: request.requestId,
    });
    return;
  }

  const policy = policyForSpec(request);
  let diagnostics: SearchDecisionDiagnostics | undefined;
  const selected = await Promise.resolve(
    policy.selectAction({
      state: request.state,
      view: request.view,
      legalActions: request.legalActions,
      random: rngFromSeed(request.randomSeed),
      onSearchDiagnostics(value) {
        if (diagnostics) {
          throw new Error('Worker policy emitted duplicate search diagnostics.');
        }
        diagnostics = structuredClone(value);
      },
    })
  );

  if (cancelledRequestIds.has(request.requestId)) {
    cancelledRequestIds.delete(request.requestId);
    postSelectedAction({
      requestId: request.requestId,
    });
    return;
  }

  postSelectedAction({
    requestId: request.requestId,
    ...(selected ? { actionKey: actionStableKey(selected) } : {}),
    ...(diagnostics ? { diagnostics } : {}),
  });
}

function policyForSpec(request: BotWorkerSelectActionRequest): ActionPolicy {
  const specKey = JSON.stringify(request.spec);
  const cached = policyBySpecKey.get(specKey);
  if (cached) {
    return cached;
  }
  const created = createPolicyFromBotSpec(request.spec);
  policyBySpecKey.set(specKey, created);
  return created;
}

function postSelectedAction(response: {
  requestId: number;
  actionKey?: string;
  diagnostics?: SearchDecisionDiagnostics;
}): void {
  workerScope.postMessage({
    type: 'selected-action',
    ...response,
  });
}

function postError(requestId: number, error: unknown): void {
  const normalized =
    error instanceof Error ? error : new Error(String(error));
  workerScope.postMessage({
    type: 'error',
    requestId,
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
