import { actionStableKey } from '../engine/actionSurface';
import { rngFromSeed } from '../engine/rng';
import type { GameAction } from '../engine/types';
import { createPolicyFromBotSpec, type BotSpec } from './botSpec';
import {
  DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
  preloadTdRootBrowserModel,
} from './modelRuntimeCache';
import {
  rolloutSearchRootBudget,
  selectRolloutSearchActionParallel,
  selectRolloutSearchActionSync,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchRootGuideFactory,
  type RolloutSearchWorkerGuidance,
} from './rolloutSearchCore';
import {
  createSearchWorkerPool,
  type SearchWorkerPool,
} from './searchWorkerPool';
import {
  createTdRootSearchRolloutGuidance,
  createTdRootSearchRootGuide,
} from './tdRootSearchPolicy';
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
let searchWorkerPool: SearchWorkerPool | null = null;
let searchWorkerPoolSize = 0;

type RolloutLikeSearchBotSpec =
  | Extract<BotSpec, { kind: 'search' }>
  | Extract<BotSpec, { kind: 'td-root-search' }>;

interface SearchGuidanceFactories {
  createRootGuide?: RolloutSearchRootGuideFactory;
  rolloutGuidance?: RolloutSearchRuntimeGuidance;
  workerGuidance?: RolloutSearchWorkerGuidance;
}

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

  let diagnostics: SearchDecisionDiagnostics | undefined;
  const captureDiagnostics = (value: SearchDecisionDiagnostics): void => {
    if (diagnostics) {
      throw new Error('Worker policy emitted duplicate search diagnostics.');
    }
    diagnostics = structuredClone(value);
  };
  const selected = isRolloutLikeSearchSpec(request.spec)
    ? await selectSearchAction(request, captureDiagnostics)
    : await selectGenericPolicyAction(request, captureDiagnostics);

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

async function selectGenericPolicyAction(
  request: BotWorkerSelectActionRequest,
  onSearchDiagnostics: (diagnostics: SearchDecisionDiagnostics) => void
): Promise<GameAction | undefined> {
  const policy = policyForSpec(request);
  return await Promise.resolve(
    policy.selectAction({
      state: request.state,
      view: request.view,
      legalActions: request.legalActions,
      random: rngFromSeed(request.randomSeed),
      randomSeed: request.randomSeed,
      onSearchDiagnostics,
    })
  );
}

async function selectSearchAction(
  request: BotWorkerSelectActionRequest,
  onSearchDiagnostics: (diagnostics: SearchDecisionDiagnostics) => void
): Promise<GameAction | undefined> {
  if (!isRolloutLikeSearchSpec(request.spec)) {
    throw new Error(
      `selectSearchAction received unsupported bot kind ${request.spec.kind}.`
    );
  }
  const spec = request.spec;
  const guidance = await createGuidanceForSpec(spec);

  const workerCount = resolveRolloutSearchWorkerCount(request);
  if (workerCount > 1) {
    const pool = ensureSearchWorkerPool(workerCount);
    try {
      return await selectRolloutSearchActionParallel({
        state: request.state,
        view: request.view,
        candidateActions: request.legalActions,
        config: spec.config,
        random: rngFromSeed(request.randomSeed),
        randomSeed: request.randomSeed,
        ...(guidance.createRootGuide
          ? { createRootGuide: guidance.createRootGuide }
          : {}),
        ...(guidance.workerGuidance
          ? { workerGuidance: guidance.workerGuidance }
          : {}),
        batchSize: resolveRolloutSearchBatchSize(request, workerCount),
        parallelWorkers: workerCount,
        onSearchDiagnostics,
        runBatch(tasks, context) {
          return pool.runBatch(tasks, context);
        },
      });
    } catch (error) {
      closeSearchWorkerPool();
      throw error;
    }
  }

  return selectRolloutSearchActionSync({
    state: request.state,
    view: request.view,
    candidateActions: request.legalActions,
    config: spec.config,
    random: rngFromSeed(request.randomSeed),
    randomSeed: request.randomSeed,
    ...(guidance.createRootGuide
      ? { createRootGuide: guidance.createRootGuide }
      : {}),
    ...(guidance.rolloutGuidance
      ? { rolloutGuidance: guidance.rolloutGuidance }
      : {}),
    onSearchDiagnostics,
  });
}

async function createGuidanceForSpec(
  spec: RolloutLikeSearchBotSpec
): Promise<SearchGuidanceFactories> {
  if (spec.kind === 'search') {
    return {};
  }
  const modelIndexPath = spec.modelIndexPath ?? DEFAULT_TD_ROOT_MODEL_INDEX_PATH;
  const model = await preloadTdRootBrowserModel(modelIndexPath);
  return {
    createRootGuide(input) {
      return createTdRootSearchRootGuide({
        ...input,
        model,
      });
    },
    rolloutGuidance: createTdRootSearchRolloutGuidance({ model }),
    workerGuidance: {
      kind: 'td-root',
      modelIndexPath,
    },
  };
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

function ensureSearchWorkerPool(workerCount: number): SearchWorkerPool {
  if (searchWorkerPool && searchWorkerPoolSize === workerCount) {
    return searchWorkerPool;
  }
  closeSearchWorkerPool();
  searchWorkerPool = createSearchWorkerPool({ workerCount });
  searchWorkerPoolSize = workerCount;
  return searchWorkerPool;
}

function closeSearchWorkerPool(): void {
  searchWorkerPool?.close();
  searchWorkerPool = null;
  searchWorkerPoolSize = 0;
}

function resolveRolloutSearchWorkerCount(
  request: BotWorkerSelectActionRequest
): number {
  if (!isRolloutLikeSearchSpec(request.spec)) {
    return 1;
  }
  const rootBudget = rolloutSearchRootBudget(
    request.spec.config,
    request.spec.config.worlds
  );
  const hardwareConcurrency =
    typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency || 1;
  const availableWorkers = Math.max(1, hardwareConcurrency - 2);
  return Math.max(
    1,
    Math.min(8, availableWorkers, rootBudget, request.legalActions.length * 4)
  );
}

function resolveRolloutSearchBatchSize(
  request: BotWorkerSelectActionRequest,
  workerCount: number
): number {
  if (!isRolloutLikeSearchSpec(request.spec)) {
    return 1;
  }
  const rootBudget = rolloutSearchRootBudget(
    request.spec.config,
    request.spec.config.worlds
  );
  return Math.max(1, Math.min(rootBudget, workerCount * 2));
}

function isRolloutLikeSearchSpec(
  spec: BotSpec
): spec is RolloutLikeSearchBotSpec {
  return spec.kind === 'search' || spec.kind === 'td-root-search';
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
  const normalized = error instanceof Error ? error : new Error(String(error));
  workerScope.postMessage({
    type: 'error',
    requestId,
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
}
