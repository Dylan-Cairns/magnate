import { actionStableKey } from '../engine/actionSurface';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { rngFromSeed } from '../engine/rng';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import type { GameAction, GameState } from '../engine/types';
import { heuristicPolicy } from '../policies/heuristicPolicy';
import {
  DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
  preloadTdRootBrowserModel,
} from '../policies/modelRuntimeCache';
import {
  resolveManifestUrl,
  resolvePublicAssetUrl,
} from '../policies/modelPackUtils';
import {
  policyRandomForState,
  policyRandomSeedForState,
} from '../policies/policyRandom';
import {
  rolloutSearchRootBudget,
  selectRolloutSearchActionParallel,
} from '../policies/rolloutSearchCore';
import type { SearchPolicyConfig } from '../policies/searchConfig';
import {
  createSearchWorkerPool,
  type SearchWorkerPool,
} from '../policies/searchWorkerPool';
import { createTdRootSearchRootGuide } from '../policies/tdRootSearchPolicy';
import type { LoadedTdGuidanceModel } from '../policies/tdGuidanceModel';
import type { SearchDecisionDiagnostics } from '../policies/types';

interface SearchBenchmarkOptions {
  mode: 'smoke' | 'full';
  states: number;
  repetitions: number;
  warmupDecisions: number;
  workers: number;
  transcriptGames: number;
  minimumSpeedup: number;
}

interface DecisionResult {
  actionKey: string;
  diagnostics: SearchDecisionDiagnostics;
  elapsedMs: number;
}

interface MismatchSample {
  stateIndex: number;
  repetition: number;
  scalarActionKey: string;
  resumableScalarActionKey: string;
  pairedActionKey: string;
  scalarMachineDiagnosticsEqual: boolean;
  diagnosticsEqual: boolean;
  scalarRootActions: SearchDecisionDiagnostics['rootActions'];
  pairedRootActions: SearchDecisionDiagnostics['rootActions'];
}

const MEDIUM_CONFIG: SearchPolicyConfig = {
  worlds: 10,
  rollouts: 1,
  depth: 40,
  maxRootActions: 16,
  rolloutEpsilon: 0,
};

self.onmessage = (event: MessageEvent<SearchBenchmarkOptions>) => {
  void runBenchmark(event.data)
    .then((result) => self.postMessage({ ok: true, result }))
    .catch((error: unknown) =>
      self.postMessage({
        ok: false,
        error:
          error instanceof Error
            ? (error.stack ?? error.message)
            : String(error),
      })
    );
};

async function runBenchmark(options: SearchBenchmarkOptions) {
  const [model, provenance, states] = await Promise.all([
    preloadTdRootBrowserModel(),
    loadModelProvenance(),
    collectDecisionStates(options.states),
  ]);
  const scalarPool = createSearchWorkerPool({
    workerCount: options.workers,
    executionMode: 'legacy',
  });
  const pairedPool = createSearchWorkerPool({
    workerCount: options.workers,
    executionMode: 'resumable-paired-td',
  });
  const resumableScalarPool = createSearchWorkerPool({
    workerCount: options.workers,
    executionMode: 'resumable-scalar',
  });

  try {
    const transcripts = await compareTranscripts(
      options.transcriptGames,
      scalarPool,
      pairedPool,
      model,
      options.workers
    );

    for (let index = 0; index < options.warmupDecisions; index += 1) {
      const state = states[index % states.length];
      const seed = `td-two-lane-search:warmup:${String(index)}`;
      await runDecision(state, seed, scalarPool, model, options.workers);
      await runDecision(state, seed, pairedPool, model, options.workers);
      await runDecision(
        state,
        seed,
        resumableScalarPool,
        model,
        options.workers
      );
    }

    const scalarMs: number[] = [];
    const pairedMs: number[] = [];
    let selectedActionMismatchCount = 0;
    let diagnosticsMismatchCount = 0;
    let scalarMachineActionMismatchCount = 0;
    let scalarMachineDiagnosticsMismatchCount = 0;
    const mismatchSamples: MismatchSample[] = [];
    for (
      let repetition = 0;
      repetition < options.repetitions;
      repetition += 1
    ) {
      for (let stateIndex = 0; stateIndex < states.length; stateIndex += 1) {
        const state = states[stateIndex];
        const seed = `td-two-lane-search:state:${String(stateIndex)}`;
        let scalar: DecisionResult;
        let paired: DecisionResult;
        let resumableScalar: DecisionResult;
        if ((repetition + stateIndex) % 2 === 0) {
          scalar = await runDecision(
            state,
            seed,
            scalarPool,
            model,
            options.workers
          );
          paired = await runDecision(
            state,
            seed,
            pairedPool,
            model,
            options.workers
          );
        } else {
          paired = await runDecision(
            state,
            seed,
            pairedPool,
            model,
            options.workers
          );
          scalar = await runDecision(
            state,
            seed,
            scalarPool,
            model,
            options.workers
          );
        }
        resumableScalar = await runDecision(
          state,
          seed,
          resumableScalarPool,
          model,
          options.workers
        );
        scalarMs.push(scalar.elapsedMs);
        pairedMs.push(paired.elapsedMs);
        const actionsEqual = scalar.actionKey === paired.actionKey;
        const diagnosticsEqual =
          JSON.stringify(scalar.diagnostics) ===
          JSON.stringify(paired.diagnostics);
        const scalarMachineActionsEqual =
          scalar.actionKey === resumableScalar.actionKey;
        const scalarMachineDiagnosticsEqual =
          JSON.stringify(scalar.diagnostics) ===
          JSON.stringify(resumableScalar.diagnostics);
        selectedActionMismatchCount += Number(!actionsEqual);
        diagnosticsMismatchCount += Number(!diagnosticsEqual);
        scalarMachineActionMismatchCount += Number(!scalarMachineActionsEqual);
        scalarMachineDiagnosticsMismatchCount += Number(
          !scalarMachineDiagnosticsEqual
        );
        if (
          (!actionsEqual || !diagnosticsEqual) &&
          mismatchSamples.length < 10
        ) {
          mismatchSamples.push({
            stateIndex,
            repetition,
            scalarActionKey: scalar.actionKey,
            resumableScalarActionKey: resumableScalar.actionKey,
            pairedActionKey: paired.actionKey,
            scalarMachineDiagnosticsEqual,
            diagnosticsEqual,
            scalarRootActions: scalar.diagnostics.rootActions,
            pairedRootActions: paired.diagnostics.rootActions,
          });
        }
      }
    }

    const scalar = summarizeTiming(scalarMs);
    const paired = summarizeTiming(pairedMs);
    const speedup = scalar.totalMs / paired.totalMs;
    const exactParity =
      selectedActionMismatchCount === 0 &&
      diagnosticsMismatchCount === 0 &&
      scalarMachineActionMismatchCount === 0 &&
      scalarMachineDiagnosticsMismatchCount === 0 &&
      transcripts.mismatchGames === 0;
    const speedupPassed = speedup >= options.minimumSpeedup;
    return {
      schemaVersion: 1,
      generatedAtUtc: new Date().toISOString(),
      benchmark: 'td-root-search-two-lane-browser-workers',
      mode: options.mode,
      environment: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated: self.crossOriginIsolated,
      },
      options,
      search: {
        profile: 'td-root-search-v2-medium',
        config: MEDIUM_CONFIG,
        rootVisitBudget: rolloutSearchRootBudget(
          MEDIUM_CONFIG,
          MEDIUM_CONFIG.worlds
        ),
        batchSize: batchSize(options.workers),
      },
      model: provenance,
      corpus: { states: states.length },
      correctness: {
        comparedDecisions: states.length * options.repetitions,
        selectedActionMismatchCount,
        diagnosticsMismatchCount,
        scalarMachineActionMismatchCount,
        scalarMachineDiagnosticsMismatchCount,
        mismatchSamples,
        transcripts,
      },
      timing: { scalar, paired, speedup },
      gate: {
        exactParity,
        speedupThreshold: options.minimumSpeedup,
        speedupPassed,
        recommendation:
          options.mode === 'smoke'
            ? 'smoke-only-no-performance-decision'
            : exactParity && speedupPassed
              ? 'consider-production-shadow-validation'
              : exactParity
                ? 'do-not-invest-further-speedup-too-small'
                : 'do-not-invest-further-parity-failed',
      },
    };
  } finally {
    scalarPool.close();
    pairedPool.close();
    resumableScalarPool.close();
  }
}

async function runDecision(
  state: GameState,
  randomSeed: string,
  pool: SearchWorkerPool,
  model: LoadedTdGuidanceModel,
  workers: number
): Promise<DecisionResult> {
  const decisionPlayer = decisionPlayerIdForState(state);
  if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
    throw new Error('Search benchmark could not resolve decision player.');
  }
  const view = toDecisionPlayerView(state, decisionPlayer);
  const candidateActions = legalActionsForDecisionPlayer(state, decisionPlayer);
  if (candidateActions.length < 2) {
    throw new Error('Search benchmark decision requires multiple actions.');
  }
  let diagnostics: SearchDecisionDiagnostics | undefined;
  const startedAt = performance.now();
  const selected = await selectRolloutSearchActionParallel({
    state,
    view,
    candidateActions,
    config: MEDIUM_CONFIG,
    random: rngFromSeed(randomSeed),
    randomSeed,
    createRootGuide(input) {
      return createTdRootSearchRootGuide({ ...input, model });
    },
    workerGuidance: {
      kind: 'td-root',
      modelIndexPath: DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
      rollout: 'td',
      leaf: 'td',
    },
    guidanceKind: 'td-root',
    batchSize: batchSize(workers),
    parallelWorkers: workers,
    onSearchDiagnostics(value) {
      diagnostics = value;
    },
    runBatch(tasks, context) {
      return pool.runBatch(tasks, context);
    },
  });
  const elapsedMs = performance.now() - startedAt;
  if (!selected || !diagnostics) {
    throw new Error(
      'Search benchmark did not return an action and diagnostics.'
    );
  }
  return { actionKey: actionStableKey(selected), diagnostics, elapsedMs };
}

function batchSize(workers: number): number {
  return Math.max(
    1,
    Math.min(
      rolloutSearchRootBudget(MEDIUM_CONFIG, MEDIUM_CONFIG.worlds),
      workers * 2
    )
  );
}

async function collectDecisionStates(count: number): Promise<GameState[]> {
  const states: GameState[] = [];
  for (
    let gameIndex = 0;
    gameIndex < 100 && states.length < count;
    gameIndex += 1
  ) {
    let state = createSession(
      `td-two-lane-search-corpus:game:${String(gameIndex)}`,
      gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB'
    );
    for (
      let decisionIndex = 0;
      decisionIndex < 500 && !isTerminal(state) && states.length < count;
      decisionIndex += 1
    ) {
      const decisionPlayer = decisionPlayerIdForState(state);
      if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
        throw new Error('Could not resolve benchmark corpus decision player.');
      }
      const view = toDecisionPlayerView(state, decisionPlayer);
      const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
      if (actions.length >= 2) {
        states.push(structuredClone(state) as GameState);
      }
      const selected = await Promise.resolve(
        heuristicPolicy.selectAction({
          state,
          view,
          legalActions: actions,
          random: policyRandomForState(state, 'td-two-lane-search-corpus'),
          randomSeed: policyRandomSeedForState(
            state,
            'td-two-lane-search-corpus'
          ),
        })
      );
      if (!selected) {
        throw new Error('Benchmark corpus source policy returned no action.');
      }
      state = stepToDecision(state, selected);
    }
  }
  if (states.length !== count) {
    throw new Error(
      `Collected ${String(states.length)} search states; expected ${String(count)}.`
    );
  }
  return states;
}

async function compareTranscripts(
  gameCount: number,
  scalarPool: SearchWorkerPool,
  pairedPool: SearchWorkerPool,
  model: LoadedTdGuidanceModel,
  workers: number
) {
  let mismatchGames = 0;
  const games = [];
  for (let gameIndex = 0; gameIndex < gameCount; gameIndex += 1) {
    const seed = `td-two-lane-search-transcript:${String(gameIndex)}`;
    const scalar = await runTranscript(seed, scalarPool, model, workers);
    const paired = await runTranscript(seed, pairedPool, model, workers);
    const actionKeysEqual =
      JSON.stringify(scalar.actionKeys) === JSON.stringify(paired.actionKeys);
    const diagnosticsEqual =
      JSON.stringify(scalar.diagnostics) === JSON.stringify(paired.diagnostics);
    const finalStateEqual =
      JSON.stringify(scalar.finalState) === JSON.stringify(paired.finalState);
    const equal = actionKeysEqual && diagnosticsEqual && finalStateEqual;
    mismatchGames += Number(!equal);
    games.push({
      gameIndex,
      decisions: scalar.actionKeys.length,
      actionKeysEqual,
      diagnosticsEqual,
      finalStateEqual,
    });
  }
  return { comparedGames: gameCount, mismatchGames, games };
}

async function runTranscript(
  seed: string,
  pool: SearchWorkerPool,
  model: LoadedTdGuidanceModel,
  workers: number
) {
  let state = createSession(seed, 'PlayerA');
  const actionKeys: string[] = [];
  const diagnostics: SearchDecisionDiagnostics[] = [];
  for (
    let decisionIndex = 0;
    decisionIndex < 500 && !isTerminal(state);
    decisionIndex += 1
  ) {
    const decisionPlayer = decisionPlayerIdForState(state);
    if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
      throw new Error(
        'Transcript benchmark could not resolve decision player.'
      );
    }
    const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
    let selected: GameAction | undefined = actions[0];
    if (actions.length > 1) {
      const result = await runDecision(
        state,
        `${seed}:decision:${String(decisionIndex)}`,
        pool,
        model,
        workers
      );
      selected = actions.find(
        (action) => actionStableKey(action) === result.actionKey
      );
      diagnostics.push(result.diagnostics);
    }
    if (!selected) {
      throw new Error('Transcript benchmark could not select a legal action.');
    }
    actionKeys.push(actionStableKey(selected));
    state = stepToDecision(state, selected);
  }
  if (!isTerminal(state)) {
    throw new Error('Transcript benchmark exceeded 500 decisions.');
  }
  return { actionKeys, diagnostics, finalState: state };
}

function summarizeTiming(values: readonly number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const totalMs = values.reduce((total, value) => total + value, 0);
  return {
    samples: values.length,
    totalMs,
    meanMs: totalMs / values.length,
    p50Ms: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
  };
}

function percentile(sorted: readonly number[], fraction: number): number {
  const index = Math.max(0, Math.ceil(sorted.length * fraction) - 1);
  return sorted[index];
}

async function loadModelProvenance() {
  const indexUrl = resolvePublicAssetUrl(DEFAULT_TD_ROOT_MODEL_INDEX_PATH);
  const indexResponse = await fetch(indexUrl);
  if (!indexResponse.ok) {
    throw new Error(`Could not fetch model index: ${indexResponse.status}.`);
  }
  const index = (await indexResponse.json()) as {
    defaultPackId: string | null;
    packs: Array<{ id: string; manifestPath: string }>;
  };
  const selected =
    index.packs.find((pack) => pack.id === index.defaultPackId) ??
    index.packs[0];
  if (!selected) {
    throw new Error('Model index does not contain a TD pack.');
  }
  const manifestUrl = resolveManifestUrl(indexUrl, selected.manifestPath);
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) {
    throw new Error(
      `Could not fetch model manifest: ${manifestResponse.status}.`
    );
  }
  const manifest = (await manifestResponse.json()) as {
    packId: string;
    model: { weightsPath: string };
  };
  const weightsUrl = new URL(manifest.model.weightsPath, manifestUrl);
  const weightsResponse = await fetch(weightsUrl);
  if (!weightsResponse.ok) {
    throw new Error(
      `Could not fetch model weights: ${weightsResponse.status}.`
    );
  }
  const digest = await crypto.subtle.digest(
    'SHA-256',
    await weightsResponse.arrayBuffer()
  );
  return {
    indexPath: DEFAULT_TD_ROOT_MODEL_INDEX_PATH,
    packId: manifest.packId,
    weightsUrl: weightsUrl.toString(),
    weightsSha256: Array.from(new Uint8Array(digest), (value) =>
      value.toString(16).padStart(2, '0')
    ).join(''),
  };
}
