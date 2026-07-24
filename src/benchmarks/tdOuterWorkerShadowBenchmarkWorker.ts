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
import { getBotProfile } from '../policies/catalog';
import { heuristicPolicy } from '../policies/heuristicPolicy';
import { DEFAULT_TD_ROOT_MODEL_INDEX_PATH } from '../policies/modelRuntimeCache';
import {
  resolveManifestUrl,
  resolvePublicAssetUrl,
} from '../policies/modelPackUtils';
import {
  policyRandomForState,
  policyRandomSeedForState,
} from '../policies/policyRandom';
import type { SearchWorkerExecutionMode } from '../policies/searchWorkerProtocol';
import type { SearchDecisionDiagnostics } from '../policies/types';
import {
  createWorkerBackedPolicy,
  type WorkerBackedActionPolicy,
} from '../policies/workerPolicy';
import { legacyAuthoritativeValue } from './shadowAuthority';

interface OuterShadowOptions {
  mode: 'smoke' | 'full';
  states: number;
  repetitions: number;
  warmupDecisions: number;
  shadowGames: number;
  minimumSpeedup: number;
}

interface CorpusState {
  state: GameState;
  sourceGameIndex: number;
  sourceDecisionIndex: number;
  legalActionTypes: readonly string[];
}

interface DecisionResult {
  action: GameAction;
  actionKey: string;
  diagnostics: SearchDecisionDiagnostics;
  elapsedMs: number;
}

interface MismatchSample {
  scope: 'corpus' | 'shadow-game';
  stateIndex?: number;
  repetition?: number;
  gameIndex?: number;
  decisionIndex?: number;
  legacyActionKey: string;
  pairedActionKey: string;
  diagnosticsEqual: boolean;
  legacyRootActions: SearchDecisionDiagnostics['rootActions'];
  pairedRootActions: SearchDecisionDiagnostics['rootActions'];
}

type EffectiveSearchExecutionMode = SearchWorkerExecutionMode | 'synchronous';

self.onmessage = (event: MessageEvent<OuterShadowOptions>) => {
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

async function runBenchmark(options: OuterShadowOptions) {
  const profile = getBotProfile('td-root-search-v2-medium');
  if (profile.spec.kind !== 'td-root-search') {
    throw new Error('TD Medium catalog profile is not TD-root search.');
  }
  const spec = structuredClone(profile.spec);
  const [provenance, corpus] = await Promise.all([
    loadModelProvenance(),
    collectDecisionStates(options.states),
  ]);
  const observedLegacyExecutionModes = new Set<EffectiveSearchExecutionMode>();
  const observedCandidateExecutionModes =
    new Set<EffectiveSearchExecutionMode>();
  const legacy = createWorkerBackedPolicy(spec, {
    searchExecutionMode: 'legacy',
    onSearchExecutionMode(mode) {
      observedLegacyExecutionModes.add(mode);
    },
  });
  const paired = createWorkerBackedPolicy(spec, {
    onSearchExecutionMode(mode) {
      observedCandidateExecutionModes.add(mode);
    },
  });

  try {
    for (let index = 0; index < options.warmupDecisions; index += 1) {
      const entry = corpus[index % corpus.length];
      const seed = `td-outer-shadow:warmup:${String(index)}`;
      await runDecision(entry.state, seed, legacy);
      await runDecision(entry.state, seed, paired);
    }

    const mismatchSamples: MismatchSample[] = [];
    let selectedActionMismatchCount = 0;
    let diagnosticsMismatchCount = 0;
    const legacyMs: number[] = [];
    const pairedMs: number[] = [];
    const parallelWorkers = new Set<number>();
    const parallelBatchSizes = new Set<number>();
    const rootVisitBudgets = new Set<number>();

    for (
      let repetition = 0;
      repetition < options.repetitions;
      repetition += 1
    ) {
      for (let stateIndex = 0; stateIndex < corpus.length; stateIndex += 1) {
        const entry = corpus[stateIndex];
        const seed = `td-outer-shadow:corpus:${String(stateIndex)}`;
        let legacyResult: DecisionResult;
        let pairedResult: DecisionResult;
        if ((repetition + stateIndex) % 2 === 0) {
          legacyResult = await runDecision(entry.state, seed, legacy);
          pairedResult = await runDecision(entry.state, seed, paired);
        } else {
          pairedResult = await runDecision(entry.state, seed, paired);
          legacyResult = await runDecision(entry.state, seed, legacy);
        }
        legacyMs.push(legacyResult.elapsedMs);
        pairedMs.push(pairedResult.elapsedMs);
        collectExecutionMetadata(
          legacyResult.diagnostics,
          parallelWorkers,
          parallelBatchSizes,
          rootVisitBudgets
        );
        collectExecutionMetadata(
          pairedResult.diagnostics,
          parallelWorkers,
          parallelBatchSizes,
          rootVisitBudgets
        );
        const actionsEqual = legacyResult.actionKey === pairedResult.actionKey;
        const diagnosticsEqual = exactJsonEqual(
          legacyResult.diagnostics,
          pairedResult.diagnostics
        );
        selectedActionMismatchCount += Number(!actionsEqual);
        diagnosticsMismatchCount += Number(!diagnosticsEqual);
        maybeRecordMismatch(mismatchSamples, {
          scope: 'corpus',
          stateIndex,
          repetition,
          legacyActionKey: legacyResult.actionKey,
          pairedActionKey: pairedResult.actionKey,
          diagnosticsEqual,
          legacyRootActions: legacyResult.diagnostics.rootActions,
          pairedRootActions: pairedResult.diagnostics.rootActions,
        });
      }
    }

    const shadowGames = [];
    let shadowActionMismatchCount = 0;
    let shadowDiagnosticsMismatchCount = 0;
    for (let gameIndex = 0; gameIndex < options.shadowGames; gameIndex += 1) {
      const game = await runShadowGame(
        gameIndex,
        legacy,
        paired,
        mismatchSamples,
        parallelWorkers,
        parallelBatchSizes,
        rootVisitBudgets
      );
      shadowGames.push(game);
      shadowActionMismatchCount += game.actionMismatchCount;
      shadowDiagnosticsMismatchCount += game.diagnosticsMismatchCount;
    }

    const legacyTiming = summarizeTiming(legacyMs);
    const pairedTiming = summarizeTiming(pairedMs);
    const speedup = legacyTiming.totalMs / pairedTiming.totalMs;
    const exactParity =
      selectedActionMismatchCount === 0 &&
      diagnosticsMismatchCount === 0 &&
      shadowActionMismatchCount === 0 &&
      shadowDiagnosticsMismatchCount === 0;
    const speedupPassed = speedup >= options.minimumSpeedup;
    const p95NonRegression = pairedTiming.p95Ms <= legacyTiming.p95Ms;
    const executionModeRouting = {
      requestedLegacyOverride: 'legacy',
      requestedCandidateOverride: null,
      observedLegacy: [...observedLegacyExecutionModes].sort(),
      observedCandidate: [...observedCandidateExecutionModes].sort(),
      passed:
        setContainsOnly(observedLegacyExecutionModes, 'legacy') &&
        setContainsOnly(observedCandidateExecutionModes, 'resumable-paired-td'),
    };

    return {
      schemaVersion: 1,
      generatedAtUtc: new Date().toISOString(),
      benchmark: 'td-root-search-outer-worker-shadow',
      mode: options.mode,
      environment: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        crossOriginIsolated: self.crossOriginIsolated,
      },
      options,
      policy: {
        profileId: profile.id,
        spec,
        legacyExecutionMode: 'legacy',
        candidateExecutionMode: 'resumable-paired-td',
        candidateExecutionModeRequest: 'omitted-production-default',
        authority: 'legacy',
      },
      execution: {
        modeRouting: executionModeRouting,
        parallelWorkers: [...parallelWorkers].sort(
          (left, right) => left - right
        ),
        parallelBatchSizes: [...parallelBatchSizes].sort(
          (left, right) => left - right
        ),
        rootVisitBudgets: [...rootVisitBudgets].sort(
          (left, right) => left - right
        ),
      },
      model: provenance,
      corpus: {
        states: corpus.length,
        fingerprintSha256: await corpusFingerprint(corpus),
        sourceGames: new Set(corpus.map((entry) => entry.sourceGameIndex)).size,
        turnRange: {
          minimum: Math.min(...corpus.map((entry) => entry.state.turn)),
          maximum: Math.max(...corpus.map((entry) => entry.state.turn)),
        },
        phases: countStrings(corpus.map((entry) => entry.state.phase)),
        legalActionTypes: countStrings(
          corpus.flatMap((entry) => entry.legalActionTypes)
        ),
      },
      correctness: {
        comparedCorpusDecisions: corpus.length * options.repetitions,
        selectedActionMismatchCount,
        diagnosticsMismatchCount,
        shadow: {
          comparedGames: shadowGames.length,
          searchedDecisions: shadowGames.reduce(
            (total, game) => total + game.searchedDecisions,
            0
          ),
          actionMismatchCount: shadowActionMismatchCount,
          diagnosticsMismatchCount: shadowDiagnosticsMismatchCount,
          games: shadowGames,
        },
        mismatchSamples,
      },
      timing: {
        legacy: legacyTiming,
        paired: pairedTiming,
        speedup,
      },
      gate: {
        exactParity,
        activationPassed: executionModeRouting.passed,
        speedupThreshold: options.minimumSpeedup,
        speedupPassed,
        p95NonRegression,
        passed:
          executionModeRouting.passed &&
          exactParity &&
          speedupPassed &&
          p95NonRegression,
        recommendation: !executionModeRouting.passed
          ? 'retain-legacy-execution-mode-routing-failed'
          : options.mode === 'smoke'
            ? 'smoke-only-no-performance-decision'
            : exactParity && speedupPassed && p95NonRegression
              ? 'prepare-separate-default-enable-change'
              : exactParity
                ? 'retain-legacy-performance-gate-failed'
                : 'retain-legacy-parity-failed',
      },
    };
  } finally {
    legacy.close();
    paired.close();
  }
}

async function runDecision(
  state: GameState,
  randomSeed: string,
  policy: WorkerBackedActionPolicy
): Promise<DecisionResult> {
  const decisionPlayer = decisionPlayerIdForState(state);
  if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
    throw new Error('Outer shadow could not resolve decision player.');
  }
  const view = toDecisionPlayerView(state, decisionPlayer);
  const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
  if (actions.length < 2) {
    throw new Error('Outer shadow timed decision requires multiple actions.');
  }
  let diagnostics: SearchDecisionDiagnostics | undefined;
  const startedAt = performance.now();
  const action = await Promise.resolve(
    policy.selectAction({
      state,
      view,
      legalActions: actions,
      random: rngFromSeed(randomSeed),
      randomSeed,
      onSearchDiagnostics(value) {
        diagnostics = value;
      },
    })
  );
  const elapsedMs = performance.now() - startedAt;
  if (!action || !diagnostics) {
    throw new Error('Outer shadow policy did not return action diagnostics.');
  }
  return {
    action,
    actionKey: actionStableKey(action),
    diagnostics,
    elapsedMs,
  };
}

async function runShadowGame(
  gameIndex: number,
  legacy: WorkerBackedActionPolicy,
  paired: WorkerBackedActionPolicy,
  mismatchSamples: MismatchSample[],
  parallelWorkers: Set<number>,
  parallelBatchSizes: Set<number>,
  rootVisitBudgets: Set<number>
) {
  const gameSeed = `td-outer-shadow:game:${String(gameIndex)}`;
  let state = createSession(
    gameSeed,
    gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB'
  );
  let searchedDecisions = 0;
  let actionMismatchCount = 0;
  let diagnosticsMismatchCount = 0;
  const legacyActionKeys: string[] = [];
  const pairedActionKeys: string[] = [];
  for (
    let decisionIndex = 0;
    decisionIndex < 500 && !isTerminal(state);
    decisionIndex += 1
  ) {
    const decisionPlayer = decisionPlayerIdForState(state);
    if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
      throw new Error('Outer shadow game could not resolve decision player.');
    }
    const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
    let authoritativeAction = actions[0];
    let pairedAction = actions[0];
    if (actions.length > 1) {
      searchedDecisions += 1;
      const seed = `${gameSeed}:decision:${String(decisionIndex)}`;
      let legacyResult: DecisionResult;
      let pairedResult: DecisionResult;
      if ((gameIndex + decisionIndex) % 2 === 0) {
        legacyResult = await runDecision(state, seed, legacy);
        pairedResult = await runDecision(state, seed, paired);
      } else {
        pairedResult = await runDecision(state, seed, paired);
        legacyResult = await runDecision(state, seed, legacy);
      }
      authoritativeAction = legacyAuthoritativeValue(
        legacyResult.action,
        pairedResult.action
      );
      pairedAction = pairedResult.action;
      collectExecutionMetadata(
        legacyResult.diagnostics,
        parallelWorkers,
        parallelBatchSizes,
        rootVisitBudgets
      );
      collectExecutionMetadata(
        pairedResult.diagnostics,
        parallelWorkers,
        parallelBatchSizes,
        rootVisitBudgets
      );
      const actionsEqual = legacyResult.actionKey === pairedResult.actionKey;
      const diagnosticsEqual = exactJsonEqual(
        legacyResult.diagnostics,
        pairedResult.diagnostics
      );
      actionMismatchCount += Number(!actionsEqual);
      diagnosticsMismatchCount += Number(!diagnosticsEqual);
      maybeRecordMismatch(mismatchSamples, {
        scope: 'shadow-game',
        gameIndex,
        decisionIndex,
        legacyActionKey: legacyResult.actionKey,
        pairedActionKey: pairedResult.actionKey,
        diagnosticsEqual,
        legacyRootActions: legacyResult.diagnostics.rootActions,
        pairedRootActions: pairedResult.diagnostics.rootActions,
      });
    }
    if (!authoritativeAction || !pairedAction) {
      throw new Error('Outer shadow game encountered no legal action.');
    }
    legacyActionKeys.push(actionStableKey(authoritativeAction));
    pairedActionKeys.push(actionStableKey(pairedAction));
    // Non-negotiable authority rule: candidate output never advances the game.
    state = stepToDecision(state, authoritativeAction);
  }
  if (!isTerminal(state)) {
    throw new Error('Outer shadow game exceeded 500 decisions.');
  }
  return {
    gameIndex,
    seed: gameSeed,
    decisions: legacyActionKeys.length,
    searchedDecisions,
    actionMismatchCount,
    diagnosticsMismatchCount,
    transcriptsEqual: exactJsonEqual(legacyActionKeys, pairedActionKeys),
    authoritativeFinalStateSha256: await sha256Json(state),
  };
}

async function collectDecisionStates(count: number): Promise<CorpusState[]> {
  const gameCount = Math.min(12, Math.max(1, Math.ceil(count / 8)));
  const byGame: CorpusState[][] = [];
  for (let gameIndex = 0; gameIndex < gameCount; gameIndex += 1) {
    let state = createSession(
      `td-outer-shadow:source-game:${String(gameIndex)}`,
      gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB'
    );
    const candidates: CorpusState[] = [];
    for (
      let decisionIndex = 0;
      decisionIndex < 500 && !isTerminal(state);
      decisionIndex += 1
    ) {
      const decisionPlayer = decisionPlayerIdForState(state);
      if (decisionPlayer !== 'PlayerA' && decisionPlayer !== 'PlayerB') {
        throw new Error('Shadow corpus could not resolve decision player.');
      }
      const view = toDecisionPlayerView(state, decisionPlayer);
      const actions = legalActionsForDecisionPlayer(state, decisionPlayer);
      if (actions.length >= 2) {
        candidates.push({
          state: structuredClone(state) as GameState,
          sourceGameIndex: gameIndex,
          sourceDecisionIndex: decisionIndex,
          legalActionTypes: [
            ...new Set(actions.map((action) => action.type)),
          ].sort(),
        });
      }
      const selected = await Promise.resolve(
        heuristicPolicy.selectAction({
          state,
          view,
          legalActions: actions,
          random: policyRandomForState(state, 'td-outer-shadow-source'),
          randomSeed: policyRandomSeedForState(state, 'td-outer-shadow-source'),
        })
      );
      if (!selected) {
        throw new Error('Shadow corpus source policy returned no action.');
      }
      state = stepToDecision(state, selected);
    }
    if (candidates.length === 0) {
      throw new Error('Shadow corpus source game had no multi-action states.');
    }
    byGame.push(candidates);
  }

  const selected: CorpusState[] = [];
  const targetPerGame = Math.ceil(count / byGame.length);
  for (
    let quantile = 0;
    quantile < targetPerGame && selected.length < count;
    quantile += 1
  ) {
    for (const candidates of byGame) {
      if (selected.length >= count) {
        break;
      }
      const index = Math.min(
        candidates.length - 1,
        Math.floor(((quantile + 0.5) * candidates.length) / targetPerGame)
      );
      selected.push(candidates[index]);
    }
  }
  if (selected.length !== count) {
    throw new Error(
      `Selected ${String(selected.length)} shadow states; expected ${String(count)}.`
    );
  }
  return selected;
}

function collectExecutionMetadata(
  diagnostics: SearchDecisionDiagnostics,
  workers: Set<number>,
  batchSizes: Set<number>,
  budgets: Set<number>
): void {
  if (diagnostics.parallelWorkers !== undefined) {
    workers.add(diagnostics.parallelWorkers);
  }
  if (diagnostics.parallelBatchSize !== undefined) {
    batchSizes.add(diagnostics.parallelBatchSize);
  }
  budgets.add(diagnostics.rootVisitBudget);
}

function maybeRecordMismatch(
  samples: MismatchSample[],
  sample: MismatchSample
): void {
  if (
    samples.length < 10 &&
    (sample.legacyActionKey !== sample.pairedActionKey ||
      !sample.diagnosticsEqual)
  ) {
    samples.push(sample);
  }
}

function exactJsonEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function setContainsOnly<T>(values: ReadonlySet<T>, expected: T): boolean {
  return values.size === 1 && values.has(expected);
}

function countStrings(values: readonly string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right))
  );
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

async function corpusFingerprint(
  corpus: readonly CorpusState[]
): Promise<string> {
  return sha256Json(
    corpus.map((entry) => ({
      sourceGameIndex: entry.sourceGameIndex,
      sourceDecisionIndex: entry.sourceDecisionIndex,
      state: entry.state,
    }))
  );
}

async function sha256Json(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (item) =>
    item.toString(16).padStart(2, '0')
  ).join('');
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
