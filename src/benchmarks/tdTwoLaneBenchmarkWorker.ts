import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import type { GameAction, PlayerView } from '../engine/types';
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
import { TdRootOpponentNetwork } from '../policies/tdRootModelPack';
import { encodeObservation } from '../policies/trainingEncoding';

interface BenchmarkOptions {
  mode: 'smoke' | 'full';
  states: number;
  rounds: number;
  warmupRounds: number;
  minimumSpeedup: number;
}

interface InferenceRequest {
  view: PlayerView;
  actions: readonly GameAction[];
}

interface TimingSummary {
  totalMs: number;
  meanRoundMs: number;
  p50RoundMs: number;
  p95RoundMs: number;
  requestsPerSecond: number;
  checksum: number;
}

interface BenchmarkResult {
  schemaVersion: 1;
  generatedAtUtc: string;
  benchmark: 'td-opponent-two-lane-browser-worker';
  mode: BenchmarkOptions['mode'];
  environment: {
    userAgent: string;
    hardwareConcurrency: number;
    crossOriginIsolated: boolean;
  };
  options: BenchmarkOptions;
  corpus: {
    requests: number;
    meanActions: number;
    minActions: number;
    maxActions: number;
  };
  model: {
    indexPath: string;
    packId: string;
    weightsUrl: string;
    weightsSha256: string;
  };
  correctness: {
    comparedLogits: number;
    exactMismatchCount: number;
    argmaxMismatchCount: number;
    maxAbsoluteDifference: number;
  };
  timing: {
    scalar: TimingSummary;
    paired: TimingSummary;
    speedup: number;
  };
  gate: {
    exactParity: boolean;
    speedupThreshold: number;
    speedupPassed: boolean;
    recommendation:
      | 'smoke-only-no-performance-decision'
      | 'continue-to-lockstep-spike'
      | 'stop-browser-batching';
  };
}

self.onmessage = (event: MessageEvent<BenchmarkOptions>) => {
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

async function runBenchmark(
  options: BenchmarkOptions
): Promise<BenchmarkResult> {
  const [loadedModel, provenance, requests] = await Promise.all([
    preloadTdRootBrowserModel(),
    loadModelProvenance(),
    collectInferenceRequests(options.states),
  ]);
  if (!(loadedModel.opponentScorer instanceof TdRootOpponentNetwork)) {
    throw new Error(
      'Loaded TD opponent scorer does not support paired inference.'
    );
  }
  const model = loadedModel.opponentScorer;
  const correctness = compareScalarAndPaired(model, requests);

  for (let index = 0; index < options.warmupRounds; index += 1) {
    runScalarRound(model, requests);
    runPairedRound(model, requests);
  }

  const scalarRoundMs: number[] = [];
  const pairedRoundMs: number[] = [];
  let scalarChecksum = 0;
  let pairedChecksum = 0;
  for (let round = 0; round < options.rounds; round += 1) {
    if (round % 2 === 0) {
      const scalar = runScalarRound(model, requests);
      const paired = runPairedRound(model, requests);
      scalarRoundMs.push(scalar.elapsedMs);
      pairedRoundMs.push(paired.elapsedMs);
      scalarChecksum += scalar.checksum;
      pairedChecksum += paired.checksum;
    } else {
      const paired = runPairedRound(model, requests);
      const scalar = runScalarRound(model, requests);
      pairedRoundMs.push(paired.elapsedMs);
      scalarRoundMs.push(scalar.elapsedMs);
      pairedChecksum += paired.checksum;
      scalarChecksum += scalar.checksum;
    }
  }

  const scalar = summarizeTiming(
    scalarRoundMs,
    requests.length,
    scalarChecksum
  );
  const paired = summarizeTiming(
    pairedRoundMs,
    requests.length,
    pairedChecksum
  );
  const speedup = scalar.totalMs / paired.totalMs;
  const exactParity = correctness.exactMismatchCount === 0;
  const speedupPassed = speedup >= options.minimumSpeedup;
  const actionCounts = requests.map((request) => request.actions.length);

  return {
    schemaVersion: 1,
    generatedAtUtc: new Date().toISOString(),
    benchmark: 'td-opponent-two-lane-browser-worker',
    mode: options.mode,
    environment: {
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      crossOriginIsolated: self.crossOriginIsolated,
    },
    options,
    corpus: {
      requests: requests.length,
      meanActions:
        actionCounts.reduce((total, count) => total + count, 0) /
        actionCounts.length,
      minActions: Math.min(...actionCounts),
      maxActions: Math.max(...actionCounts),
    },
    model: provenance,
    correctness,
    timing: { scalar, paired, speedup },
    gate: {
      exactParity,
      speedupThreshold: options.minimumSpeedup,
      speedupPassed,
      recommendation:
        options.mode === 'smoke'
          ? 'smoke-only-no-performance-decision'
          : exactParity && speedupPassed
            ? 'continue-to-lockstep-spike'
            : 'stop-browser-batching',
    },
  };
}

async function collectInferenceRequests(
  requestCount: number
): Promise<InferenceRequest[]> {
  const requests: InferenceRequest[] = [];
  for (
    let gameIndex = 0;
    gameIndex < 100 && requests.length < requestCount;
    gameIndex += 1
  ) {
    let state = createSession(
      `td-two-lane-benchmark:game:${String(gameIndex)}`,
      gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB'
    );
    for (
      let decisionIndex = 0;
      decisionIndex < 500 &&
      !isTerminal(state) &&
      requests.length < requestCount;
      decisionIndex += 1
    ) {
      const activePlayer = decisionPlayerIdForState(state);
      if (activePlayer !== 'PlayerA' && activePlayer !== 'PlayerB') {
        throw new Error(
          'Could not resolve decision player while building benchmark corpus.'
        );
      }
      const view = toDecisionPlayerView(state, activePlayer);
      const actions = legalActionsForDecisionPlayer(state, activePlayer);
      if (actions.length === 0) {
        throw new Error(
          'Benchmark corpus encountered a decision with no legal actions.'
        );
      }
      requests.push({ view, actions });

      const selected = await Promise.resolve(
        heuristicPolicy.selectAction({
          state,
          view,
          legalActions: actions,
          random: policyRandomForState(state, 'td-two-lane-benchmark-source'),
          randomSeed: policyRandomSeedForState(
            state,
            'td-two-lane-benchmark-source'
          ),
        })
      );
      if (!selected) {
        throw new Error('Benchmark source policy did not select an action.');
      }
      state = stepToDecision(state, selected);
    }
  }
  if (requests.length !== requestCount) {
    throw new Error(
      `Collected ${String(requests.length)} inference requests; expected ${String(requestCount)}.`
    );
  }
  return requests;
}

function compareScalarAndPaired(
  model: TdRootOpponentNetwork,
  requests: readonly InferenceRequest[]
): BenchmarkResult['correctness'] {
  let comparedLogits = 0;
  let exactMismatchCount = 0;
  let argmaxMismatchCount = 0;
  let maxAbsoluteDifference = 0;
  for (let index = 0; index < requests.length; index += 2) {
    const requestA = requests[index];
    const requestB = requests[index + 1];
    const observationA = encodeObservation(requestA.view);
    const observationB = encodeObservation(requestB.view);
    const scalarA = model.logitsForActions(observationA, requestA.actions);
    const scalarB = model.logitsForActions(observationB, requestB.actions);
    const [pairedA, pairedB] = model.logitsForActionPair(
      observationA,
      requestA.actions,
      observationB,
      requestB.actions
    );
    const pairs = [
      [scalarA, pairedA],
      [scalarB, pairedB],
    ] as const;
    for (const [scalar, paired] of pairs) {
      if (scalar.length !== paired.length) {
        throw new Error('Paired logits did not preserve candidate count.');
      }
      argmaxMismatchCount += Number(argmax(scalar) !== argmax(paired));
      for (let logitIndex = 0; logitIndex < scalar.length; logitIndex += 1) {
        comparedLogits += 1;
        exactMismatchCount += Number(
          !Object.is(scalar[logitIndex], paired[logitIndex])
        );
        maxAbsoluteDifference = Math.max(
          maxAbsoluteDifference,
          Math.abs(scalar[logitIndex] - paired[logitIndex])
        );
      }
    }
  }
  return {
    comparedLogits,
    exactMismatchCount,
    argmaxMismatchCount,
    maxAbsoluteDifference,
  };
}

function runScalarRound(
  model: TdRootOpponentNetwork,
  requests: readonly InferenceRequest[]
): { elapsedMs: number; checksum: number } {
  const startedAt = performance.now();
  let checksum = 0;
  for (const request of requests) {
    const logits = model.logitsForActions(
      encodeObservation(request.view),
      request.actions
    );
    checksum += checksumLogits(logits);
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

function runPairedRound(
  model: TdRootOpponentNetwork,
  requests: readonly InferenceRequest[]
): { elapsedMs: number; checksum: number } {
  const startedAt = performance.now();
  let checksum = 0;
  for (let index = 0; index < requests.length; index += 2) {
    const requestA = requests[index];
    const requestB = requests[index + 1];
    const [logitsA, logitsB] = model.logitsForActionPair(
      encodeObservation(requestA.view),
      requestA.actions,
      encodeObservation(requestB.view),
      requestB.actions
    );
    checksum += checksumLogits(logitsA) + checksumLogits(logitsB);
  }
  return { elapsedMs: performance.now() - startedAt, checksum };
}

function checksumLogits(logits: Float32Array): number {
  let checksum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    checksum += logits[index] * (index + 1);
  }
  return checksum;
}

function argmax(values: Float32Array): number {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) {
      bestIndex = index;
    }
  }
  return bestIndex;
}

function summarizeTiming(
  roundMs: readonly number[],
  requestsPerRound: number,
  checksum: number
): TimingSummary {
  const sorted = [...roundMs].sort((left, right) => left - right);
  const totalMs = roundMs.reduce((total, value) => total + value, 0);
  return {
    totalMs,
    meanRoundMs: totalMs / roundMs.length,
    p50RoundMs: percentile(sorted, 0.5),
    p95RoundMs: percentile(sorted, 0.95),
    requestsPerSecond: (requestsPerRound * roundMs.length * 1000) / totalMs,
    checksum,
  };
}

function percentile(sorted: readonly number[], fraction: number): number {
  const index = Math.max(0, Math.ceil(sorted.length * fraction) - 1);
  return sorted[index];
}

async function loadModelProvenance(): Promise<BenchmarkResult['model']> {
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
