import { mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import { actionStableKey } from '../src/engine/actionSurface';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../src/engine/decisionActor';
import { isTerminal } from '../src/engine/scoring';
import { createSession, stepToDecision } from '../src/engine/session';
import type { GameAction, GamePhase, GameState, PlayerId } from '../src/engine/types';
import { installLocalPublicFetch, tdModelIndexPath } from '../src/botEval/localPublicFetch';
import {
  ROOT_ACTION_COUNT_BUCKETS,
  type LatencySummary,
  type RootActionCountBucket,
} from '../src/botEval/types';
import {
  rootActionCountBucket,
  summarizeLatencies,
  summarizeSearchLatenciesByRootActionCount,
  summarizeSearchWork,
} from '../src/botEval/stats';
import { getBotProfile } from '../src/policies/catalog';
import {
  createPolicyFromBotSpec,
  parseBotSpec,
  type BotSpec,
} from '../src/policies/botSpec';
import {
  policyRandomForState,
  policyRandomSeedForState,
} from '../src/policies/policyRandom';
import type {
  ActionPolicy,
  SearchDecisionDiagnostics,
} from '../src/policies/types';

interface Options {
  states: number;
  seed: string;
  sourceProfile: string;
  configsPath?: string;
  preset: 'default' | 'smoke';
  outDir?: string;
  includeSingleAction: boolean;
  warmupStates: number;
  maxSourceGames: number;
  maxDecisionsPerSourceGame: number;
  tdPackId?: string;
}

interface DecisionSample {
  sampleId: string;
  state: GameState;
  activePlayerId: PlayerId;
  turn: number;
  phase: GamePhase;
  legalActionCount: number;
  gameIndex: number;
  decisionIndex: number;
}

interface DecisionSampleMetadata {
  sampleId: string;
  activePlayerId: PlayerId;
  turn: number;
  phase: GamePhase;
  legalActionCount: number;
  legalActionBucket: RootActionCountBucket | '1';
  gameIndex: number;
  decisionIndex: number;
}

interface CalibrationRow {
  configId: string;
  sampleId: string;
  turn: number;
  phase: GamePhase;
  activePlayerId: PlayerId;
  legalActionCount: number;
  legalActionBucket: RootActionCountBucket | '1';
  latencyMs: number;
  selectedActionKey: string;
  guidance?: SearchDecisionDiagnostics['guidance'];
  rootVisitBudget?: number;
  configProxyCost?: number;
  simulatedActionSteps?: number;
  expandedRootActions?: number;
  terminalRollouts?: number;
}

interface CalibrationSummary {
  configId: string;
  kind: BotSpec['kind'];
  all: LatencySummary & { p90Ms: number };
  multiChoice: LatencySummary & { p90Ms: number };
  byLegalActionBucket: Record<RootActionCountBucket, LatencySummary>;
  searchedDecisions: number;
  meanRootVisitBudget: number;
  meanSimulatedActionSteps: number;
  meanConfigProxyCost: number;
}

interface CalibrationArtifact {
  schemaVersion: 1;
  generatedAtUtc: string;
  options: Options;
  configs: BotSpec[];
  samples: DecisionSampleMetadata[];
  summaries: CalibrationSummary[];
  rows: CalibrationRow[];
}

const DEFAULT_OPTIONS: Options = {
  states: 100,
  seed: 'latency-calibration',
  sourceProfile: 'heuristic',
  preset: 'default',
  includeSingleAction: false,
  warmupStates: 3,
  maxSourceGames: 25,
  maxDecisionsPerSourceGame: 500,
};

async function main(): Promise<void> {
  installLocalPublicFetch();
  const options = parseOptions(process.argv.slice(2));
  const configs = options.configsPath
    ? await loadConfigs(options.configsPath)
    : defaultCalibrationConfigs(options.tdPackId, options.preset);
  const outDir =
    options.outDir ??
    path.join(
      'artifacts',
      'ts-bot-evals',
      `latency-calibration-${timestampForPath(new Date())}`
    );

  process.stderr.write(
    `[latency-cal] sampling states=${String(options.states)} source=${options.sourceProfile} seed=${options.seed}\n`
  );
  const samples = await collectDecisionSamples(options);
  process.stderr.write(
    `[latency-cal] sampled ${String(samples.length)} states; configs=${String(configs.length)}\n`
  );

  const rows: CalibrationRow[] = [];
  for (const spec of configs) {
    process.stderr.write(`[latency-cal] measuring ${spec.id}\n`);
    rows.push(...(await measureConfig(spec, samples, options)));
  }

  const summaries = summarizeRows(configs, rows);
  const artifact: CalibrationArtifact = {
    schemaVersion: 1,
    generatedAtUtc: new Date().toISOString(),
    options,
    configs,
    samples: samples.map(sampleMetadata),
    summaries,
    rows,
  };
  await writeArtifacts(outDir, artifact);
  process.stdout.write(`${renderSummaryMarkdown(summaries)}\n`);
  process.stderr.write(`[latency-cal] artifacts=${path.resolve(outDir)}\n`);
}

async function collectDecisionSamples(
  options: Options
): Promise<DecisionSample[]> {
  const sourceSpec = resolveSourceSpec(options.sourceProfile);
  const sourcePolicy = createPolicyFromBotSpec(sourceSpec);
  const samples: DecisionSample[] = [];
  for (
    let gameIndex = 0;
    gameIndex < options.maxSourceGames && samples.length < options.states;
    gameIndex += 1
  ) {
    let state = createSession(
      `${options.seed}:source-game:${String(gameIndex)}`,
      gameIndex % 2 === 0 ? 'PlayerA' : 'PlayerB'
    );
    for (
      let decisionIndex = 0;
      !isTerminal(state) &&
      decisionIndex < options.maxDecisionsPerSourceGame &&
      samples.length < options.states;
      decisionIndex += 1
    ) {
      const activePlayerId = decisionPlayerIdForState(state);
      if (activePlayerId !== 'PlayerA' && activePlayerId !== 'PlayerB') {
        throw new Error('Could not resolve active player while sampling.');
      }
      const legalActions = legalActionsForDecisionPlayer(state, activePlayerId);
      if (
        options.includeSingleAction ||
        legalActions.length > 1
      ) {
        samples.push({
          sampleId: `sample-${String(samples.length + 1).padStart(4, '0')}`,
          state: structuredClone(state) as GameState,
          activePlayerId,
          turn: state.turn,
          phase: state.phase,
          legalActionCount: legalActions.length,
          gameIndex,
          decisionIndex,
        });
      }

      const selected = await selectSourceAction({
        spec: sourceSpec,
        policy: sourcePolicy,
        state,
        activePlayerId,
        legalActions,
      });
      state = stepToDecision(state, selected);
    }
  }
  if (samples.length < options.states) {
    throw new Error(
      `Only collected ${String(samples.length)} states; increase --max-source-games or include single-action states.`
    );
  }
  return samples;
}

async function selectSourceAction({
  spec,
  policy,
  state,
  activePlayerId,
  legalActions,
}: {
  spec: BotSpec;
  policy: ActionPolicy;
  state: GameState;
  activePlayerId: PlayerId;
  legalActions: readonly GameAction[];
}): Promise<GameAction> {
  const selected = await Promise.resolve(
    policy.selectAction({
      state,
      view: toDecisionPlayerView(state, activePlayerId),
      legalActions,
      random: policyRandomForState(state, spec.id),
      randomSeed: policyRandomSeedForState(state, spec.id),
    })
  );
  if (!selected) {
    throw new Error(`Source policy ${spec.id} did not select an action.`);
  }
  const actionKey = actionStableKey(selected);
  const canonical = legalActions.find(
    (action) => actionStableKey(action) === actionKey
  );
  if (!canonical) {
    throw new Error(`Source policy ${spec.id} selected illegal action ${actionKey}.`);
  }
  return canonical;
}

async function measureConfig(
  spec: BotSpec,
  samples: readonly DecisionSample[],
  options: Options
): Promise<CalibrationRow[]> {
  const policy = createPolicyFromBotSpec(spec);
  const warmupSamples = samples.slice(0, Math.min(options.warmupStates, samples.length));
  for (const sample of warmupSamples) {
    await selectMeasuredAction(spec, policy, sample);
  }

  const rows: CalibrationRow[] = [];
  for (const sample of samples) {
    const measured = await selectMeasuredAction(spec, policy, sample);
    rows.push({
      configId: spec.id,
      sampleId: sample.sampleId,
      turn: sample.turn,
      phase: sample.phase,
      activePlayerId: sample.activePlayerId,
      legalActionCount: sample.legalActionCount,
      legalActionBucket: legalActionBucket(sample.legalActionCount),
      latencyMs: measured.latencyMs,
      selectedActionKey: measured.selectedActionKey,
      ...(measured.diagnostics?.guidance
        ? { guidance: measured.diagnostics.guidance }
        : {}),
      ...(measured.diagnostics
        ? {
            rootVisitBudget: measured.diagnostics.rootVisitBudget,
            configProxyCost: measured.diagnostics.configProxyCost,
            simulatedActionSteps: measured.diagnostics.simulatedActionSteps,
            expandedRootActions: measured.diagnostics.expandedRootActions,
            terminalRollouts: measured.diagnostics.terminalRollouts,
          }
        : {}),
    });
  }
  return rows;
}

async function selectMeasuredAction(
  spec: BotSpec,
  policy: ActionPolicy,
  sample: DecisionSample
): Promise<{
  selectedActionKey: string;
  latencyMs: number;
  diagnostics?: SearchDecisionDiagnostics;
}> {
  const state = structuredClone(sample.state) as GameState;
  const activePlayerId = decisionPlayerIdForState(state);
  if (activePlayerId !== sample.activePlayerId) {
    throw new Error(`Sample ${sample.sampleId} active player changed.`);
  }
  const legalActions = legalActionsForDecisionPlayer(state, activePlayerId);
  let diagnostics: SearchDecisionDiagnostics | undefined;
  const startedAt = performance.now();
  const selected = await Promise.resolve(
    policy.selectAction({
      state,
      view: toDecisionPlayerView(state, activePlayerId),
      legalActions,
      random: policyRandomForState(state, spec.id),
      randomSeed: policyRandomSeedForState(state, spec.id),
      onSearchDiagnostics(value) {
        diagnostics = structuredClone(value);
      },
    })
  );
  const latencyMs = performance.now() - startedAt;
  if (!selected) {
    throw new Error(`Policy ${spec.id} did not select an action for ${sample.sampleId}.`);
  }
  const selectedActionKey = actionStableKey(selected);
  if (!legalActions.some((action) => actionStableKey(action) === selectedActionKey)) {
    throw new Error(
      `Policy ${spec.id} selected illegal action ${selectedActionKey} for ${sample.sampleId}.`
    );
  }
  return {
    selectedActionKey,
    latencyMs,
    ...(diagnostics ? { diagnostics } : {}),
  };
}

function summarizeRows(
  configs: readonly BotSpec[],
  rows: readonly CalibrationRow[]
): CalibrationSummary[] {
  return configs.map((config) => {
    const configRows = rows.filter((row) => row.configId === config.id);
    const multiRows = configRows.filter((row) => row.legalActionCount > 1);
    const diagnostics = configRows
      .filter((row) => row.rootVisitBudget !== undefined)
      .map(rowToDiagnosticsLike);
    const searchWork = summarizeSearchWork(diagnostics);
    return {
      configId: config.id,
      kind: config.kind,
      all: extendedLatencySummary(configRows.map((row) => row.latencyMs)),
      multiChoice: extendedLatencySummary(multiRows.map((row) => row.latencyMs)),
      byLegalActionBucket: summarizeSearchLatenciesByRootActionCount(
        multiRows.map((row) => ({
          legalRootActions: row.legalActionCount,
          latencyMs: row.latencyMs,
        }))
      ),
      searchedDecisions: diagnostics.length,
      meanRootVisitBudget: meanDefined(configRows, 'rootVisitBudget'),
      meanSimulatedActionSteps: searchWork.meanSimulatedActionSteps,
      meanConfigProxyCost: meanDefined(configRows, 'configProxyCost'),
    };
  });
}

function rowToDiagnosticsLike(row: CalibrationRow): SearchDecisionDiagnostics {
  return {
    kind: 'search',
    legalRootActions: row.legalActionCount,
    expandedRootActions: row.expandedRootActions ?? 0,
    rootVisitBudget: row.rootVisitBudget ?? 0,
    configProxyCost: row.configProxyCost ?? 0,
    maxSimulatedActionSteps: row.simulatedActionSteps ?? 0,
    simulatedActionSteps: row.simulatedActionSteps ?? 0,
    terminalRollouts: row.terminalRollouts ?? 0,
    terminalRate: 0,
    selectedActionKey: row.selectedActionKey,
    selectedActionVisits: 0,
    selectedActionMeanValue: 0,
    selectedActionTerminalRollouts: 0,
    selectedActionTerminalRate: 0,
    rootActions: [],
  };
}

function extendedLatencySummary(
  values: readonly number[]
): LatencySummary & { p90Ms: number } {
  const base = summarizeLatencies(values);
  return {
    ...base,
    p90Ms: percentile(values, 0.9),
  };
}

function percentile(values: readonly number[], percentileValue: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(
    0,
    Math.min(sorted.length - 1, Math.ceil(percentileValue * sorted.length) - 1)
  );
  return sorted[index];
}

function meanDefined(
  rows: readonly CalibrationRow[],
  key: 'rootVisitBudget' | 'configProxyCost'
): number {
  const values = rows
    .map((row) => row[key])
    .filter((value): value is number => value !== undefined);
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sampleMetadata(sample: DecisionSample): DecisionSampleMetadata {
  return {
    sampleId: sample.sampleId,
    activePlayerId: sample.activePlayerId,
    turn: sample.turn,
    phase: sample.phase,
    legalActionCount: sample.legalActionCount,
    legalActionBucket: legalActionBucket(sample.legalActionCount),
    gameIndex: sample.gameIndex,
    decisionIndex: sample.decisionIndex,
  };
}

function legalActionBucket(count: number): RootActionCountBucket | '1' {
  return count <= 1 ? '1' : rootActionCountBucket(count);
}

function resolveSourceSpec(sourceProfile: string): BotSpec {
  if (sourceProfile === 'random') {
    return { id: 'source-random', kind: 'random' };
  }
  if (sourceProfile === 'heuristic') {
    return { id: 'source-heuristic', kind: 'heuristic' };
  }
  return structuredClone(getBotProfile(sourceProfile).spec);
}

function defaultCalibrationConfigs(
  tdPackId: string | undefined,
  preset: Options['preset']
): BotSpec[] {
  const modelIndexPath = tdModelIndexPath(tdPackId);
  const tdConfig = (
    id: string,
    worlds: number,
    depth: number,
    maxRootActions: number
  ): BotSpec => ({
    id,
    kind: 'td-root-search',
    modelIndexPath,
    config: {
      worlds,
      rollouts: 1,
      depth,
      maxRootActions,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    },
  });
  const searchConfig = (
    id: string,
    worlds: number,
    depth: number,
    maxRootActions: number
  ): BotSpec => ({
    id,
    kind: 'search',
    config: {
      worlds,
      rollouts: 1,
      depth,
      maxRootActions,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    },
  });
  if (preset === 'smoke') {
    return [
      searchConfig('heuristic-v2-smoke', 1, 1, 1),
      tdConfig('td-root-smoke', 1, 1, 1),
    ];
  }
  return [
    searchConfig('heuristic-v2-medium', 10, 40, 16),
    searchConfig('heuristic-v2-hard', 50, 270, 16),
    tdConfig('td-root-3w20d-8a', 3, 20, 8),
    tdConfig('td-root-5w20d-8a', 5, 20, 8),
    tdConfig('td-root-5w40d-8a', 5, 40, 8),
    tdConfig('td-root-10w20d-8a', 10, 20, 8),
    tdConfig('td-root-medium', 10, 40, 16),
  ];
}

async function loadConfigs(configsPath: string): Promise<BotSpec[]> {
  const payload = JSON.parse(await readFile(configsPath, 'utf8')) as unknown;
  const rawConfigs = Array.isArray(payload)
    ? payload
    : requiredRecord(payload, 'configs file').configs;
  if (!Array.isArray(rawConfigs)) {
    throw new Error('Config file must be an array or an object with configs array.');
  }
  return rawConfigs.map((entry, index) =>
    parseBotSpec(entry, `configs[${String(index)}]`)
  );
}

async function writeArtifacts(
  outDir: string,
  artifact: CalibrationArtifact
): Promise<void> {
  await mkdir(outDir, { recursive: true });
  await writeFile(
    path.join(outDir, 'latency-calibration.json'),
    `${JSON.stringify(artifact, null, 2)}\n`,
    'utf8'
  );
  await writeFile(path.join(outDir, 'latency-calibration.csv'), csvRows(artifact.rows), 'utf8');
  await writeFile(
    path.join(outDir, 'summary.md'),
    renderSummaryMarkdown(artifact.summaries),
    'utf8'
  );
}

function csvRows(rows: readonly CalibrationRow[]): string {
  const headers = [
    'configId',
    'sampleId',
    'turn',
    'phase',
    'activePlayerId',
    'legalActionCount',
    'legalActionBucket',
    'latencyMs',
    'selectedActionKey',
    'guidance',
    'rootVisitBudget',
    'configProxyCost',
    'simulatedActionSteps',
    'expandedRootActions',
    'terminalRollouts',
  ];
  return [
    headers.join(','),
    ...rows.map((row) =>
      headers.map((header) => csvCell(row[header as keyof CalibrationRow])).join(',')
    ),
  ].join('\n') + '\n';
}

function csvCell(value: unknown): string {
  if (value === undefined) {
    return '';
  }
  const raw = String(value);
  return /[",\n]/.test(raw) ? `"${raw.replaceAll('"', '""')}"` : raw;
}

function renderSummaryMarkdown(summaries: readonly CalibrationSummary[]): string {
  const sorted = [...summaries].sort(
    (left, right) => left.multiChoice.p95Ms - right.multiChoice.p95Ms
  );
  const lines = [
    '# Bot Latency Calibration',
    '',
    '| config | kind | multi actions | multi mean | multi p50 | multi p90 | multi p95 | multi max | mean root visits | mean sim steps |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ...sorted.map(
      (summary) =>
        `| ${summary.configId} | ${summary.kind} | ${String(summary.multiChoice.actions)} | ${formatMs(summary.multiChoice.meanMs)} | ${formatMs(summary.multiChoice.p50Ms)} | ${formatMs(summary.multiChoice.p90Ms)} | ${formatMs(summary.multiChoice.p95Ms)} | ${formatMs(summary.multiChoice.maxMs)} | ${summary.meanRootVisitBudget.toFixed(1)} | ${summary.meanSimulatedActionSteps.toFixed(1)} |`
    ),
    '',
    '## P95 By Legal Action Bucket',
    '',
    `| config | ${ROOT_ACTION_COUNT_BUCKETS.join(' | ')} |`,
    `| --- | ${ROOT_ACTION_COUNT_BUCKETS.map(() => '---:').join(' | ')} |`,
    ...sorted.map(
      (summary) =>
        `| ${summary.configId} | ${ROOT_ACTION_COUNT_BUCKETS.map((bucket) => formatMs(summary.byLegalActionBucket[bucket].p95Ms)).join(' | ')} |`
    ),
  ];
  return `${lines.join('\n')}\n`;
}

function formatMs(value: number): string {
  return `${value.toFixed(1)}ms`;
}

function requiredRecord(
  value: unknown,
  label: string
): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function parseOptions(args: readonly string[]): Options {
  const flags = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`Invalid argument near ${String(key)}.`);
    }
    flags.set(key, value);
  }
  return {
    ...DEFAULT_OPTIONS,
    states: optionalInt(flags, '--states', DEFAULT_OPTIONS.states),
    seed: flags.get('--seed') ?? DEFAULT_OPTIONS.seed,
    sourceProfile: flags.get('--source-profile') ?? DEFAULT_OPTIONS.sourceProfile,
    configsPath: flags.get('--configs'),
    preset: optionalPreset(flags, '--preset', DEFAULT_OPTIONS.preset),
    outDir: flags.get('--out-dir'),
    includeSingleAction: optionalBoolean(
      flags,
      '--include-single-action',
      DEFAULT_OPTIONS.includeSingleAction
    ),
    warmupStates: optionalInt(flags, '--warmup-states', DEFAULT_OPTIONS.warmupStates),
    maxSourceGames: optionalInt(
      flags,
      '--max-source-games',
      DEFAULT_OPTIONS.maxSourceGames
    ),
    maxDecisionsPerSourceGame: optionalInt(
      flags,
      '--max-decisions-per-source-game',
      DEFAULT_OPTIONS.maxDecisionsPerSourceGame
    ),
    tdPackId: flags.get('--td-pack-id'),
  };
}

function optionalPreset(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: Options['preset']
): Options['preset'] {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === 'default' || raw === 'smoke') {
    return raw;
  }
  throw new Error(`${name} must be default or smoke.`);
}

function optionalInt(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: number
): number {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function optionalBoolean(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: boolean
): boolean {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === 'true') {
    return true;
  }
  if (raw === 'false') {
    return false;
  }
  throw new Error(`${name} must be true or false.`);
}

function timestampForPath(date: Date): string {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
}

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`[latency-cal] ${message}\n`);
  process.exitCode = 1;
});
