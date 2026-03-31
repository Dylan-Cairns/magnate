import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';

import { parseRolloutSearchSweepConfig } from './config';
import { collectGitMetadata } from './gitMetadata';
import {
  createHeadToHeadArtifact,
  type HeadToHeadArtifactOptions,
  writeHeadToHeadArtifacts,
} from './artifacts';
import {
  ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION,
  ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE,
  type RolloutSearchSweepArtifact,
  type RolloutSearchSweepArtifactRow,
  type RolloutSearchSweepArtifactStatus,
  type RolloutSearchSweepRun,
} from './types';
import {
  appendRootActionLatencyTable,
  defaultBotEvalOutputDirectory,
  format,
  requiredRecord,
  slug,
  writeAtomic,
} from './artifactUtils';

export interface WrittenRolloutSearchSweepArtifactPaths {
  artifact: RolloutSearchSweepArtifact;
  artifactPath: string;
  csvPath: string;
  summaryPath: string;
}

export interface RolloutSearchSweepArtifactOptions extends HeadToHeadArtifactOptions {
  status?: RolloutSearchSweepArtifactStatus;
  matchupIndicesToWrite?: readonly number[];
}

export function createRolloutSearchSweepArtifact(
  run: RolloutSearchSweepRun,
  options: RolloutSearchSweepArtifactOptions = {}
): RolloutSearchSweepArtifact {
  if (run.matchups.length > run.config.candidates.length) {
    throw new Error(
      'Rollout-search sweep has more matchups than configured candidates.'
    );
  }
  const status =
    options.status ??
    (run.matchups.length === run.config.candidates.length
      ? 'completed'
      : 'running');
  if (
    status === 'completed' &&
    run.matchups.length !== run.config.candidates.length
  ) {
    throw new Error(
      'Completed rollout-search sweeps require every candidate matchup.'
    );
  }
  return {
    schemaVersion: ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION,
    artifactType: ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    runtime: {
      nodeVersion: options.nodeVersion ?? process.version,
    },
    git: options.git ?? collectGitMetadata(options.cwd),
    execution: structuredClone(run.execution),
    status,
    completedCandidates: run.matchups.length,
    totalCandidates: run.config.candidates.length,
    config: structuredClone(run.config),
    rows: run.matchups.map((matchup, index) => ({
      candidate: structuredClone(run.config.candidates[index]),
      execution: structuredClone(matchup.execution),
      summary: structuredClone(matchup.summary),
      matchupArtifactPath: `${matchupRelativeDirectory(
        index,
        run.config.candidates[index].id
      )}/matchup.json`,
    })),
  };
}

export async function writeRolloutSearchSweepArtifacts(
  run: RolloutSearchSweepRun,
  outputDirectory: string,
  options: RolloutSearchSweepArtifactOptions = {}
): Promise<WrittenRolloutSearchSweepArtifactPaths> {
  const artifact = createRolloutSearchSweepArtifact(run, options);
  await mkdir(outputDirectory, { recursive: true });
  const matchupIndices =
    options.matchupIndicesToWrite ?? run.matchups.map((_, index) => index);
  for (const index of matchupIndices) {
    if (!Number.isInteger(index) || index < 0 || index >= run.matchups.length) {
      throw new Error(
        `Rollout-search sweep matchup index is out of range: ${String(index)}.`
      );
    }
    const childDirectory = path.join(
      outputDirectory,
      matchupRelativeDirectory(index, run.config.candidates[index].id)
    );
    await writeHeadToHeadArtifacts(
      createHeadToHeadArtifact(run.matchups[index], {
        ...options,
        generatedAtUtc: artifact.generatedAtUtc,
        git: artifact.git,
        nodeVersion: artifact.runtime.nodeVersion,
      }),
      childDirectory
    );
  }

  const artifactPath = path.join(outputDirectory, 'sweep.json');
  const csvPath = path.join(outputDirectory, 'sweep.csv');
  const summaryPath = path.join(outputDirectory, 'summary.md');
  await writeAtomic(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeAtomic(csvPath, renderRolloutSearchSweepCsv(artifact));
  await writeAtomic(summaryPath, renderRolloutSearchSweepSummary(artifact));
  return {
    artifact,
    artifactPath,
    csvPath,
    summaryPath,
  };
}

export async function loadRolloutSearchSweepArtifact(
  artifactPath: string
): Promise<RolloutSearchSweepArtifact> {
  const payload: unknown = JSON.parse(await readFile(artifactPath, 'utf8'));
  const source = requiredRecord(payload, 'rollout-search sweep artifact');
  if (
    source.schemaVersion !== 1 &&
    source.schemaVersion !== ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION
  ) {
    throw new Error(
      `Unsupported rollout-search sweep artifact schemaVersion=${String(source.schemaVersion)}.`
    );
  }
  if (source.artifactType !== ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE) {
    throw new Error(
      `Unsupported rollout-search sweep artifact type=${String(source.artifactType)}.`
    );
  }
  const config = parseRolloutSearchSweepConfig(source.config);
  if (!Array.isArray(source.rows)) {
    throw new Error('rollout-search sweep artifact.rows must be an array.');
  }
  if (source.status !== 'running' && source.status !== 'completed') {
    throw new Error('rollout-search sweep artifact.status is invalid.');
  }
  if (source.completedCandidates !== source.rows.length) {
    throw new Error(
      'rollout-search sweep artifact.completedCandidates must match rows.'
    );
  }
  if (source.totalCandidates !== config.candidates.length) {
    throw new Error(
      'rollout-search sweep artifact.totalCandidates must match config.'
    );
  }
  if (
    source.status === 'completed' &&
    source.completedCandidates !== source.totalCandidates
  ) {
    throw new Error(
      'completed rollout-search sweep artifacts require every candidate.'
    );
  }
  return payload as RolloutSearchSweepArtifact;
}

export function defaultRolloutSearchSweepOutputDirectory(
  runLabel: string,
  generatedAt = new Date()
): string {
  return defaultBotEvalOutputDirectory(runLabel, generatedAt);
}

export function renderRolloutSearchSweepCsv(
  artifact: RolloutSearchSweepArtifact
): string {
  const headers = [
    'candidateId',
    'opponentId',
    'worlds',
    'rollouts',
    'depth',
    'maxRootActions',
    'rolloutEpsilon',
    'workers',
    'latencyMode',
    'configProxyCost',
    'totalGames',
    'candidateWins',
    'opponentWins',
    'draws',
    'candidateWinRate',
    'ci95Low',
    'ci95High',
    'sideGap',
    'gamesPerMinute',
    'allDecisionP95Ms',
    'multiChoiceP50Ms',
    'multiChoiceP95Ms',
    'multiChoiceMaxMs',
    'searchedDecisions',
    'meanActualSimulatedSteps',
    'actualSimulatedSteps',
    'maxSimulatedActionSteps',
    'stepUtilization',
    'terminalRolloutRate',
    'meanSelectedActionValue',
    'meanSelectedActionVisits',
    'meanSelectedActionTerminalRate',
  ];
  const rows = artifact.rows.map((row) => {
    const summary = row.summary;
    const candidateId = row.candidate.id;
    const allLatency = requiredBotEntry(
      summary.latencyByBotId,
      candidateId,
      'latency'
    );
    const multiChoiceLatency = requiredBotEntry(
      summary.multiChoiceLatencyByBotId,
      candidateId,
      'multi-choice latency'
    );
    const searchWork = requiredBotEntry(
      summary.searchWorkByBotId,
      candidateId,
      'search work'
    );
    return [
      candidateId,
      summary.opponentId,
      row.candidate.config.worlds,
      row.candidate.config.rollouts,
      row.candidate.config.depth,
      row.candidate.config.maxRootActions,
      row.candidate.config.rolloutEpsilon,
      row.execution?.workers ?? 1,
      row.execution?.latencyMode ?? 'isolated',
      configProxyCost(row),
      summary.totalGames,
      summary.candidateWins,
      summary.opponentWins,
      summary.draws,
      summary.candidateWinRate,
      summary.candidateWinRateCi95.low,
      summary.candidateWinRateCi95.high,
      summary.sideGap,
      summary.gamesPerMinute,
      allLatency.p95Ms,
      multiChoiceLatency.p50Ms,
      multiChoiceLatency.p95Ms,
      multiChoiceLatency.maxMs,
      searchWork.searchedDecisions,
      searchWork.meanSimulatedActionSteps,
      searchWork.simulatedActionSteps,
      searchWork.maxSimulatedActionSteps,
      searchWork.stepUtilization,
      searchWork.terminalRate,
      searchWork.meanSelectedActionValue,
      searchWork.meanSelectedActionVisits,
      searchWork.meanSelectedActionTerminalRate,
    ];
  });
  return `${[headers, ...rows]
    .map((row) => row.map(csvCell).join(','))
    .join('\n')}\n`;
}

export function renderRolloutSearchSweepSummary(
  artifact: RolloutSearchSweepArtifact
): string {
  const lines = [
    `# TypeScript Rollout Search Sweep: ${artifact.config.runLabel}`,
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Execution: workers=${String(artifact.execution?.workers ?? 1)} requestedWorkers=${String(artifact.execution?.requestedWorkers ?? 1)} parallelUnit=${artifact.execution?.parallelUnit ?? 'paired-seed'} latencyMode=${artifact.execution?.latencyMode ?? 'isolated'}`,
    '',
    `Status: ${artifact.status} (${String(artifact.completedCandidates)}/${String(artifact.totalCandidates)} candidates)`,
    '',
    '| candidate | opponent | config | workers | latency mode | proxy cost | games | win rate | ci95 | side gap | games/min | multi p50 ms | multi p95 ms | multi max ms | actual steps | utilization | terminal rate | selected value | selected visits |',
    '|:---|:---|:---|---:|:---|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
  ];
  for (const row of artifact.rows) {
    const summary = row.summary;
    const candidateId = row.candidate.id;
    const latency = requiredBotEntry(
      summary.multiChoiceLatencyByBotId,
      candidateId,
      'multi-choice latency'
    );
    const work = requiredBotEntry(
      summary.searchWorkByBotId,
      candidateId,
      'search work'
    );
    lines.push(
      `| ${candidateId} | ${summary.opponentId} | ${searchConfigLabel(row)} | ${String(row.execution?.workers ?? 1)} | ${row.execution?.latencyMode ?? 'isolated'} | ${configProxyCost(row)} | ${summary.totalGames} | ${format(summary.candidateWinRate)} | [${format(summary.candidateWinRateCi95.low)}, ${format(summary.candidateWinRateCi95.high)}] | ${format(summary.sideGap)} | ${format(summary.gamesPerMinute)} | ${format(latency.p50Ms)} | ${format(latency.p95Ms)} | ${format(latency.maxMs)} | ${work.simulatedActionSteps} | ${format(work.stepUtilization)} | ${format(work.terminalRate)} | ${format(work.meanSelectedActionValue)} | ${format(work.meanSelectedActionVisits)} |`
    );
  }
  for (const row of artifact.rows) {
    const candidateId = row.candidate.id;
    const buckets = requiredBotEntry(
      row.summary.searchLatencyByRootActionCountByBotId,
      candidateId,
      'search latency buckets'
    );
    appendRootActionLatencyTable(
      lines,
      `## Root-Action Latency: ${candidateId}`,
      buckets
    );
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

function configProxyCost(row: RolloutSearchSweepArtifactRow): number {
  const config = row.candidate.config;
  return config.worlds * config.rollouts * config.maxRootActions * config.depth;
}

function searchConfigLabel(row: RolloutSearchSweepArtifactRow): string {
  const config = row.candidate.config;
  return `${config.worlds}w/${config.rollouts}r/${config.depth}d/${config.maxRootActions}a/e${config.rolloutEpsilon}`;
}

function matchupRelativeDirectory(index: number, candidateId: string): string {
  return `matchups/${String(index + 1).padStart(3, '0')}-${slug(candidateId)}`;
}

function requiredBotEntry<T>(
  entries: Readonly<Record<string, T>>,
  botId: string,
  label: string
): T {
  const entry = entries[botId];
  if (!entry) {
    throw new Error(`Rollout-search sweep is missing ${label} for ${botId}.`);
  }
  return entry;
}

function csvCell(value: unknown): string {
  const text = String(value);
  if (!/[",\r\n]/.test(text)) {
    return text;
  }
  return `"${text.replace(/"/g, '""')}"`;
}
