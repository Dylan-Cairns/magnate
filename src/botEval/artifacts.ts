import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import { parseHeadToHeadConfig } from './config';
import { collectGitMetadata } from './gitMetadata';
import {
  HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION,
  HEAD_TO_HEAD_ARTIFACT_TYPE,
  ROOT_ACTION_COUNT_BUCKETS,
  type HeadToHeadArtifact,
  type HeadToHeadRun,
} from './types';

export interface HeadToHeadArtifactOptions {
  cwd?: string;
  generatedAtUtc?: string;
  git?: HeadToHeadArtifact['git'];
  nodeVersion?: string;
}

export interface WrittenArtifactPaths {
  artifactPath: string;
  summaryPath: string;
}

export function createHeadToHeadArtifact(
  run: HeadToHeadRun,
  options: HeadToHeadArtifactOptions = {}
): HeadToHeadArtifact {
  return {
    schemaVersion: HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION,
    artifactType: HEAD_TO_HEAD_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    policyRandomSchemeVersion: POLICY_RANDOM_SCHEME_VERSION,
    runtime: {
      nodeVersion: options.nodeVersion ?? process.version,
    },
    git: options.git ?? collectGitMetadata(options.cwd),
    execution: structuredClone(run.execution),
    config: structuredClone(run.config),
    summary: structuredClone(run.summary),
    games: structuredClone(run.games),
  };
}

export async function writeHeadToHeadArtifacts(
  artifact: HeadToHeadArtifact,
  outputDirectory: string
): Promise<WrittenArtifactPaths> {
  await mkdir(outputDirectory, { recursive: true });
  const artifactPath = path.join(outputDirectory, 'matchup.json');
  const summaryPath = path.join(outputDirectory, 'summary.md');
  await writeAtomic(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeAtomic(summaryPath, renderHeadToHeadSummary(artifact));
  return {
    artifactPath,
    summaryPath,
  };
}

export async function loadHeadToHeadArtifact(
  artifactPath: string
): Promise<HeadToHeadArtifact> {
  const payload: unknown = JSON.parse(await readFile(artifactPath, 'utf8'));
  const source = requiredRecord(payload, 'head-to-head artifact');
  if (
    source.schemaVersion !== 1 &&
    source.schemaVersion !== 2 &&
    source.schemaVersion !== HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION
  ) {
    throw new Error(
      `Unsupported head-to-head artifact schemaVersion=${String(source.schemaVersion)}.`
    );
  }
  if (source.artifactType !== HEAD_TO_HEAD_ARTIFACT_TYPE) {
    throw new Error(
      `Unsupported head-to-head artifact type=${String(source.artifactType)}.`
    );
  }
  parseHeadToHeadConfig(source.config);
  if (!Array.isArray(source.games)) {
    throw new Error('head-to-head artifact.games must be an array.');
  }
  return payload as HeadToHeadArtifact;
}

export function defaultHeadToHeadOutputDirectory(
  runLabel: string,
  generatedAt = new Date()
): string {
  const stamp = generatedAt
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, 'Z');
  return path.join('artifacts', 'ts-bot-evals', `${stamp}-${slug(runLabel)}`);
}

export function renderHeadToHeadSummary(artifact: HeadToHeadArtifact): string {
  if (artifact.schemaVersion !== HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION) {
    throw new Error(
      `Head-to-head Markdown summaries require schemaVersion=${String(HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION)}.`
    );
  }
  const summary = artifact.summary;
  const ci = summary.candidateWinRateCi95;
  const lines = [
    `# TypeScript Bot Evaluation: ${artifact.config.runLabel}`,
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Execution: workers=${String(artifact.execution?.workers ?? 1)} requestedWorkers=${String(artifact.execution?.requestedWorkers ?? 1)} parallelUnit=${artifact.execution?.parallelUnit ?? 'paired-seed'} latencyMode=${artifact.execution?.latencyMode ?? 'isolated'}`,
    '',
    '| candidate | opponent | games | wins | losses | draws | win rate | ci95 | side gap | avg turns | games/min |',
    '|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|',
    `| ${summary.candidateId} | ${summary.opponentId} | ${summary.totalGames} | ${summary.candidateWins} | ${summary.opponentWins} | ${summary.draws} | ${format(summary.candidateWinRate)} | [${format(ci.low)}, ${format(ci.high)}] | ${format(summary.sideGap)} | ${format(summary.averageTurns)} | ${format(summary.gamesPerMinute)} |`,
    '',
    '| bot | actions | mean ms | p50 ms | p95 ms | max ms |',
    '|:---|---:|---:|---:|---:|---:|',
  ];
  for (const [botId, latency] of Object.entries(summary.latencyByBotId)) {
    lines.push(
      `| ${botId} | ${latency.actions} | ${format(latency.meanMs)} | ${format(latency.p50Ms)} | ${format(latency.p95Ms)} | ${format(latency.maxMs)} |`
    );
  }
  lines.push(
    '',
    '## Multi-Choice Decision Latency',
    '',
    '| bot | actions | mean ms | p50 ms | p95 ms | max ms |',
    '|:---|---:|---:|---:|---:|---:|'
  );
  for (const [botId, latency] of Object.entries(
    summary.multiChoiceLatencyByBotId
  )) {
    lines.push(
      `| ${botId} | ${latency.actions} | ${format(latency.meanMs)} | ${format(latency.p50Ms)} | ${format(latency.p95Ms)} | ${format(latency.maxMs)} |`
    );
  }
  lines.push(
    '',
    '## Rollout Search Work',
    '',
    '| bot | searched decisions | root visits | proxy cost | actual steps | max steps | utilization | mean steps | terminal rollouts | terminal rate | selected value | selected visits | selected terminal rate |',
    '|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
  );
  for (const [botId, work] of Object.entries(summary.searchWorkByBotId)) {
    lines.push(
      `| ${botId} | ${work.searchedDecisions} | ${work.rootVisits} | ${work.configProxyCost} | ${work.simulatedActionSteps} | ${work.maxSimulatedActionSteps} | ${format(work.stepUtilization)} | ${format(work.meanSimulatedActionSteps)} | ${work.terminalRollouts} | ${format(work.terminalRate)} | ${format(work.meanSelectedActionValue)} | ${format(work.meanSelectedActionVisits)} | ${format(work.meanSelectedActionTerminalRate)} |`
    );
  }
  for (const [botId, buckets] of Object.entries(
    summary.searchLatencyByRootActionCountByBotId
  )) {
    if (summary.searchWorkByBotId[botId]?.searchedDecisions === 0) {
      continue;
    }
    lines.push(
      '',
      `## Rollout Search Latency By Root Actions: ${botId}`,
      '',
      '| legal root actions | decisions | mean ms | p50 ms | p95 ms | max ms |',
      '|:---|---:|---:|---:|---:|---:|'
    );
    for (const bucket of ROOT_ACTION_COUNT_BUCKETS) {
      const latency = buckets[bucket];
      lines.push(
        `| ${bucket} | ${latency.actions} | ${format(latency.meanMs)} | ${format(latency.p50Ms)} | ${format(latency.p95Ms)} | ${format(latency.maxMs)} |`
      );
    }
  }
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function writeAtomic(
  targetPath: string,
  contents: string
): Promise<void> {
  const tempPath = `${targetPath}.tmp`;
  await writeFile(tempPath, contents, 'utf8');
  await rename(tempPath, targetPath);
}

function format(value: number): string {
  return value.toFixed(3);
}

function slug(value: string): string {
  return (
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'run'
  );
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
