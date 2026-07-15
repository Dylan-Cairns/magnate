import { mkdir } from 'node:fs/promises';
import path from 'node:path';

import { collectGitMetadata } from './gitMetadata';
import { defaultBotEvalOutputDirectory, writeAtomic } from './artifactUtils';
import type { TdSymmetryAuditRun, TdSymmetryStratum } from './tdSymmetry';
import type { GitMetadata } from './types';

export const TD_SYMMETRY_ARTIFACT_SCHEMA_VERSION = 1 as const;
export const TD_SYMMETRY_ARTIFACT_TYPE = 'ts-td-symmetry-audit' as const;

export interface TdSymmetryArtifact {
  readonly schemaVersion: typeof TD_SYMMETRY_ARTIFACT_SCHEMA_VERSION;
  readonly artifactType: typeof TD_SYMMETRY_ARTIFACT_TYPE;
  readonly generatedAtUtc: string;
  readonly runtime: { readonly nodeVersion: string };
  readonly git: GitMetadata;
  readonly run: TdSymmetryAuditRun;
}

export interface TdSymmetryArtifactOptions {
  readonly cwd?: string;
  readonly generatedAtUtc?: string;
  readonly git?: GitMetadata;
  readonly nodeVersion?: string;
}

export function createTdSymmetryArtifact(
  run: TdSymmetryAuditRun,
  options: TdSymmetryArtifactOptions = {}
): TdSymmetryArtifact {
  return {
    schemaVersion: TD_SYMMETRY_ARTIFACT_SCHEMA_VERSION,
    artifactType: TD_SYMMETRY_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    runtime: { nodeVersion: options.nodeVersion ?? process.version },
    git: options.git ?? collectGitMetadata(options.cwd),
    run: structuredClone(run),
  };
}

export async function writeTdSymmetryArtifacts(
  artifact: TdSymmetryArtifact,
  outputDirectory: string
): Promise<{ readonly artifactPath: string; readonly summaryPath: string }> {
  await mkdir(outputDirectory, { recursive: true });
  const artifactPath = path.join(outputDirectory, 'symmetry.json');
  const summaryPath = path.join(outputDirectory, 'summary.md');
  await writeAtomic(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeAtomic(summaryPath, renderTdSymmetrySummary(artifact));
  return { artifactPath, summaryPath };
}

export function defaultTdSymmetryOutputDirectory(
  generatedAt = new Date()
): string {
  return defaultBotEvalOutputDirectory('td-symmetry-audit-v1', generatedAt);
}

export function renderTdSymmetrySummary(artifact: TdSymmetryArtifact): string {
  const run = artifact.run;
  const aggregate = run.aggregate;
  const lines = [
    '# TD District-Symmetry Audit',
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Model pack: \`${run.model.packId}\``,
    '',
    `Replay rows scanned: ${String(run.replay.rowsScanned)} across ${String(run.replay.sourceFiles.length)} files; deterministic sample: ${String(run.replay.sampledRows)} (requested ${String(run.replay.requestedSampleSize)}; seed \`${run.replay.samplingSeed}\`).`,
    '',
    `Each sampled position was evaluated under all ${String(run.permutations.length)} permutations of D1, D2, D4, and D5 while D3 remained fixed. Board blocks and district-referenced action features moved together.`,
    '',
    'This measures whether the fixed model changes its output when equivalent Pawn districts are relabeled. It is not a playing-strength evaluation.',
    '',
    '## Aggregate',
    '',
    '| measure | result |',
    '|:---|---:|',
    `| sampled decisions with at least one preferred-action change | ${String(aggregate.samplesWithAnyTopActionFlip)}/${String(aggregate.samples)} (${percent(safeDivide(aggregate.samplesWithAnyTopActionFlip, aggregate.samples))}) |`,
    `| preferred-action agreement across non-identity comparisons | ${percent(aggregate.topActionMatchRate)} |`,
    `| mean / maximum Jensen-Shannon divergence | ${format(aggregate.jensenShannonDivergence.mean)} / ${format(aggregate.jensenShannonDivergence.max)} |`,
    `| mean / maximum probability change | ${format(aggregate.maxProbabilityDelta.mean)} / ${format(aggregate.maxProbabilityDelta.max)} |`,
    `| mean / maximum centered-logit change | ${format(aggregate.maxCenteredLogitDelta.mean)} / ${format(aggregate.maxCenteredLogitDelta.max)} |`,
    `| mean / maximum value-estimate change | ${format(aggregate.valueAbsoluteDelta.mean)} / ${format(aggregate.valueAbsoluteDelta.max)} |`,
    '',
    '## Physical Pawn-Slot Effects',
    '',
    'All 24 permutations balance the same district/action content across the four physical Pawn slots. Centered logits remove each legal-action set’s common offset.',
    '',
    '| slot | action observations | mean probability | mean centered logit | replay files represented | files where uniquely highest |',
    '|:---|---:|---:|---:|---:|---:|',
    ...run.pawnSlotEffects.map(
      (slot) =>
        `| ${slot.districtId} | ${String(slot.observations)} | ${format(slot.meanProbability)} | ${format(slot.meanCenteredLogit)} | ${String(slot.sourceFilesWithObservations)} | ${String(slot.sourceFilesUniquelyHighestCenteredLogit)} |`
    ),
    '',
    '## Baseline Preference Margin',
    '',
    'These bins separate changes in near-ties from changes to choices the baseline model preferred more clearly.',
    '',
    ...renderStrataTable(run.byBaselineTopProbabilityMargin),
    '',
    '## District-Action Availability',
    '',
    ...renderStrataTable(run.byDistrictActionAvailability),
    '',
    '## Baseline Preferred Action Type',
    '',
    ...renderStrataTable(run.byBaselineTopActionType),
    '',
    '## Largest Changes',
    '',
    '| source | line | permutation | baseline → permuted type | baseline margin | JS divergence | max probability change | centered-logit change | value change |',
    '|:---|---:|:---|:---|---:|---:|---:|---:|---:|',
    ...run.worstCases.map(
      (entry) =>
        `| ${escapeCell(path.basename(entry.sourcePath))} | ${String(entry.sourceLine)} | ${escapeCell(entry.permutationId)} | ${escapeCell(`${entry.baselineTopActionType} → ${entry.permutedTopActionType}`)} | ${format(entry.baselineTopProbabilityMargin)} | ${format(entry.jensenShannonDivergence)} | ${format(entry.maxProbabilityDelta)} | ${format(entry.maxCenteredLogitDelta)} | ${format(entry.valueAbsoluteDelta)} |`
    ),
    '',
  ];
  return `${lines.join('\n')}\n`;
}

function renderStrataTable(
  strata: Readonly<Record<string, TdSymmetryStratum>>
): string[] {
  return [
    '| stratum | decisions | comparisons | preferred-action agreement | mean JS divergence | mean max probability change |',
    '|:---|---:|---:|---:|---:|---:|',
    ...Object.entries(strata).map(
      ([name, value]) =>
        `| ${escapeCell(name)} | ${String(value.samples)} | ${String(value.comparisons)} | ${percent(value.topActionMatchRate)} | ${format(value.meanJensenShannonDivergence)} | ${format(value.meanMaxProbabilityDelta)} |`
    ),
  ];
}

function format(value: number): string {
  return value.toFixed(6);
}

function percent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function safeDivide(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}
