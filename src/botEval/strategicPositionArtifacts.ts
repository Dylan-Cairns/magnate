import { mkdir } from 'node:fs/promises';
import path from 'node:path';

import { collectGitMetadata } from './gitMetadata';
import {
  defaultBotEvalOutputDirectory,
  format,
  writeAtomic,
} from './artifactUtils';
import type {
  StrategicPositionComparisonRunV0,
  StrategicVariantDecisionV0,
} from './strategicPositionComparison';
import type { GitMetadata } from './types';

export const STRATEGIC_POSITION_ARTIFACT_SCHEMA_VERSION = 1 as const;
export const STRATEGIC_POSITION_ARTIFACT_TYPE =
  'ts-strategic-position-comparison' as const;

export interface StrategicPositionArtifactV0 {
  readonly schemaVersion: typeof STRATEGIC_POSITION_ARTIFACT_SCHEMA_VERSION;
  readonly artifactType: typeof STRATEGIC_POSITION_ARTIFACT_TYPE;
  readonly generatedAtUtc: string;
  readonly runtime: { readonly nodeVersion: string };
  readonly git: GitMetadata;
  readonly run: StrategicPositionComparisonRunV0;
}

export interface StrategicPositionArtifactOptionsV0 {
  readonly cwd?: string;
  readonly generatedAtUtc?: string;
  readonly git?: GitMetadata;
  readonly nodeVersion?: string;
}

export interface WrittenStrategicPositionArtifactsV0 {
  readonly artifactPath: string;
  readonly summaryPath: string;
}

export function createStrategicPositionArtifactV0(
  run: StrategicPositionComparisonRunV0,
  options: StrategicPositionArtifactOptionsV0 = {}
): StrategicPositionArtifactV0 {
  return {
    schemaVersion: STRATEGIC_POSITION_ARTIFACT_SCHEMA_VERSION,
    artifactType: STRATEGIC_POSITION_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    runtime: { nodeVersion: options.nodeVersion ?? process.version },
    git: options.git ?? collectGitMetadata(options.cwd),
    run: structuredClone(run),
  };
}

export async function writeStrategicPositionArtifactsV0(
  artifact: StrategicPositionArtifactV0,
  outputDirectory: string
): Promise<WrittenStrategicPositionArtifactsV0> {
  await mkdir(outputDirectory, { recursive: true });
  const artifactPath = path.join(outputDirectory, 'positions.json');
  const summaryPath = path.join(outputDirectory, 'summary.md');
  await writeAtomic(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeAtomic(summaryPath, renderStrategicPositionSummaryV0(artifact));
  return { artifactPath, summaryPath };
}

export function defaultStrategicPositionOutputDirectoryV0(
  generatedAt = new Date()
): string {
  return defaultBotEvalOutputDirectory(
    'strategic-position-comparison-v0',
    generatedAt
  );
}

export function renderStrategicPositionSummaryV0(
  artifact: StrategicPositionArtifactV0
): string {
  const run = artifact.run;
  const lines = [
    '# Strategic Position Comparison v0',
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Catalog: v${String(run.catalogVersion)}; repetitions: ${String(run.repetitions)}; seed scheme: ${run.seedScheme}`,
    '',
    'This is a diagnostic characterization. An expected preference is a reviewed pairwise strategic thesis, not a passing assertion against current bots.',
    '',
    'Policies execute in-process in Node without browser or Web Worker wrappers. Latency is diagnostic for this execution mode.',
    '',
    '## Preference Summary',
    '',
    '| variant | assessed comparisons | preferred selected | preference rate |',
    '|:---|---:|---:|---:|',
  ];

  for (const variant of run.variants) {
    const decisions = allDecisions(run).filter(
      (decision) =>
        decision.variantId === variant.id &&
        decision.matchesExpectedPreference !== null
    );
    const matched = decisions.filter(
      (decision) => decision.matchesExpectedPreference
    ).length;
    lines.push(
      `| ${escapeCell(variant.label)} | ${String(decisions.length)} | ${String(matched)} | ${decisions.length === 0 ? '—' : format(matched / decisions.length)} |`
    );
  }

  lines.push(
    '',
    '## Decisions',
    '',
    '| position | rep | expected focus | variant | selected focus/action | pairwise result | Node in-process latency ms | selected value | visits |',
    '|:---|---:|:---|:---|:---|:---:|---:|---:|---:|'
  );
  for (const position of run.positions) {
    for (const repetition of position.repetitions) {
      for (const decision of repetition.decisions) {
        const diagnostics = decision.searchDiagnostics;
        lines.push(
          `| ${escapeCell(position.positionId)} | ${String(repetition.repetition)} | ${escapeCell(position.expectedPreference?.preferredFocusActionId ?? '—')} | ${escapeCell(decision.variantId)} | ${escapeCell(decision.selectedFocusActionId ?? decision.selectedActionKey)} | ${formatMatch(decision.matchesExpectedPreference)} | ${format(decision.latencyMs)} | ${formatNullable(diagnostics?.selectedActionMeanValue)} | ${integerNullable(diagnostics?.selectedActionVisits)} |`
        );
      }
    }
  }

  lines.push(
    '',
    '## Focus-Action Signals',
    '',
    '| position | rep | variant | focus | heuristic score | heuristic rank | search value | search visits |',
    '|:---|---:|:---|:---|---:|---:|---:|---:|'
  );
  for (const position of run.positions) {
    for (const repetition of position.repetitions) {
      for (const decision of repetition.decisions) {
        for (const signal of decision.focusSignals) {
          lines.push(
            `| ${escapeCell(position.positionId)} | ${String(repetition.repetition)} | ${escapeCell(decision.variantId)} | ${escapeCell(signal.focusActionId)} | ${formatNullable(signal.heuristicScore)} | ${integerNullable(signal.heuristicRank)} | ${formatNullable(signal.searchMeanValue)} | ${integerNullable(signal.searchVisits)} |`
          );
        }
      }
    }
  }

  lines.push('', '## Position Theses', '');
  for (const position of run.positions) {
    lines.push(
      `### ${position.title}`,
      '',
      `Fingerprint: \`${position.positionFingerprint}\``,
      '',
      position.thesis,
      '',
      ...position.expectedFacts.map((fact) => `- ${fact}`),
      ''
    );
  }
  return `${lines.join('\n')}\n`;
}

function allDecisions(
  run: StrategicPositionComparisonRunV0
): StrategicVariantDecisionV0[] {
  return run.positions.flatMap((position) =>
    position.repetitions.flatMap((repetition) => repetition.decisions)
  );
}

function formatMatch(value: boolean | null): string {
  if (value === null) {
    return 'not assessed';
  }
  return value ? 'preferred' : 'declared alternative';
}

function formatNullable(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : format(value);
}

function integerNullable(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : String(value);
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}
