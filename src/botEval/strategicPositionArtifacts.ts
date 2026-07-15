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
    '# Strategic Position Comparison',
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Artifact schema: v${String(artifact.schemaVersion)}; catalog: v${String(run.catalogVersion)}; repetitions: ${String(run.repetitions)} starting at ${String(run.repetitionStart ?? 0)}; seed scheme: ${run.seedScheme}`,
    '',
    'This is a diagnostic characterization. An expected preference is a reviewed pairwise strategic thesis, not a passing assertion against current bots.',
    '',
    'Policies execute in-process in Node without browser or Web Worker wrappers. Latency is diagnostic for this execution mode.',
    '',
    'Repeated seeds characterize policy stability in each fixed position; they are not independent games or proof that a reviewed preference is strategically correct. Direct heuristic v2 is deterministic, so its repeated rows confirm repeatability rather than add evidence.',
    '',
    '## Selection Stability',
    '',
    '| position | variant | seeds | selection histogram | modal count | preferred | declared alternative | unassessed | observed result |',
    '|:---|:---|---:|:---|---:|---:|---:|---:|:---|',
  ];

  for (const position of run.positions) {
    for (const variant of run.variants) {
      const decisions = decisionsForVariant(position, variant.id);
      const preferred = decisions.filter(
        (decision) => decision.matchesExpectedPreference === true
      ).length;
      const alternative = decisions.filter(
        (decision) => decision.matchesExpectedPreference === false
      ).length;
      const unassessed = decisions.length - preferred - alternative;
      const histogram = selectionHistogram(decisions);
      lines.push(
        `| ${escapeCell(position.positionId)} | ${escapeCell(variant.label)} | ${String(decisions.length)} | ${escapeCell(histogram.label)} | ${histogram.modalCount === 0 ? '—' : `${String(histogram.modalCount)}/${String(decisions.length)}`} | ${position.expectedPreference ? String(preferred) : '—'} | ${position.expectedPreference ? String(alternative) : '—'} | ${position.expectedPreference ? String(unassessed) : '—'} | ${escapeCell(observedPreferenceResult({ position, preferred, alternative, unassessed }))} |`
      );
    }
  }

  lines.push(
    '',
    '## Pairwise Focus Gaps',
    '',
    'Positive gaps favor the reviewed preferred focus action. Heuristic scores and search values are only compared within one variant and position; search means use adaptive, potentially unequal visit counts.',
    '',
    '| position | rep | variant | preferred | alternative | heuristic gap | search-value gap | preferred visits | alternative visits | search coverage |',
    '|:---|---:|:---|:---|:---|---:|---:|---:|---:|:---|'
  );
  for (const position of run.positions) {
    const preference = position.expectedPreference;
    if (!preference) {
      continue;
    }
    for (const repetition of position.repetitions) {
      for (const decision of repetition.decisions) {
        const preferredSignal = decision.focusSignals.find(
          (signal) => signal.focusActionId === preference.preferredFocusActionId
        );
        for (const alternativeId of preference.overFocusActionIds) {
          const alternativeSignal = decision.focusSignals.find(
            (signal) => signal.focusActionId === alternativeId
          );
          lines.push(
            `| ${escapeCell(position.positionId)} | ${String(repetition.repetition)} | ${escapeCell(decision.variantId)} | ${escapeCell(preference.preferredFocusActionId)} | ${escapeCell(alternativeId)} | ${formatGap(preferredSignal?.heuristicScore, alternativeSignal?.heuristicScore)} | ${formatGap(preferredSignal?.searchMeanValue, alternativeSignal?.searchMeanValue)} | ${integerNullable(preferredSignal?.searchVisits)} | ${integerNullable(alternativeSignal?.searchVisits)} | ${searchCoverage(decision.searchDiagnostics !== null, preferredSignal?.searchVisits, alternativeSignal?.searchVisits)} |`
          );
        }
      }
    }
  }

  lines.push(
    '',
    '## Counterfactual Groups',
    '',
    'Members of each group share repetition seeds. Selection transitions are diagnostic; adaptive search values are not fixed-budget paired estimates.',
    '',
    '| group | rep | variant | selections |',
    '|:---|---:|:---|:---|'
  );
  for (const group of counterfactualGroups(run)) {
    const repetitions = new Set(
      group.positions.flatMap((position) =>
        position.repetitions.map((entry) => entry.repetition)
      )
    );
    for (const repetition of [...repetitions].sort(
      (left, right) => left - right
    )) {
      for (const variant of run.variants) {
        const selections = group.positions.map((position) => {
          const decision = position.repetitions
            .find((entry) => entry.repetition === repetition)
            ?.decisions.find((entry) => entry.variantId === variant.id);
          return `${position.positionId}=${decision ? selectedDecisionLabel(decision) : 'missing'}`;
        });
        lines.push(
          `| ${escapeCell(group.id)} | ${String(repetition)} | ${escapeCell(variant.id)} | ${escapeCell(selections.join('; '))} |`
        );
      }
    }
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

function formatMatch(value: boolean | null): string {
  if (value === null) {
    return 'not assessed';
  }
  return value ? 'preferred' : 'declared alternative';
}

function decisionsForVariant(
  position: StrategicPositionComparisonRunV0['positions'][number],
  variantId: string
): StrategicVariantDecisionV0[] {
  return position.repetitions.flatMap((repetition) =>
    repetition.decisions.filter((decision) => decision.variantId === variantId)
  );
}

function selectedDecisionLabel(decision: StrategicVariantDecisionV0): string {
  return decision.selectedFocusActionId ?? decision.selectedActionKey;
}

function selectionHistogram(decisions: readonly StrategicVariantDecisionV0[]): {
  readonly label: string;
  readonly modalCount: number;
} {
  const counts = new Map<string, number>();
  for (const decision of decisions) {
    const label = selectedDecisionLabel(decision);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const entries = [...counts.entries()].sort(
    ([leftLabel, leftCount], [rightLabel, rightCount]) =>
      rightCount - leftCount || leftLabel.localeCompare(rightLabel)
  );
  return {
    label:
      entries.length === 0
        ? '—'
        : entries
            .map(([label, count]) => `${label} × ${String(count)}`)
            .join('; '),
    modalCount: entries[0]?.[1] ?? 0,
  };
}

function observedPreferenceResult({
  position,
  preferred,
  alternative,
  unassessed,
}: {
  position: StrategicPositionComparisonRunV0['positions'][number];
  preferred: number;
  alternative: number;
  unassessed: number;
}): string {
  if (!position.expectedPreference) {
    return 'factual only';
  }
  if (preferred + alternative === 0) {
    return 'unassessed';
  }
  if (preferred > 0 && alternative > 0) {
    return unassessed > 0
      ? 'mixed assessed seeds; partially unassessed'
      : 'mixed assessed seeds';
  }
  if (unassessed > 0) {
    return preferred > 0
      ? 'all assessed seeds preferred; partially unassessed'
      : 'all assessed seeds alternative; partially unassessed';
  }
  if (alternative === 0) {
    return 'all assessed seeds preferred';
  }
  return 'all assessed seeds alternative';
}

function formatGap(
  preferred: number | null | undefined,
  alternative: number | null | undefined
): string {
  if (
    preferred === null ||
    preferred === undefined ||
    alternative === null ||
    alternative === undefined
  ) {
    return '—';
  }
  return format(preferred - alternative);
}

function searchCoverage(
  hasSearchDiagnostics: boolean,
  preferredVisits: number | null | undefined,
  alternativeVisits: number | null | undefined
): string {
  if (!hasSearchDiagnostics) {
    return 'not applicable';
  }
  const preferredExpanded =
    preferredVisits !== null &&
    preferredVisits !== undefined &&
    preferredVisits > 0;
  const alternativeExpanded =
    alternativeVisits !== null &&
    alternativeVisits !== undefined &&
    alternativeVisits > 0;
  if (preferredExpanded && alternativeExpanded) {
    return 'both';
  }
  if (preferredExpanded || alternativeExpanded) {
    return 'partial';
  }
  return 'none';
}

function counterfactualGroups(run: StrategicPositionComparisonRunV0): Array<{
  readonly id: string;
  readonly positions: StrategicPositionComparisonRunV0['positions'];
}> {
  const byId = new Map<
    string,
    StrategicPositionComparisonRunV0['positions'][number][]
  >();
  for (const position of run.positions) {
    const positions = byId.get(position.randomGroupId) ?? [];
    positions.push(position);
    byId.set(position.randomGroupId, positions);
  }
  return [...byId.entries()]
    .filter((entry) => entry[1].length > 1)
    .map(([id, positions]) => ({ id, positions }));
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
