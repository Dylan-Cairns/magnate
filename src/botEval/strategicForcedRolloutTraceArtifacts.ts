import { mkdir } from 'node:fs/promises';
import path from 'node:path';

import { collectGitMetadata } from './gitMetadata';
import {
  defaultBotEvalOutputDirectory,
  format,
  writeAtomic,
} from './artifactUtils';
import type {
  StrategicContinuationStatusV0,
  StrategicForcedRolloutCaseV0,
  StrategicForcedRolloutGuideV0,
  StrategicForcedRolloutTraceRunV0,
  StrategicForcedRolloutTraceV0,
} from './strategicForcedRolloutTrace';
import type { GitMetadata } from './types';

export const STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_SCHEMA_VERSION =
  1 as const;
export const STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_TYPE =
  'ts-strategic-forced-rollout-trace' as const;

const ROOT_FOCUS_ACTION_IDS = ['preserve-option', 'overwrite-option'] as const;
const TRACE_GUIDES: readonly StrategicForcedRolloutGuideV0[] = [
  'td',
  'heuristic-v2',
];
const CONTINUATION_STATUSES: readonly StrategicContinuationStatusV0[] = [
  'realized',
  'legal-but-not-used',
  'held-but-never-legal',
  'opponent-held',
  'not-reached-by-player-a',
];

export interface StrategicForcedRolloutTraceArtifactV0 {
  readonly schemaVersion: typeof STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_SCHEMA_VERSION;
  readonly artifactType: typeof STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_TYPE;
  readonly generatedAtUtc: string;
  readonly runtime: { readonly nodeVersion: string };
  readonly git: GitMetadata;
  readonly run: StrategicForcedRolloutTraceRunV0;
}

export interface StrategicForcedRolloutTraceArtifactOptionsV0 {
  readonly cwd?: string;
  readonly generatedAtUtc?: string;
  readonly git?: GitMetadata;
  readonly nodeVersion?: string;
}

export interface WrittenStrategicForcedRolloutTraceArtifactsV0 {
  readonly artifactPath: string;
  readonly summaryPath: string;
}

export function createStrategicForcedRolloutTraceArtifactV0(
  run: StrategicForcedRolloutTraceRunV0,
  options: StrategicForcedRolloutTraceArtifactOptionsV0 = {}
): StrategicForcedRolloutTraceArtifactV0 {
  return {
    schemaVersion: STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_SCHEMA_VERSION,
    artifactType: STRATEGIC_FORCED_ROLLOUT_TRACE_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    runtime: { nodeVersion: options.nodeVersion ?? process.version },
    git: options.git ?? collectGitMetadata(options.cwd),
    run: structuredClone(run),
  };
}

export async function writeStrategicForcedRolloutTraceArtifactsV0(
  artifact: StrategicForcedRolloutTraceArtifactV0,
  outputDirectory: string
): Promise<WrittenStrategicForcedRolloutTraceArtifactsV0> {
  await mkdir(outputDirectory, { recursive: true });
  const artifactPath = path.join(outputDirectory, 'traces.json');
  const summaryPath = path.join(outputDirectory, 'summary.md');
  await writeAtomic(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeAtomic(
    summaryPath,
    renderStrategicForcedRolloutTraceSummaryV0(artifact)
  );
  return { artifactPath, summaryPath };
}

export function defaultStrategicForcedRolloutTraceOutputDirectoryV0(
  generatedAt = new Date()
): string {
  return defaultBotEvalOutputDirectory(
    'strategic-forced-rollout-trace-v0',
    generatedAt
  );
}

export function renderStrategicForcedRolloutTraceSummaryV0(
  artifact: StrategicForcedRolloutTraceArtifactV0
): string {
  const run = artifact.run;
  const lines = [
    '# Strategic Forced-Rollout Trace',
    '',
    `Generated: ${artifact.generatedAtUtc}`,
    '',
    `Artifact schema: v${String(artifact.schemaVersion)}; trace schema: v${String(run.schemaVersion)}; catalog: v${String(run.catalogVersion)}; model index: \`${run.modelIndexPath}\``,
    '',
    '## Matched-Scenario Method',
    '',
    'For each position, repetition, and scenario index, preserve and overwrite are forced through the same sampled hidden assignment, simulated engine seed, and rollout random seed under both TD and heuristic-v2 rollout guidance. Root selection, root priors, and adaptive root-visit allocation do not participate. Rollout epsilon is zero, and every recorded trace reaches terminal scoring before the depth limit.',
    '',
    'Each trajectory follows the named guide. At every non-root state, the trace also records what TD and heuristic v2 would propose from that exact same state. These are fixed-position diagnostics, not full-game strength estimates.',
    '',
    '## Outcomes by Position, Forced Root, and Guide',
    '',
    '| position | forced root | trajectory guide | scenarios | mean score | PlayerA W/D/L | continuation status histogram |',
    '|:---|:---|:---|---:|---:|:---|:---|',
  ];

  for (const position of run.positions) {
    for (const rootFocusActionId of ROOT_FOCUS_ACTION_IDS) {
      for (const guide of TRACE_GUIDES) {
        const traces = tracesFor(position, rootFocusActionId, guide);
        const wins = traces.filter(
          (entry) => entry.trace.finalScore.winner === 'PlayerA'
        ).length;
        const draws = traces.filter(
          (entry) => entry.trace.finalScore.winner === 'Draw'
        ).length;
        const losses = traces.filter(
          (entry) => entry.trace.finalScore.winner === 'PlayerB'
        ).length;
        lines.push(
          `| ${escapeCell(position.positionId)} | ${rootFocusActionId} | ${guide} | ${String(traces.length)} | ${formatMeanScore(traces)} | ${String(wins)}/${String(draws)}/${String(losses)} | ${escapeCell(continuationStatusHistogram(traces))} |`
        );
      }
    }
  }

  lines.push(
    '',
    '## First Continuation-Relevant Proposal Divergences',
    '',
    "Each non-realized preserve trace contributes at most one observation: its first post-root Player A state where TD and heuristic v2 proposed different actions while the target was still in Player A's hand or pending in the draw pile. The trajectory action is what the named guide actually followed; the counterfactual proposal comes from the other guide on that same state.",
    '',
    'Counts are diagnostic correlations. They identify recurring decision patterns associated with a missed continuation, but do not establish that the first proposal divergence caused the continuation to be missed.',
    ''
  );

  const firstRelevantDivergences =
    firstContinuationRelevantProposalDivergenceGroups(run);
  if (firstRelevantDivergences.length === 0) {
    lines.push(
      'No first continuation-relevant proposal divergences were recorded.'
    );
  } else {
    lines.push(
      '| position | trajectory guide | continuation status | target location | trajectory action | counterfactual proposal (other guide) | traces |',
      '|:---|:---|:---|:---|:---|:---|---:|'
    );
    for (const entry of firstRelevantDivergences) {
      lines.push(
        `| ${escapeCell(entry.positionId)} | ${entry.guide} | ${entry.continuationStatus} | ${entry.targetLocationCategory} | ${escapeCell(entry.trajectoryAction)} | ${escapeCell(entry.counterfactualProposal)} | ${String(entry.count)} |`
      );
    }
  }

  lines.push(
    '',
    '## Target-Legal Proposal Divergences',
    '',
    'Rows below are states where Player A could legally develop the target card in the valuable lane, but TD and heuristic v2 proposed different actions. The trajectory action is the proposal actually followed by the named trajectory guide.',
    ''
  );

  const divergences = targetLegalProposalDivergences(run);
  if (divergences.length === 0) {
    lines.push('No target-legal proposal divergences were recorded.');
  } else {
    lines.push(
      '| position | rep | scenario | forced root | trajectory guide | step | trajectory action | TD proposal | heuristic-v2 proposal |',
      '|:---|---:|---:|:---|:---|---:|:---|:---|:---|'
    );
    for (const entry of divergences) {
      lines.push(
        `| ${escapeCell(entry.positionId)} | ${String(entry.repetition)} | ${String(entry.scenarioIndex)} | ${entry.rootFocusActionId} | ${entry.guide} | ${String(entry.stepIndex)} | ${escapeCell(entry.trajectoryAction)} | ${escapeCell(entry.tdAction)} | ${escapeCell(entry.heuristicAction)} |`
      );
    }
  }

  return `${lines.join('\n')}\n`;
}

interface LocatedTrace {
  readonly repetition: number;
  readonly scenarioIndex: number;
  readonly trace: StrategicForcedRolloutTraceV0;
}

function tracesFor(
  position: StrategicForcedRolloutCaseV0,
  rootFocusActionId: (typeof ROOT_FOCUS_ACTION_IDS)[number],
  guide: StrategicForcedRolloutGuideV0
): LocatedTrace[] {
  return position.repetitions.flatMap((repetition) =>
    repetition.scenarios.flatMap((scenario) =>
      scenario.traces
        .filter(
          (trace) =>
            trace.rootFocusActionId === rootFocusActionId &&
            trace.guide === guide
        )
        .map((trace) => ({
          repetition: repetition.repetition,
          scenarioIndex: scenario.scenarioIndex,
          trace,
        }))
    )
  );
}

function formatMeanScore(traces: readonly LocatedTrace[]): string {
  if (traces.length === 0) {
    return '—';
  }
  return format(
    traces.reduce((total, entry) => total + entry.trace.score, 0) /
      traces.length
  );
}

function continuationStatusHistogram(traces: readonly LocatedTrace[]): string {
  const counts = new Map<StrategicContinuationStatusV0, number>(
    CONTINUATION_STATUSES.map((status) => [status, 0])
  );
  for (const entry of traces) {
    const status = entry.trace.continuationStatus;
    counts.set(status, (counts.get(status) ?? 0) + 1);
  }
  return CONTINUATION_STATUSES.map(
    (status) => `${status}=${String(counts.get(status) ?? 0)}`
  ).join('; ');
}

type ContinuationRelevantTargetLocationCategory = 'hand' | 'pending draw';

interface FirstContinuationRelevantProposalDivergenceGroup {
  readonly positionId: string;
  readonly guide: StrategicForcedRolloutGuideV0;
  readonly continuationStatus: Exclude<
    StrategicContinuationStatusV0,
    'realized'
  >;
  readonly targetLocationCategory: ContinuationRelevantTargetLocationCategory;
  readonly trajectoryAction: string;
  readonly counterfactualProposal: string;
  readonly count: number;
}

function firstContinuationRelevantProposalDivergenceGroups(
  run: StrategicForcedRolloutTraceRunV0
): FirstContinuationRelevantProposalDivergenceGroup[] {
  const byKey = new Map<
    string,
    FirstContinuationRelevantProposalDivergenceGroup
  >();
  for (const position of run.positions) {
    for (const repetition of position.repetitions) {
      for (const scenario of repetition.scenarios) {
        for (const trace of scenario.traces) {
          if (
            trace.rootFocusActionId !== 'preserve-option' ||
            trace.continuationStatus === 'realized'
          ) {
            continue;
          }
          const first = firstContinuationRelevantProposalDivergence(trace);
          if (!first) {
            continue;
          }
          const proposals = first.step.proposals!;
          const counterfactual =
            trace.guide === 'td' ? proposals.heuristicV2 : proposals.td;
          const value = {
            positionId: position.positionId,
            guide: trace.guide,
            continuationStatus: trace.continuationStatus,
            targetLocationCategory: first.targetLocationCategory,
            trajectoryAction: actionDescription(
              first.step.actionLabel,
              first.step.actionKey
            ),
            counterfactualProposal: actionDescription(
              counterfactual.actionLabel,
              counterfactual.actionKey
            ),
          } satisfies Omit<
            FirstContinuationRelevantProposalDivergenceGroup,
            'count'
          >;
          const key = JSON.stringify(value);
          const existing = byKey.get(key);
          byKey.set(key, {
            ...value,
            count: (existing?.count ?? 0) + 1,
          });
        }
      }
    }
  }

  return [...byKey.values()].sort(
    (left, right) =>
      left.positionId.localeCompare(right.positionId) ||
      TRACE_GUIDES.indexOf(left.guide) - TRACE_GUIDES.indexOf(right.guide) ||
      right.count - left.count ||
      left.continuationStatus.localeCompare(right.continuationStatus) ||
      left.targetLocationCategory.localeCompare(right.targetLocationCategory) ||
      left.trajectoryAction.localeCompare(right.trajectoryAction) ||
      left.counterfactualProposal.localeCompare(right.counterfactualProposal)
  );
}

function firstContinuationRelevantProposalDivergence(
  trace: StrategicForcedRolloutTraceV0
): {
  readonly step: StrategicForcedRolloutTraceV0['steps'][number];
  readonly targetLocationCategory: ContinuationRelevantTargetLocationCategory;
} | null {
  let first:
    | {
        readonly step: StrategicForcedRolloutTraceV0['steps'][number];
        readonly targetLocationCategory: ContinuationRelevantTargetLocationCategory;
      }
    | undefined;
  for (const step of trace.steps) {
    const proposals = step.proposals;
    const targetLocationCategory = continuationRelevantTargetLocationCategory(
      step.targetLocationBefore
    );
    if (
      step.stepIndex <= 0 ||
      step.decisionPlayer !== 'PlayerA' ||
      !proposals ||
      proposals.td.actionKey === proposals.heuristicV2.actionKey ||
      !targetLocationCategory
    ) {
      continue;
    }
    if (!first || step.stepIndex < first.step.stepIndex) {
      first = { step, targetLocationCategory };
    }
  }
  return first ?? null;
}

function continuationRelevantTargetLocationCategory(
  location: string
): ContinuationRelevantTargetLocationCategory | null {
  if (location === 'PlayerA-hand') {
    return 'hand';
  }
  return location.startsWith('draw:') ? 'pending draw' : null;
}

interface TargetLegalProposalDivergence {
  readonly positionId: string;
  readonly repetition: number;
  readonly scenarioIndex: number;
  readonly rootFocusActionId: StrategicForcedRolloutTraceV0['rootFocusActionId'];
  readonly guide: StrategicForcedRolloutGuideV0;
  readonly stepIndex: number;
  readonly trajectoryAction: string;
  readonly tdAction: string;
  readonly heuristicAction: string;
}

function targetLegalProposalDivergences(
  run: StrategicForcedRolloutTraceRunV0
): TargetLegalProposalDivergence[] {
  const entries: TargetLegalProposalDivergence[] = [];
  for (const position of run.positions) {
    for (const repetition of position.repetitions) {
      for (const scenario of repetition.scenarios) {
        for (const trace of scenario.traces) {
          for (const step of trace.steps) {
            const proposals = step.proposals;
            if (
              !proposals ||
              proposals.td.actionKey === proposals.heuristicV2.actionKey ||
              !step.targetLegalDistrictsBefore.includes(
                position.valuableDistrictId
              )
            ) {
              continue;
            }
            entries.push({
              positionId: position.positionId,
              repetition: repetition.repetition,
              scenarioIndex: scenario.scenarioIndex,
              rootFocusActionId: trace.rootFocusActionId,
              guide: trace.guide,
              stepIndex: step.stepIndex,
              trajectoryAction: actionDescription(
                step.actionLabel,
                step.actionKey
              ),
              tdAction: actionDescription(
                proposals.td.actionLabel,
                proposals.td.actionKey
              ),
              heuristicAction: actionDescription(
                proposals.heuristicV2.actionLabel,
                proposals.heuristicV2.actionKey
              ),
            });
          }
        }
      }
    }
  }
  return entries;
}

function actionDescription(label: string, actionKey: string): string {
  return `${label} [${actionKey}]`;
}

function escapeCell(value: string): string {
  return value.replace(/\|/g, '\\|').replace(/\r?\n/g, ' ');
}
