import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import type {
  ActionPolicy,
  SearchDecisionDiagnostics,
} from '../policies/types';
import { createStrategicPositionCatalogV0 } from './strategicPositionCatalog';
import {
  createStrategicPositionArtifactV0,
  renderStrategicPositionSummaryV0,
  writeStrategicPositionArtifactsV0,
} from './strategicPositionArtifacts';
import { runStrategicPositionComparisonV0 } from './strategicPositionComparison';

describe('strategic position artifacts', () => {
  it('writes schema-versioned JSON and a diagnostic Markdown summary', async () => {
    const outputDirectory = await mkdtemp(
      path.join(tmpdir(), 'magnate-strategic-position-')
    );
    try {
      const run = await runStrategicPositionComparisonV0({
        positions: [
          createStrategicPositionCatalogV0().find(
            (position) => position.id === 'minimum-winning-coalition'
          )!,
        ],
        variants: [
          {
            descriptor: {
              kind: 'custom',
              id: 'first-legal',
              label: 'First legal',
              implementationId: 'test:first-legal-v1',
            },
            policy: firstLegalPolicy,
          },
        ],
        now: () => 0,
      });
      const artifact = createStrategicPositionArtifactV0(run, {
        generatedAtUtc: '2026-07-13T00:00:00.000Z',
        git: { commit: 'test', dirty: false },
        nodeVersion: 'v22.23.1',
      });
      const written = await writeStrategicPositionArtifactsV0(
        artifact,
        outputDirectory
      );

      const json = JSON.parse(await readFile(written.artifactPath, 'utf8'));
      const markdown = await readFile(written.summaryPath, 'utf8');
      expect(json).toMatchObject({
        schemaVersion: 1,
        artifactType: 'ts-strategic-position-comparison',
      });
      expect(markdown).toBe(renderStrategicPositionSummaryV0(artifact));
      expect(markdown).toContain('diagnostic characterization');
      expect(markdown).toContain('minimum-winning-coalition');
      expect(markdown).toContain('## Selection Stability');
      expect(markdown).toContain('## Pairwise Focus Gaps');
      expect(markdown).toContain('| unassessed |');
      expect(markdown).toContain('| not applicable |');
    } finally {
      await rm(outputDirectory, { recursive: true, force: true });
    }
  });

  it('distinguishes mixed, partial, and factual-only characterization rows', async () => {
    const position = createStrategicPositionCatalogV0().find(
      (candidate) => candidate.id === 'minimum-winning-coalition'
    )!;
    const run = await runStrategicPositionComparisonV0({
      positions: [position],
      variants: [
        {
          descriptor: {
            kind: 'custom',
            id: 'first-legal',
            label: 'First legal',
            implementationId: 'test:first-legal-v1',
          },
          policy: firstLegalPolicy,
        },
      ],
      now: () => 0,
    });
    const comparisonCase = run.positions[0];
    const baseRepetition = comparisonCase.repetitions[0];
    const baseDecision = baseRepetition.decisions[0];
    const preference = comparisonCase.expectedPreference!;
    const preferred = comparisonCase.focusActions.find(
      (focus) => focus.id === preference.preferredFocusActionId
    )!;
    const alternative = comparisonCase.focusActions.find(
      (focus) => focus.id === preference.overFocusActionIds[0]
    )!;
    const decisions = [
      {
        ...baseDecision,
        selectedActionKey: preferred.actionKey,
        selectedFocusActionId: preferred.id,
        matchesExpectedPreference: true,
      },
      {
        ...baseDecision,
        selectedActionKey: alternative.actionKey,
        selectedFocusActionId: alternative.id,
        matchesExpectedPreference: false,
      },
      {
        ...baseDecision,
        selectedActionKey: 'outside-focus',
        selectedFocusActionId: null,
        matchesExpectedPreference: null,
      },
    ] as const;
    const mixedCase = {
      ...comparisonCase,
      repetitions: decisions.map((decision, repetition) => ({
        ...baseRepetition,
        repetition,
        sharedRandomSeed: `seed-${String(repetition)}`,
        decisions: [decision],
      })),
    };
    const mixedRun = {
      ...run,
      repetitions: 3,
      positions: [mixedCase],
    };
    const mixedMarkdown = renderStrategicPositionSummaryV0(
      createStrategicPositionArtifactV0(mixedRun)
    );
    expect(mixedMarkdown).toContain(
      '| 1 | 1 | 1 | mixed assessed seeds; partially unassessed |'
    );

    const unexpandedSearchRun = {
      ...run,
      positions: [
        {
          ...comparisonCase,
          repetitions: [
            {
              ...baseRepetition,
              decisions: [
                {
                  ...baseDecision,
                  selectedActionKey: 'outside-focus',
                  selectedFocusActionId: null,
                  matchesExpectedPreference: null,
                  searchDiagnostics: searchDiagnosticsWithoutFocusActions(),
                },
              ],
            },
          ],
        },
      ],
    };
    const unexpandedSearchMarkdown = renderStrategicPositionSummaryV0(
      createStrategicPositionArtifactV0(unexpandedSearchRun)
    );
    expect(unexpandedSearchMarkdown).toContain('| none |');

    const factualRun = {
      ...run,
      positions: [
        {
          ...comparisonCase,
          positionId: 'factual-case',
          expectedPreference: null,
        },
      ],
    };
    const factualMarkdown = renderStrategicPositionSummaryV0(
      createStrategicPositionArtifactV0(factualRun)
    );
    expect(factualMarkdown).toContain('| — | — | — | factual only |');

    const pairedRun = {
      ...run,
      positions: [
        { ...comparisonCase, positionId: 'pair-a', randomGroupId: 'pair' },
        { ...comparisonCase, positionId: 'pair-b', randomGroupId: 'pair' },
      ],
    };
    const pairedMarkdown = renderStrategicPositionSummaryV0(
      createStrategicPositionArtifactV0(pairedRun)
    );
    expect(pairedMarkdown).toContain('| pair | 0 | first-legal |');
    expect(pairedMarkdown).toContain('pair-a=');
    expect(pairedMarkdown).toContain('pair-b=');
  });
});

const firstLegalPolicy: ActionPolicy = {
  selectAction(context) {
    return context.legalActions[0];
  },
};

function searchDiagnosticsWithoutFocusActions(): SearchDecisionDiagnostics {
  return {
    kind: 'search',
    legalRootActions: 1,
    expandedRootActions: 1,
    rootVisitBudget: 1,
    configProxyCost: 1,
    maxSimulatedActionSteps: 1,
    simulatedActionSteps: 1,
    terminalRollouts: 0,
    terminalRate: 0,
    selectedActionKey: 'outside-focus',
    selectedActionVisits: 1,
    selectedActionMeanValue: 0,
    selectedActionTerminalRollouts: 0,
    selectedActionTerminalRate: 0,
    rootActions: [
      {
        actionKey: 'outside-focus',
        visits: 1,
        meanValue: 0,
        terminalRollouts: 0,
        terminalRate: 0,
        prior: 1,
      },
    ],
  };
}
