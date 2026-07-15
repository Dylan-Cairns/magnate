import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import type { FinalScore, ResourcePool } from '../engine/types';
import { STRATEGIC_POSITION_CATALOG_VERSION } from './strategicPositionCatalog';
import { STRATEGIC_POSITION_COMPARISON_SEED_SCHEME } from './strategicPositionComparison';
import {
  STRATEGIC_FORCED_ROLLOUT_TRACE_SCHEMA_VERSION,
  type StrategicContinuationStatusV0,
  type StrategicForcedRolloutCaseV0,
  type StrategicForcedRolloutGuideV0,
  type StrategicForcedRolloutTraceRunV0,
  type StrategicForcedRolloutTraceV0,
} from './strategicForcedRolloutTrace';
import {
  createStrategicForcedRolloutTraceArtifactV0,
  defaultStrategicForcedRolloutTraceOutputDirectoryV0,
  renderStrategicForcedRolloutTraceSummaryV0,
  writeStrategicForcedRolloutTraceArtifactsV0,
} from './strategicForcedRolloutTraceArtifacts';

describe('strategic forced-rollout trace artifacts', () => {
  it('writes schema-versioned JSON and an aggregated Markdown summary', async () => {
    const outputDirectory = await mkdtemp(
      path.join(tmpdir(), 'magnate-forced-rollout-trace-')
    );
    try {
      const artifact = createStrategicForcedRolloutTraceArtifactV0(
        fixtureRun(true),
        {
          generatedAtUtc: '2026-07-14T01:02:03.000Z',
          git: { commit: 'test-commit', dirty: false },
          nodeVersion: 'v20.19.0',
        }
      );
      const written = await writeStrategicForcedRolloutTraceArtifactsV0(
        artifact,
        outputDirectory
      );

      const json = JSON.parse(await readFile(written.artifactPath, 'utf8'));
      const markdown = await readFile(written.summaryPath, 'utf8');
      expect(path.basename(written.artifactPath)).toBe('traces.json');
      expect(path.basename(written.summaryPath)).toBe('summary.md');
      expect(json).toEqual(artifact);
      expect(json).toMatchObject({
        schemaVersion: 1,
        artifactType: 'ts-strategic-forced-rollout-trace',
        runtime: { nodeVersion: 'v20.19.0' },
        git: { commit: 'test-commit', dirty: false },
      });
      expect(markdown).toBe(
        renderStrategicForcedRolloutTraceSummaryV0(artifact)
      );
      expect(markdown).toContain('## Matched-Scenario Method');
      expect(markdown).toContain(
        'same sampled hidden assignment, simulated engine seed, and rollout random seed'
      );
      expect(markdown).toContain(
        '| known-hand-optionality-original | preserve-option | td | 2 | 0.500 | 1/1/0 | realized=1; legal-but-not-used=1; held-but-never-legal=0; opponent-held=0; not-reached-by-player-a=0 |'
      );
      expect(markdown).toContain(
        '| known-hand-optionality-original | 7 | 0 | preserve-option | td | 1 | Develop Author \\| valuable [develop-outright:6:D1] | Develop Author \\| valuable [develop-outright:6:D1] | Sell Author [sell-card:6] |'
      );
    } finally {
      await rm(outputDirectory, { recursive: true, force: true });
    }
  });

  it('reports no target-legal proposal rows when proposals agree', () => {
    const markdown = renderStrategicForcedRolloutTraceSummaryV0(
      createStrategicForcedRolloutTraceArtifactV0(fixtureRun(false), {
        generatedAtUtc: '2026-07-14T01:02:03.000Z',
        git: { commit: null, dirty: null },
        nodeVersion: 'test-node',
      })
    );

    expect(markdown).toContain(
      'No target-legal proposal divergences were recorded.'
    );
    expect(markdown).not.toContain('| trajectory action | TD proposal |');
    expect(
      defaultStrategicForcedRolloutTraceOutputDirectoryV0(
        new Date('2026-07-14T01:02:03.000Z')
      )
    ).toBe(
      path.join(
        'artifacts',
        'ts-bot-evals',
        '20260714T010203Z-strategic-forced-rollout-trace-v0'
      )
    );
  });

  it('groups the first pending-draw proposal split for missed preserve continuations', () => {
    const base = fixtureRun(false);
    const knownPosition = base.positions[0];
    const knownRepetition = knownPosition.repetitions[0];
    const artifact = createStrategicForcedRolloutTraceArtifactV0(
      {
        ...base,
        scenarioIndices: [0, 1, 2],
        positions: [
          {
            ...knownPosition,
            repetitions: [
              {
                ...knownRepetition,
                scenarios: [
                  ...knownRepetition.scenarios,
                  fixtureScenario(2, false),
                ],
              },
            ],
          },
          fixtureUnknownContinuationCase(),
        ],
      },
      {
        generatedAtUtc: '2026-07-14T01:02:03.000Z',
        git: { commit: 'test', dirty: false },
        nodeVersion: 'test-node',
      }
    );
    const markdown = renderStrategicForcedRolloutTraceSummaryV0(artifact);
    const section = markdown.slice(
      markdown.indexOf('## First Continuation-Relevant Proposal Divergences'),
      markdown.indexOf('## Target-Legal Proposal Divergences')
    );
    const repeatedRow =
      '| unknown-pool-optionality-mirror | heuristic-v2 | not-reached-by-player-a | pending draw | Trade Wyrms for Moons [trade:Wyrms:Moons] | End turn [end-turn] | 2 |';
    const singletonRow =
      '| unknown-pool-optionality-mirror | heuristic-v2 | not-reached-by-player-a | pending draw | Sell The Island [sell-card:29] | End turn [end-turn] | 1 |';

    expect(section).toContain('diagnostic correlations');
    expect(section).toContain(repeatedRow);
    expect(section).toContain(singletonRow);
    expect(section.indexOf(repeatedRow)).toBeLessThan(
      section.indexOf(singletonRow)
    );
    expect(section).not.toContain('Trade Leaves for Suns');
    expect(markdown).toContain('## Target-Legal Proposal Divergences');
  });
});

function fixtureRun(
  includeTargetLegalDivergence: boolean
): StrategicForcedRolloutTraceRunV0 {
  return {
    schemaVersion: STRATEGIC_FORCED_ROLLOUT_TRACE_SCHEMA_VERSION,
    catalogVersion: STRATEGIC_POSITION_CATALOG_VERSION,
    seedScheme: STRATEGIC_POSITION_COMPARISON_SEED_SCHEME,
    modelIndexPath: 'model-packs/index.json',
    config: {
      worlds: 50,
      rollouts: 1,
      depth: 40,
      maxRootActions: 16,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    },
    repetitionIds: [7],
    scenarioIndices: [0, 1],
    positions: [
      {
        positionId: 'known-hand-optionality-original',
        randomGroupId: 'known-hand-optionality-mirror-pair',
        family: 'known-hand',
        targetCardId: '6',
        targetCardName: 'The Author',
        valuableDistrictId: 'D1',
        alternativeDistrictId: 'D4',
        focusActions: [
          { id: 'preserve-option', actionKey: 'preserve-root' },
          { id: 'overwrite-option', actionKey: 'overwrite-root' },
        ],
        repetitions: [
          {
            repetition: 7,
            sharedRandomSeed: 'shared-seed-7',
            scenarios: [
              fixtureScenario(0, includeTargetLegalDivergence),
              fixtureScenario(1, false),
            ],
          },
        ],
      },
    ],
  };
}

function fixtureScenario(
  scenarioIndex: number,
  includeTargetLegalDivergence: boolean
) {
  return {
    scenarioIndex,
    worldIndex: scenarioIndex,
    engineSeed: `engine-${String(scenarioIndex)}`,
    rolloutRandomSeed: `rollout-${String(scenarioIndex)}`,
    hiddenAssignmentFingerprint: `sha256:hidden-${String(scenarioIndex)}`,
    targetInitialLocation: 'PlayerA-hand',
    traces: [
      fixtureTrace({
        rootFocusActionId: 'preserve-option',
        guide: 'td',
        score: scenarioIndex === 0 ? 1 : 0,
        winner: scenarioIndex === 0 ? 'PlayerA' : 'Draw',
        continuationStatus:
          scenarioIndex === 0 ? 'realized' : 'legal-but-not-used',
        includeTargetLegalDivergence,
      }),
      fixtureTrace({
        rootFocusActionId: 'preserve-option',
        guide: 'heuristic-v2',
        score: 1,
        winner: 'PlayerA',
        continuationStatus: 'realized',
      }),
      fixtureTrace({
        rootFocusActionId: 'overwrite-option',
        guide: 'td',
        score: -1,
        winner: 'PlayerB',
        continuationStatus: 'held-but-never-legal',
      }),
      fixtureTrace({
        rootFocusActionId: 'overwrite-option',
        guide: 'heuristic-v2',
        score: -1,
        winner: 'PlayerB',
        continuationStatus: 'held-but-never-legal',
      }),
    ],
  };
}

function fixtureUnknownContinuationCase(): StrategicForcedRolloutCaseV0 {
  return {
    positionId: 'unknown-pool-optionality-mirror',
    randomGroupId: 'unknown-pool-optionality-mirror-pair',
    family: 'unknown-pool',
    targetCardId: '20',
    targetCardName: 'The Penitent',
    valuableDistrictId: 'D4',
    alternativeDistrictId: 'D1',
    focusActions: [
      { id: 'preserve-option', actionKey: 'preserve-unknown-root' },
      { id: 'overwrite-option', actionKey: 'overwrite-unknown-root' },
    ],
    repetitions: [
      {
        repetition: 7,
        sharedRandomSeed: 'unknown-shared-seed-7',
        scenarios: [
          fixtureUnknownContinuationScenario(0, 'trade'),
          fixtureUnknownContinuationScenario(1, 'trade'),
          fixtureUnknownContinuationScenario(2, 'sell'),
        ],
      },
    ],
  };
}

function fixtureUnknownContinuationScenario(
  scenarioIndex: number,
  firstAction: 'trade' | 'sell'
) {
  return {
    scenarioIndex,
    worldIndex: scenarioIndex,
    engineSeed: `unknown-engine-${String(scenarioIndex)}`,
    rolloutRandomSeed: `unknown-rollout-${String(scenarioIndex)}`,
    hiddenAssignmentFingerprint: `sha256:unknown-${String(scenarioIndex)}`,
    targetInitialLocation: 'draw:0',
    traces: [
      fixtureTrace({
        rootFocusActionId: 'preserve-option',
        guide: 'td',
        score: -1,
        winner: 'PlayerB',
        continuationStatus: 'not-reached-by-player-a',
      }),
      fixtureUnknownContinuationMissTrace(firstAction),
      fixtureTrace({
        rootFocusActionId: 'overwrite-option',
        guide: 'td',
        score: -1,
        winner: 'PlayerB',
        continuationStatus: 'held-but-never-legal',
      }),
      fixtureTrace({
        rootFocusActionId: 'overwrite-option',
        guide: 'heuristic-v2',
        score: -1,
        winner: 'PlayerB',
        continuationStatus: 'held-but-never-legal',
      }),
    ],
  };
}

function fixtureUnknownContinuationMissTrace(
  firstAction: 'trade' | 'sell'
): StrategicForcedRolloutTraceV0 {
  const base = fixtureTrace({
    rootFocusActionId: 'preserve-option',
    guide: 'heuristic-v2',
    score: -1,
    winner: 'PlayerB',
    continuationStatus: 'not-reached-by-player-a',
  });
  const template = base.steps[1];
  const firstActionKey =
    firstAction === 'trade' ? 'trade:Wyrms:Moons' : 'sell-card:29';
  const firstActionLabel =
    firstAction === 'trade' ? 'Trade Wyrms for Moons' : 'Sell The Island';
  return {
    ...base,
    simulatedActionSteps: 3,
    steps: [
      base.steps[0],
      {
        ...template,
        stepIndex: 1,
        actionKey: firstActionKey,
        actionLabel: firstActionLabel,
        legalActionKeys: ['end-turn', firstActionKey],
        proposals: {
          td: {
            actionKey: 'end-turn',
            actionLabel: 'End turn',
            rank: 0,
            rawScore: 2,
          },
          heuristicV2: {
            actionKey: firstActionKey,
            actionLabel: firstActionLabel,
            rank: 0,
            rawScore: 3,
          },
        },
        playerAHandBefore: ['25', '29'],
        playerAHandAfter: ['25', '29'],
        targetLocationBefore: 'draw:0',
        targetLocationAfter: 'draw:0',
        targetLegalDistrictsBefore: [],
      },
      {
        ...template,
        stepIndex: 2,
        actionKey: 'trade:Leaves:Suns',
        actionLabel: 'Trade Leaves for Suns',
        legalActionKeys: ['sell-card:25', 'trade:Leaves:Suns'],
        proposals: {
          td: {
            actionKey: 'sell-card:25',
            actionLabel: 'Sell The Borderland',
            rank: 0,
            rawScore: 4,
          },
          heuristicV2: {
            actionKey: 'trade:Leaves:Suns',
            actionLabel: 'Trade Leaves for Suns',
            rank: 0,
            rawScore: 5,
          },
        },
        playerAHandBefore: ['25', '29'],
        playerAHandAfter: ['25', '29'],
        targetLocationBefore: 'draw:0',
        targetLocationAfter: 'draw:0',
        targetLegalDistrictsBefore: [],
      },
    ],
    firstProposalDivergenceStepIndex: 1,
    firstPlayerAProposalDivergenceStepIndex: 1,
    targetEverHeldByPlayerA: false,
    targetOpportunitySeen: false,
    targetDevelopedInValuableLane: false,
  };
}

function fixtureTrace({
  rootFocusActionId,
  guide,
  score,
  winner,
  continuationStatus,
  includeTargetLegalDivergence = false,
}: {
  rootFocusActionId: StrategicForcedRolloutTraceV0['rootFocusActionId'];
  guide: StrategicForcedRolloutGuideV0;
  score: number;
  winner: FinalScore['winner'];
  continuationStatus: StrategicContinuationStatusV0;
  includeTargetLegalDivergence?: boolean;
}): StrategicForcedRolloutTraceV0 {
  const tdActionKey = 'develop-outright:6:D1';
  const heuristicActionKey = includeTargetLegalDivergence
    ? 'sell-card:6'
    : tdActionKey;
  return {
    rootFocusActionId,
    guide,
    score,
    simulatedActionSteps: 2,
    terminatedBeforeDepthLimit: true,
    finalScore: finalScore(winner),
    steps: [
      {
        stepIndex: 0,
        decisionPlayer: 'PlayerA',
        turn: 31,
        phase: 'ActionWindow',
        finalTurnsRemaining: null,
        actionKey: `${rootFocusActionId}-root`,
        actionLabel: rootFocusActionId,
        legalActionKeys: ['preserve-option-root', 'overwrite-option-root'],
        proposals: null,
        playerAHandBefore: ['14', '6', '0'],
        playerAHandAfter: ['6', '0'],
        playerAResourcesBefore: emptyResources(),
        targetLocationBefore: 'PlayerA-hand',
        targetLocationAfter: 'PlayerA-hand',
        targetLegalDistrictsBefore: [],
      },
      {
        stepIndex: 1,
        decisionPlayer: 'PlayerA',
        turn: 35,
        phase: 'ActionWindow',
        finalTurnsRemaining: 2,
        actionKey: guide === 'td' ? tdActionKey : heuristicActionKey,
        actionLabel:
          guide === 'td' ? 'Develop Author | valuable' : 'Sell Author',
        legalActionKeys: [tdActionKey, heuristicActionKey],
        proposals: {
          td: {
            actionKey: tdActionKey,
            actionLabel: 'Develop Author | valuable',
            rank: 0,
            rawScore: 2,
          },
          heuristicV2: {
            actionKey: heuristicActionKey,
            actionLabel:
              heuristicActionKey === tdActionKey
                ? 'Develop Author | valuable'
                : 'Sell Author',
            rank: 0,
            rawScore: 3,
          },
        },
        playerAHandBefore: ['6', '0'],
        playerAHandAfter: ['0'],
        playerAResourcesBefore: emptyResources(),
        targetLocationBefore: 'PlayerA-hand',
        targetLocationAfter:
          guide === 'td' ? 'PlayerA-developed:D1' : 'discard',
        targetLegalDistrictsBefore: includeTargetLegalDivergence ? ['D1'] : [],
      },
    ],
    firstProposalDivergenceStepIndex: includeTargetLegalDivergence ? 1 : null,
    firstPlayerAProposalDivergenceStepIndex: includeTargetLegalDivergence
      ? 1
      : null,
    targetEverHeldByPlayerA: true,
    targetOpportunitySeen: includeTargetLegalDivergence,
    targetDevelopedInValuableLane: continuationStatus === 'realized',
    continuationStatus,
  };
}

function finalScore(winner: FinalScore['winner']): FinalScore {
  return {
    districtPoints: { PlayerA: 2, PlayerB: 1 },
    rankTotals: { PlayerA: 20, PlayerB: 18 },
    resourceTotals: { PlayerA: 0, PlayerB: 0 },
    winner,
    decidedBy: winner === 'Draw' ? 'draw' : 'districts',
  };
}

function emptyResources(): ResourcePool {
  return {
    Moons: 0,
    Suns: 0,
    Waves: 0,
    Leaves: 0,
    Wyrms: 0,
    Knots: 0,
  };
}
