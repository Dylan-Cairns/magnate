import { describe, expect, it } from 'vitest';

import { COURT_CARDS } from '../engine/cards';
import { selectHeuristicV2Action } from '../policies/heuristicScorerV2';
import type { ActionPolicy } from '../policies/types';
import { createHeadToHeadArtifact } from './artifacts';
import { createHeadToHeadCheckpoint } from './checkpointArtifacts';
import {
  buildCourtDecisionEvalReport,
  courtDecisionEvalInputFromCheckpoint,
  isTermCourtAction,
  summarizeCourtDecisionOutcomes,
} from './courtDecisionEval';
import { runHeadToHead } from './matchup';
import type { HeadToHeadArtifact, HeadToHeadConfig } from './types';

describe('court decision evaluation', () => {
  it('summarizes paired position outcomes', () => {
    const summary = summarizeCourtDecisionOutcomes(
      [
        {
          marginDelta: 2,
          courtMargin: 5,
          nonCourtMargin: 3,
          courtWinRate: 1,
          nonCourtWinRate: 0.5,
          rankTotalMarginDelta: 4,
        },
        {
          marginDelta: -1,
          courtMargin: 2,
          nonCourtMargin: 3,
          courtWinRate: 0.5,
          nonCourtWinRate: 0.75,
          rankTotalMarginDelta: -2,
        },
        {
          marginDelta: 0,
          courtMargin: 4,
          nonCourtMargin: 4,
          courtWinRate: 0.5,
          nonCourtWinRate: 0.5,
          rankTotalMarginDelta: 0,
        },
      ],
      20
    );

    expect(summary.positions).toBe(3);
    expect(summary.worldsPerPosition).toBe(20);
    expect(summary.meanMarginDelta).toBeCloseTo(1 / 3);
    expect(summary.courtBetterPositions).toBe(1);
    expect(summary.nonCourtBetterPositions).toBe(1);
    expect(summary.tiedPositions).toBe(1);
    expect(summary.marginDeltaCi95.low).toBeLessThan(summary.meanMarginDelta);
    expect(summary.marginDeltaCi95.high).toBeGreaterThan(
      summary.meanMarginDelta
    );
    expect(summary.meanCourtMargin).toBeCloseTo((5 + 2 + 4) / 3);
    expect(summary.meanNonCourtMargin).toBeCloseTo((3 + 3 + 4) / 3);
    expect(summary.meanCourtWinRate).toBeCloseTo((1 + 0.5 + 0.5) / 3);
    expect(summary.meanRankTotalMarginDelta).toBeCloseTo((4 - 2 + 0) / 3);
  });

  it('returns an empty summary for no outcomes', () => {
    const summary = summarizeCourtDecisionOutcomes([], 10);

    expect(summary.positions).toBe(0);
    expect(summary.meanMarginDelta).toBe(0);
    expect(summary.marginDeltaCi95).toEqual({ low: 0, high: 0 });
  });

  it('identifies term-owned court actions only', () => {
    const courtCardId = COURT_CARDS[0].id;

    expect(
      isTermCourtAction({
        type: 'buy-deed',
        cardId: courtCardId,
        districtId: 'D1',
      })
    ).toBe(true);
    expect(
      isTermCourtAction({
        type: 'develop-deed',
        districtId: 'D1',
        cardId: courtCardId,
        tokens: {},
      })
    ).toBe(true);
    expect(
      isTermCourtAction({
        type: 'develop-outright',
        cardId: courtCardId,
        districtId: 'D1',
        payment: {},
      })
    ).toBe(false);
    expect(
      isTermCourtAction({
        type: 'buy-deed',
        cardId: 'ace-suns',
        districtId: 'D1',
      })
    ).toBe(false);
    expect(isTermCourtAction({ type: 'end-turn' })).toBe(false);
  });

  it('evaluates recorded Court decisions against the best non-Court action', async () => {
    const artifact = await testArtifact();
    const report = buildCourtDecisionEvalReport(artifact, {
      sourceKind: 'artifact',
      sourcePath: 'court-decision-eval-test.json',
      worlds: 1,
      maxPositions: 3,
      seed: 'court-decision-eval-test',
      generatedAtUtc: '2026-09-22T00:00:00.000Z',
      git: { commit: 'test-commit', dirty: false },
    });

    expect(report.positions.collected).toBeGreaterThan(0);
    expect(report.positions.evaluated).toBe(
      Math.min(report.positions.collected, 3)
    );
    expect(report.positions.evaluated).toBeLessThanOrEqual(3);
    expect(report.positions.courtRecommended).toBe(report.recommended.positions);
    expect(report.positions.courtRejected).toBe(report.rejected.positions);
    expect(report.overall.positions).toBe(report.positions.evaluated);
    expect(
      report.recommended.positions + report.rejected.positions
    ).toBe(report.positions.evaluated);
    expect(report.overall.marginDeltaCi95.low).toBeLessThanOrEqual(
      report.overall.meanMarginDelta
    );
    expect(report.overall.marginDeltaCi95.high).toBeGreaterThanOrEqual(
      report.overall.meanMarginDelta
    );
    for (const bucket of report.byActionType) {
      expect(['buy-deed', 'develop-deed']).toContain(bucket.key);
    }
    expect(
      report.byFeasibilityRecommended.reduce(
        (sum, bucket) => sum + bucket.summary.positions,
        0
      )
    ).toBe(report.recommended.positions);
    expect(
      report.byFeasibilityRejected.reduce(
        (sum, bucket) => sum + bucket.summary.positions,
        0
      )
    ).toBe(report.rejected.positions);
    expect(
      report.byActionTypeRecommended.reduce(
        (sum, bucket) => sum + bucket.summary.positions,
        0
      )
    ).toBe(report.recommended.positions);
    expect(
      report.byActionTypeRejected.reduce(
        (sum, bucket) => sum + bucket.summary.positions,
        0
      )
    ).toBe(report.rejected.positions);

    const repeated = buildCourtDecisionEvalReport(artifact, {
      sourceKind: 'artifact',
      sourcePath: 'court-decision-eval-test.json',
      worlds: 1,
      maxPositions: 3,
      seed: 'court-decision-eval-test',
      generatedAtUtc: '2026-09-22T00:00:00.000Z',
      git: { commit: 'test-commit', dirty: false },
    });
    expect(repeated).toEqual(report);
  }, 60_000);

  it('builds an evaluation input from an interrupted run checkpoint', async () => {
    const artifact = await testArtifact();
    const first = artifact.games[0];
    const second = artifact.games[1] ?? artifact.games[0];
    const checkpoint = createHeadToHeadCheckpoint(
      artifact.config,
      [{ pairIndex: 0, games: [first, second] }],
      1_000
    );

    const input = courtDecisionEvalInputFromCheckpoint(checkpoint);

    expect(input.config).toEqual(artifact.config);
    expect(input.games).toEqual([first, second]);
  }, 60_000);

  it('rejects invalid options', async () => {
    const artifact = await testArtifact();

    expect(() =>
      buildCourtDecisionEvalReport(artifact, { worlds: 0 })
    ).toThrow('Court decision eval worlds must be a positive integer.');
    expect(() =>
      buildCourtDecisionEvalReport(artifact, { maxPositions: -1 })
    ).toThrow('Court decision eval maxPositions must be a positive integer.');
    expect(() =>
      buildCourtDecisionEvalReport(artifact, { continuationScale: -1 })
    ).toThrow('Court decision eval continuationScale must be a finite number >= 0.');
    expect(() =>
      buildCourtDecisionEvalReport(artifact, { seed: '  ' })
    ).toThrow('Court decision eval seed must be a non-empty string.');
  }, 60_000);
});

const courtLovingPolicy: ActionPolicy = {
  selectAction(context) {
    return (
      context.legalActions.find(isTermCourtAction) ??
      selectHeuristicV2Action(context)
    );
  },
};

let cachedArtifact: Promise<HeadToHeadArtifact> | undefined;

function testArtifact(): Promise<HeadToHeadArtifact> {
  cachedArtifact ??= buildTestArtifact();
  return cachedArtifact;
}

async function buildTestArtifact(): Promise<HeadToHeadArtifact> {
  const config: HeadToHeadConfig = {
    schemaVersion: 1,
    runLabel: 'court-decision-eval-test',
    seedPrefix: 'court-decision-eval-test',
    gamesPerSide: 1,
    candidate: { id: 'court-loving-a', kind: 'heuristic' },
    opponent: { id: 'court-loving-b', kind: 'heuristic' },
    ruleset: 'extended',
  };
  const run = await runHeadToHead(config, {
    createPolicy: () => courtLovingPolicy,
  });
  return createHeadToHeadArtifact(run, {
    generatedAtUtc: '2026-09-22T00:00:00.000Z',
    git: { commit: 'test-commit', dirty: false },
    nodeVersion: 'test-node',
  });
}
