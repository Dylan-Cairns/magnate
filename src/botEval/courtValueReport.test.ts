import { describe, expect, it } from 'vitest';

import { createHeadToHeadArtifact } from './artifacts';
import {
  buildCourtValueReport,
  evaluateCourtValueGates,
  renderCourtValueReportMarkdown,
  type CourtValueUsageSummary,
} from './courtValueReport';
import { runHeadToHead } from './matchup';
import type { HeadToHeadConfig } from './types';

describe('court value report', () => {
  it('replays a head-to-head artifact into per-bot court usage and diagnostics', async () => {
    const artifact = await buildExtendedArtifact();
    const report = buildCourtValueReport(artifact, 'matchup.json');

    expect(report.ruleset).toBe('extended');
    expect(report.totals.games).toBe(2);
    expect(report.totals.pairs).toBe(1);
    expect(report.perPair).toHaveLength(1);
    expect(report.candidate.courtValueScale).toBe('1');
    expect(report.opponent.courtValueScale).toBe('0');

    const candidateUsage = report.usageByBotId['court-report-candidate'];
    const opponentUsage = report.usageByBotId['court-report-opponent'];
    expect(candidateUsage.decisions).toBeGreaterThan(0);
    expect(opponentUsage.decisions).toBeGreaterThan(0);
    expect(
      report.gates.find((gate) => gate.id === 'replay-integrity')?.status
    ).toBe('pass');
    expect(
      report.gates.some((gate) => gate.id === 'extended-court-utilization')
    ).toBe(true);
    expect(
      report.gates.some((gate) => gate.id === 'extended-court-follow-through')
    ).toBe(true);

    const candidateDiagnostics =
      report.courtDecisionsByBotId['court-report-candidate'];
    expect(candidateDiagnostics.decisions).toBeGreaterThan(0);
    expect(
      candidateDiagnostics.decisionsWithCourtOption
    ).toBeLessThanOrEqual(candidateDiagnostics.decisions);
    expect(report.courtDecisionsByBotId['court-report-opponent']).toBeDefined();

    const markdown = renderCourtValueReportMarkdown(report);
    expect(markdown).toContain('# Court Value Benchmark Report');
    expect(markdown).toContain('court-report-candidate');
    expect(markdown).toContain('Court decision diagnostics');
  });

  it('gates standard non-inferiority and deed-buy behavior', () => {
    const gates = evaluateCourtValueGates({
      ruleset: 'standard',
      paired: {
        pairs: 60,
        candidateWinsMorePairs: 20,
        opponentWinsMorePairs: 18,
        tiedPairs: 22,
        meanWinMargin: 0.0167,
        meanWinMarginCi95: { low: -0.05, high: 0.0833 },
        mcnemarTwoSidedP: 0.87,
      },
      candidateUsage: usageFixture({ buys: 12, decisions: 200 }),
      opponentUsage: usageFixture({ buys: 13, decisions: 200 }),
    });

    expect(gates.map((gate) => gate.id)).toEqual([
      'standard-paired-noninferiority',
      'standard-deed-buy-behavior',
    ]);
    expect(gates[0].status).toBe('pass');
    expect(gates[1].status).toBe('pass');
  });

  it('fails standard non-inferiority when the paired interval breaches the margin', () => {
    const gates = evaluateCourtValueGates({
      ruleset: 'standard',
      paired: {
        pairs: 60,
        candidateWinsMorePairs: 10,
        opponentWinsMorePairs: 25,
        tiedPairs: 25,
        meanWinMargin: -0.125,
        meanWinMarginCi95: { low: -0.2, high: -0.05 },
        mcnemarTwoSidedP: 0.017,
      },
      candidateUsage: usageFixture({ buys: 20, decisions: 200 }),
      opponentUsage: usageFixture({ buys: 10, decisions: 200 }),
    });

    expect(gates[0].status).toBe('fail');
    expect(gates[1].status).toBe('fail');
  });

  it('observes standard runs below the minimum pair count', () => {
    const gates = evaluateCourtValueGates({
      ruleset: 'standard',
      paired: {
        pairs: 10,
        candidateWinsMorePairs: 3,
        opponentWinsMorePairs: 2,
        tiedPairs: 5,
        meanWinMargin: 0.05,
        meanWinMarginCi95: { low: -0.1, high: 0.2 },
        mcnemarTwoSidedP: 1,
      },
      candidateUsage: usageFixture({}),
      opponentUsage: usageFixture({}),
    });

    expect(gates[0].status).toBe('observe');
  });

  it('gates extended improvement, utilization, follow-through, and dumping', () => {
    const passing = evaluateCourtValueGates({
      ruleset: 'extended',
      paired: {
        pairs: 60,
        candidateWinsMorePairs: 25,
        opponentWinsMorePairs: 15,
        tiedPairs: 20,
        meanWinMargin: 0.0833,
        meanWinMarginCi95: { low: 0.01, high: 0.1566 },
        mcnemarTwoSidedP: 0.15,
      },
      candidateUsage: usageFixture({
        courtBuys: 4,
        courtOutrights: 1,
        courtCompletions: 3,
        courtSellsWithLegalCourtBuild: 0,
      }),
      opponentUsage: usageFixture({}),
    });

    expect(passing.map((gate) => gate.status)).toEqual([
      'pass',
      'pass',
      'pass',
      'pass',
    ]);

    const failing = evaluateCourtValueGates({
      ruleset: 'extended',
      paired: {
        pairs: 60,
        candidateWinsMorePairs: 12,
        opponentWinsMorePairs: 20,
        tiedPairs: 28,
        meanWinMargin: -0.0667,
        meanWinMarginCi95: { low: -0.15, high: 0.0166 },
        mcnemarTwoSidedP: 0.2,
      },
      candidateUsage: usageFixture({ courtSellsWithLegalCourtBuild: 5 }),
      opponentUsage: usageFixture({}),
    });

    expect(failing.map((gate) => gate.status)).toEqual([
      'fail',
      'fail',
      'fail',
      'fail',
    ]);
  });

  it('observes weak court follow-through below the pass ratio', () => {
    const gates = evaluateCourtValueGates({
      ruleset: 'extended',
      paired: {
        pairs: 60,
        candidateWinsMorePairs: 20,
        opponentWinsMorePairs: 20,
        tiedPairs: 20,
        meanWinMargin: 0,
        meanWinMarginCi95: { low: -0.08, high: 0.08 },
        mcnemarTwoSidedP: 1,
      },
      candidateUsage: usageFixture({
        courtBuys: 8,
        courtOutrights: 0,
        courtCompletions: 3,
        courtSellsWithLegalCourtBuild: 0,
      }),
      opponentUsage: usageFixture({}),
    });

    const followThrough = gates.find(
      (gate) => gate.id === 'extended-court-follow-through'
    );
    expect(followThrough?.status).toBe('observe');
  });
});

function usageFixture(
  overrides: Partial<CourtValueUsageSummary>
): CourtValueUsageSummary {
  const decisions = overrides.decisions ?? 150;
  const buys = overrides.buys ?? 5;
  return {
    botId: overrides.botId ?? 'bot',
    decisions,
    buys,
    buyRate: decisions > 0 ? buys / decisions : 0,
    sells: overrides.sells ?? 2,
    courtBuys: overrides.courtBuys ?? 0,
    courtOutrights: overrides.courtOutrights ?? 0,
    courtDeedDevelops: overrides.courtDeedDevelops ?? 0,
    courtCompletions: overrides.courtCompletions ?? 0,
    courtSells: overrides.courtSells ?? 0,
    courtSellsWithLegalCourtBuild:
      overrides.courtSellsWithLegalCourtBuild ?? 0,
  };
}

async function buildExtendedArtifact() {
  const config: HeadToHeadConfig = {
    schemaVersion: 1,
    runLabel: 'court-report-test',
    seedPrefix: 'court-report-test',
    gamesPerSide: 1,
    candidate: {
      id: 'court-report-candidate',
      kind: 'search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
        heuristic: 'v2',
        courtValueScale: 1,
      },
    },
    opponent: {
      id: 'court-report-opponent',
      kind: 'search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
        heuristic: 'v2',
        courtValueScale: 0,
      },
    },
    ruleset: 'extended',
  };
  const run = await runHeadToHead(config, { now: () => 0 });
  return createHeadToHeadArtifact(run, {
    generatedAtUtc: '2026-09-15T00:00:00.000Z',
    git: { commit: 'test-commit', dirty: false },
    nodeVersion: 'test-node',
  });
}
