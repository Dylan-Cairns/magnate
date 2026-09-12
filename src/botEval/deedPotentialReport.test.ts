import { describe, expect, it } from 'vitest';

import { createHeadToHeadArtifact } from './artifacts';
import {
  buildDeedPotentialReport,
  evaluateDeedGates,
  renderDeedPotentialReportMarkdown,
  type DeedUsageSummary,
} from './deedPotentialReport';
import { runHeadToHead } from './matchup';
import type { HeadToHeadConfig } from './types';

describe('deed potential report', () => {
  it('replays a head-to-head artifact into per-bot deed and court usage', async () => {
    const artifact = await buildExtendedArtifact();
    const report = buildDeedPotentialReport(artifact, 'matchup.json');

    expect(report.ruleset).toBe('extended');
    expect(report.totals.games).toBe(2);
    expect(report.totals.pairs).toBe(1);
    expect(report.perPair).toHaveLength(1);
    expect(report.candidate.deedPotentialBase).toBe('0.2');
    expect(report.opponent.deedPotentialBase).toBe('0');

    const candidateUsage = report.usageByBotId['deed-report-candidate'];
    const opponentUsage = report.usageByBotId['deed-report-opponent'];
    expect(candidateUsage.decisions).toBeGreaterThan(0);
    expect(opponentUsage.decisions).toBeGreaterThan(0);
    expect(candidateUsage.buys).toBeGreaterThanOrEqual(0);
    expect(
      report.gates.find((gate) => gate.id === 'replay-integrity')?.status
    ).toBe('pass');
    expect(
      report.gates.some((gate) => gate.id === 'extended-court-utilization')
    ).toBe(true);

    const markdown = renderDeedPotentialReportMarkdown(report);
    expect(markdown).toContain('# Deed Potential Benchmark Report');
    expect(markdown).toContain('deed-report-candidate');
  });

  it('gates standard non-inferiority and deed-buy behavior', () => {
    const gates = evaluateDeedGates({
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
    const gates = evaluateDeedGates({
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
    const gates = evaluateDeedGates({
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

  it('gates extended improvement, court utilization, and court dumping', () => {
    const passing = evaluateDeedGates({
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
    ]);

    const failing = evaluateDeedGates({
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
    ]);
  });
});

function usageFixture(
  overrides: Partial<DeedUsageSummary>
): DeedUsageSummary {
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
    runLabel: 'deed-report-test',
    seedPrefix: 'deed-report-test',
    gamesPerSide: 1,
    candidate: {
      id: 'deed-report-candidate',
      kind: 'search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
        heuristic: 'v2',
        deedPotentialBase: 0.2,
      },
    },
    opponent: {
      id: 'deed-report-opponent',
      kind: 'search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
        heuristic: 'v2',
        deedPotentialBase: 0,
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
