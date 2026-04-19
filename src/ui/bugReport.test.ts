import { describe, expect, it } from 'vitest';

import { makeGameState, PLAYER_A, PLAYER_B } from '../engine/__tests__/fixtures';
import type { GameLogEntry } from '../engine/types';
import {
  getBugReportIssueUrl,
  bugReportFilename,
  buildBugReport,
  type BugReportActionEntry,
} from './bugReport';

describe('bugReport', () => {
  it('builds a reproduction report from game data without browser metadata', () => {
    const state = makeGameState({ seed: 'report-seed', turn: 4 });
    const timelineLog: GameLogEntry[] = [
      {
        turn: 4,
        player: PLAYER_A,
        phase: 'ActionWindow',
        summary: 'trade Moons for Knots',
      },
    ];
    const actionHistory: BugReportActionEntry[] = [
      {
        turn: 4,
        phase: 'ActionWindow',
        actingPlayerId: PLAYER_A,
        action: { type: 'trade', give: 'Moons', receive: 'Knots' },
      },
    ];

    const report = buildBugReport({
      state,
      timelineLog,
      actionHistory,
      humanPlayerId: PLAYER_A,
      botPlayerId: PLAYER_B,
      botProfileId: 'heuristic',
      animationsEnabled: true,
      error: 'engine failed',
    });

    expect(report.schemaVersion).toBe(1);
    expect(report.game.state.seed).toBe('report-seed');
    expect(report.game.timelineLog).toEqual(timelineLog);
    expect(report.game.actionHistory).toEqual(actionHistory);
    expect(report.game.botProfileId).toBe('heuristic');
    expect(report.game.animationsEnabled).toBe(true);
    expect(report.error).toBe('engine failed');
    expect(bugReportFilename()).toMatch(/^magnate-log-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.json$/);

    const serialized = JSON.stringify(report);
    expect(serialized).not.toContain('userAgent');
    expect(serialized).not.toContain('timezone');
    expect(serialized).not.toContain('screen');
    expect(serialized).not.toContain('localStorage');
    expect(serialized).not.toContain('location');
    expect(serialized).not.toContain('href');
  });

  it('points at the public bug report issue form', () => {
    const url = getBugReportIssueUrl();
    expect(url).toContain('github.com/Dylan-Cairns/magnate/issues/new');
    expect(url).toContain('template=bug_report.yml');
    expect(url).toContain('Automatic%20bug%20report%20');
  });
});
