import type { BotProfileId } from '../policies/catalog';
import type {
  GameAction,
  GameLogEntry,
  GamePhase,
  GameState,
  PlayerId,
} from '../engine/types';

export const BUG_REPORT_ISSUE_URL =
  'https://github.com/Dylan-Cairns/magnate/issues/new?template=bug_report.yml&title=Bug%20report';

export interface BugReportActionEntry {
  turn: number;
  phase: GamePhase;
  actingPlayerId: PlayerId;
  action: GameAction;
}

export interface BugReport {
  schemaVersion: 1;
  app: {
    name: 'magnate';
    version: '1.0.0';
  };
  game: {
    humanPlayerId: PlayerId;
    botPlayerId: PlayerId;
    botProfileId: BotProfileId;
    animationsEnabled: boolean;
    state: GameState;
    timelineLog: readonly GameLogEntry[];
    actionHistory: readonly BugReportActionEntry[];
  };
  error: string | null;
}

export function buildBugReport({
  state,
  timelineLog,
  actionHistory,
  humanPlayerId,
  botPlayerId,
  botProfileId,
  animationsEnabled,
  error,
}: {
  state: GameState;
  timelineLog: readonly GameLogEntry[];
  actionHistory: readonly BugReportActionEntry[];
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
  botProfileId: BotProfileId;
  animationsEnabled: boolean;
  error: string | null;
}): BugReport {
  return {
    schemaVersion: 1,
    app: {
      name: 'magnate',
      version: '1.0.0',
    },
    game: {
      humanPlayerId,
      botPlayerId,
      botProfileId,
      animationsEnabled,
      state,
      timelineLog: [...timelineLog],
      actionHistory: actionHistory.map((entry) => ({
        ...entry,
        action: { ...entry.action },
      })),
    },
    error,
  };
}

export function bugReportFilename(report: BugReport): string {
  return `magnate-bug-report-turn-${String(report.game.state.turn)}.json`;
}

export function downloadBugReport(report: BugReport): void {
  const blob = new Blob([JSON.stringify(report, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = bugReportFilename(report);
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
