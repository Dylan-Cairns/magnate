import { memo, type ReactNode } from 'react';

import type { GameLogEntry, GameState, PlayerId } from '../../engine/types';
import { visibleLogEntriesForPlayer } from '../../engine/view';
import {
  formatLogSummary,
  groupLogEntriesByTurn,
  metaSummaryLabel,
  SUIT_CODE_PATTERN,
  suitCodeToSuit,
  type SuitLogCode,
} from '../logPresentation';
import { playerDisplayName } from '../playerDisplay';
import { SUIT_TOKEN_BG } from './TokenComponents';

export const LogPanel = memo(function LogPanel({
  timelineLog,
  humanPlayerId,
  state,
  animationsEnabled = true,
}: {
  timelineLog: ReadonlyArray<GameLogEntry>;
  humanPlayerId: PlayerId;
  state: Pick<
    GameState,
    'turn' | 'pendingIncomeChoices' | 'submittedIncomeChoices'
  >;
  animationsEnabled?: boolean;
}) {
  // Privacy follows canonical submissions even when presentation is still
  // showing the previous turn. Keep the full timeline for export and reveal.
  const recentLog = [
    ...visibleLogEntriesForPlayer(timelineLog, state, humanPlayerId),
  ].reverse();
  const recentLogGroups = groupLogEntriesByTurn(recentLog);

  // `timelineLog` is append-only and filtering preserves object identity, so
  // the chronological index is a stable key. Index-based keys would shift on
  // every prepend and remount the whole list each turn, replaying the entrance
  // animation on existing rows.
  const entryIds = new Map<GameLogEntry, number>();
  timelineLog.forEach((entry, index) => {
    entryIds.set(entry, index);
  });

  return (
    <section className="panel log-panel">
      <h2>Log</h2>
      {recentLog.length === 0 ? (
        <p className="empty-note empty-note-block">No actions yet.</p>
      ) : (
        <ol className={animationsEnabled ? 'log-list is-animated' : 'log-list'}>
          {recentLogGroups.map((group, groupIndex) => (
            <li
              key={`group-${
                entryIds.get(group.entries[group.entries.length - 1]) ??
                `${group.turn}-${groupIndex}`
              }`}
              className="log-turn-group"
            >
              <div className="log-turn-head">
                <span className="log-head-reveal">
                  <span className="log-turn">T{group.turn}</span>
                  <span className="log-player">
                    {playerDisplayName(group.player, humanPlayerId)}
                  </span>
                </span>
              </div>
              <ol className="log-turn-entries">
                {group.entries.map((entry, entryIndex) => {
                  const metaValue = metaSummaryLabel(entry.summary);
                  const entryKey =
                    entryIds.get(entry) ??
                    `${entry.turn}-${entry.phase}-${entry.summary}-${entryIndex}`;
                  if (metaValue !== null) {
                    return (
                      <li
                        key={`entry-${entryKey}`}
                        className="log-turn-entry log-turn-entry-seed"
                      >
                        <div className="log-turn-head">
                          <span className="log-player">{metaValue}</span>
                        </div>
                      </li>
                    );
                  }
                  return (
                    <li key={`entry-${entryKey}`} className="log-turn-entry">
                      <span className="log-summary">
                        <LogSummary
                          summary={
                            entry.player !== group.player
                              ? `[${playerDisplayName(entry.player, humanPlayerId)}] ${entry.summary}`
                              : entry.summary
                          }
                        />
                      </span>
                    </li>
                  );
                })}
              </ol>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
});

function LogSummary({ summary }: { summary: string }) {
  const text = formatLogSummary(summary);
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(SUIT_CODE_PATTERN)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      nodes.push(text.slice(cursor, index));
    }

    const suitCode = match[0] as SuitLogCode;
    const suit = suitCodeToSuit(suitCode);
    nodes.push(
      <span
        key={`log-suit-${index}-${suitCode}`}
        className="log-suit-code"
        style={{ color: SUIT_TOKEN_BG[suit] }}
      >
        {suitCode}
      </span>
    );
    cursor = index + suitCode.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? <>{nodes}</> : text;
}
