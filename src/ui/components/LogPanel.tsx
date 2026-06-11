import type { ReactNode } from 'react';

import type { GameLogEntry, PlayerId } from '../../engine/types';
import {
  formatLogSummary,
  groupLogEntriesByTurn,
  seedSummaryValue,
  SUIT_CODE_PATTERN,
  suitCodeToSuit,
  type SuitLogCode,
} from '../logPresentation';
import { playerDisplayName } from '../playerDisplay';
import { SUIT_TOKEN_BG } from './TokenComponents';

export function LogPanel({
  timelineLog,
  humanPlayerId,
}: {
  timelineLog: ReadonlyArray<GameLogEntry>;
  humanPlayerId: PlayerId;
}) {
  const recentLog = [...timelineLog].reverse();
  const recentLogGroups = groupLogEntriesByTurn(recentLog);

  return (
    <section className="panel log-panel">
      <h2>Log</h2>
      {recentLog.length === 0 ? (
        <p className="empty-note">No actions yet.</p>
      ) : (
        <ol className="log-list">
          {recentLogGroups.map((group, groupIndex) => (
            <li key={`${group.turn}-${groupIndex}`} className="log-turn-group">
              <div className="log-turn-head">
                <span className="log-turn">T{group.turn}</span>
                <span className="log-player">
                  {playerDisplayName(group.player, humanPlayerId)}
                </span>
              </div>
              <ol className="log-turn-entries">
                {group.entries.map((entry, entryIndex) => {
                  const seedValue = seedSummaryValue(entry.summary);
                  if (seedValue !== null) {
                    return (
                      <li
                        key={`${entry.turn}-${entry.phase}-${entry.summary}-${entryIndex}`}
                        className="log-turn-entry log-turn-entry-seed"
                      >
                        <div className="log-turn-head">
                          <span className="log-turn">Seed</span>
                          <span className="log-player">{seedValue}</span>
                        </div>
                      </li>
                    );
                  }
                  return (
                    <li
                      key={`${entry.turn}-${entry.phase}-${entry.summary}-${entryIndex}`}
                      className="log-turn-entry"
                    >
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
}

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
