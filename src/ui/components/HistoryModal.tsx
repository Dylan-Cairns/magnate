import { useEffect, useMemo, useState } from 'react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table';
import {
  ACHIEVEMENT_META,
  getAchievements,
  getGames,
  getStats,
  type AchievementWithGame,
} from '../../db/gameHistory';
import type { AchievementKey, GameRecord } from '../../db/db';
import type { Stats } from '../../db/historyLogic';

const DATE_FMT = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
});

function formatDate(ts: number) {
  return DATE_FMT.format(new Date(ts));
}

function formatWinner(winner: GameRecord['winner']) {
  if (winner === 'player') return 'Win';
  if (winner === 'bot') return 'Loss';
  return 'Draw';
}

function formatDecidedBy(decidedBy: GameRecord['decidedBy']) {
  if (decidedBy === 'districts') return 'Districts';
  if (decidedBy === 'rank-total') return 'Properties';
  if (decidedBy === 'resources') return 'Resources';
  return '—';
}

const columnHelper = createColumnHelper<GameRecord>();

const COLUMNS = [
  columnHelper.accessor('timestamp', {
    header: 'Date',
    cell: (info) => formatDate(info.getValue()),
    sortDescFirst: true,
  }),
  columnHelper.accessor('winner', {
    header: 'Result',
    cell: (info) => (
      <span className={`result-badge result-badge--${info.getValue()}`}>
        {formatWinner(info.getValue())}
      </span>
    ),
  }),
  columnHelper.accessor('decidedBy', {
    header: 'Decided By',
    cell: (info) => formatDecidedBy(info.getValue()),
  }),
  columnHelper.accessor('botLabel', {
    header: 'Bot',
  }),
  columnHelper.accessor('playerDistricts', {
    header: 'Districts',
    enableSorting: false,
    cell: (info) => `P${info.getValue()}–B${info.row.original.botDistricts}`,
  }),
  columnHelper.accessor('playerRankTotal', {
    header: 'Properties',
    enableSorting: false,
    cell: (info) => `P${info.getValue()}–B${info.row.original.botRankTotal}`,
  }),
  columnHelper.accessor('playerResources', {
    header: 'Resources',
    enableSorting: false,
    cell: (info) => `P${info.getValue()}–B${info.row.original.botResources}`,
  }),
];

const ACHIEVEMENT_ORDER: AchievementKey[] = [
  'shutout',
  'hard_mode',
  'hard_mode_streak',
  'tactician',
];

export function HistoryModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [games, setGames] = useState<GameRecord[]>([]);
  const [achievementData, setAchievementData] = useState<AchievementWithGame[]>([]);
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'timestamp', desc: true },
  ]);

  useEffect(() => {
    if (!open) return;
    void Promise.all([getStats(), getGames(), getAchievements()]).then(
      ([s, g, a]) => {
        setStats(s);
        setGames(g);
        setAchievementData(a);
      }
    );
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const unlockedByKey = useMemo(
    () =>
      new Map(
        achievementData.map(({ achievement, game }) => [
          achievement.achievementKey,
          { achievement, game },
        ])
      ),
    [achievementData]
  );

  const table = useReactTable({
    data: games,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (!open) return null;

  return (
    <div
      className="history-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="history-modal-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="panel history-modal">
        <div className="history-modal-header">
          <h2 id="history-modal-title">Game History</h2>
          <button
            type="button"
            className="history-modal-close"
            aria-label="Close history"
            onClick={onClose}
          >
            <CloseIcon />
          </button>
        </div>

        <div className="history-stats-row">
          <div className="history-stat">
            <span className="history-stat-value">{stats?.gamesPlayed ?? '—'}</span>
            <span className="history-stat-label">Games Played</span>
          </div>
          <div className="history-stat-divider" />
          <div className="history-stat">
            <span className="history-stat-value">{stats?.currentStreak ?? '—'}</span>
            <span className="history-stat-label">Current Streak</span>
          </div>
          <div className="history-stat-divider" />
          <div className="history-stat">
            <span className="history-stat-value">{stats?.longestStreak ?? '—'}</span>
            <span className="history-stat-label">Longest Streak</span>
          </div>
        </div>

        <div className="history-body">
          <div className="history-grid-wrap">
            {games.length === 0 ? (
              <p className="history-empty">No games recorded yet. Finish a game to see it here.</p>
            ) : (
              <div className="history-table-scroll">
                <table className="history-table">
                  <thead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <tr key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <th
                            key={header.id}
                            className={header.column.getCanSort() ? 'is-sortable' : ''}
                            onClick={header.column.getToggleSortingHandler()}
                            aria-sort={
                              header.column.getIsSorted() === 'asc'
                                ? 'ascending'
                                : header.column.getIsSorted() === 'desc'
                                  ? 'descending'
                                  : 'none'
                            }
                          >
                            {flexRender(header.column.columnDef.header, header.getContext())}
                            {header.column.getCanSort() && (
                              <span className="sort-indicator" aria-hidden="true">
                                {header.column.getIsSorted() === 'asc'
                                  ? ' ↑'
                                  : header.column.getIsSorted() === 'desc'
                                    ? ' ↓'
                                    : ' ↕'}
                              </span>
                            )}
                          </th>
                        ))}
                      </tr>
                    ))}
                  </thead>
                  <tbody>
                    {table.getRowModel().rows.map((row) => (
                      <tr
                        key={row.id}
                        className={`history-row history-row--${row.original.winner}`}
                      >
                        {row.getVisibleCells().map((cell) => (
                          <td key={cell.id}>
                            {flexRender(cell.column.columnDef.cell, cell.getContext())}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="history-achievements">
            <h3>Achievements</h3>
            <ul className="achievement-list">
              {ACHIEVEMENT_ORDER.map((key) => {
                const entry = unlockedByKey.get(key);
                const meta = ACHIEVEMENT_META[key];
                const unlocked = !!entry;
                const tooltipTitle =
                  unlocked && entry.achievement
                    ? `Unlocked ${formatDate(entry.achievement.unlockedAt)}`
                    : 'Not yet unlocked';
                return (
                  <li
                    key={key}
                    className={`achievement-item${unlocked ? ' is-unlocked' : ''}`}
                    title={tooltipTitle}
                  >
                    <span className="achievement-icon" aria-hidden="true">
                      {unlocked ? <CheckIcon /> : <LockIcon />}
                    </span>
                    <span className="achievement-text">
                      <span className="achievement-name">{meta.label}</span>
                      <span className="achievement-desc">{meta.description}</span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="history-close-icon">
      <path d="M18 6 6 18" />
      <path d="M6 6l12 12" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="achievement-icon-svg">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="achievement-icon-svg">
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}
