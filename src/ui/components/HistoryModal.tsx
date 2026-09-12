import { useEffect, useState } from 'react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table';
import { getGames } from '../../db/gameHistory';
import type { GameRecord } from '../../db/db';

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
  if (winner === 'player') return 'Victory';
  if (winner === 'bot') return 'Defeat';
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
    sortDescFirst: true,
    cell: (info) => formatDate(info.getValue()),
  }),
  columnHelper.accessor('winner', {
    header: 'Result',
    cell: (info) => (
      <span className={`history-result history-result--${info.getValue()}`}>
        {formatWinner(info.getValue())}
      </span>
    ),
  }),
  columnHelper.accessor('decidedBy', {
    header: 'Decided By',
    cell: (info) => formatDecidedBy(info.getValue()),
  }),
  columnHelper.accessor('botLabel', {
    header: 'Opponent',
  }),
  columnHelper.accessor('playerDistricts', {
    header: 'Districts',
    enableSorting: false,
    cell: (info) => `${info.getValue()}–${info.row.original.botDistricts}`,
  }),
  columnHelper.accessor('playerRankTotal', {
    header: 'Properties',
    enableSorting: false,
    cell: (info) => `${info.getValue()}–${info.row.original.botRankTotal}`,
  }),
  columnHelper.accessor('playerResources', {
    header: 'Resources',
    enableSorting: false,
    cell: (info) => `${info.getValue()}–${info.row.original.botResources}`,
  }),
];

export function HistoryModal({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [games, setGames] = useState<GameRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([
    { id: 'timestamp', desc: true },
  ]);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(false);
    void getGames().then(
      (records) => {
        if (cancelled) return;
        setGames(records);
        setLoading(false);
      },
      () => {
        if (cancelled) return;
        setError(true);
        setLoading(false);
      }
    );
    return () => {
      cancelled = true;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const table = useReactTable({
    data: games,
    columns: COLUMNS,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const victories = games.filter((game) => game.winner === 'player').length;
  const defeats = games.filter((game) => game.winner === 'bot').length;
  const draws = games.length - victories - defeats;

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
          <h2 id="history-modal-title">History</h2>
        </div>

        {!loading && !error && games.length > 0 && (
          <div className="history-summary-row">
            <p className="history-summary">
              {games.length} {games.length === 1 ? 'game' : 'games'} played
              {' · '}
              {victories} {victories === 1 ? 'victory' : 'victories'}
              {' · '}
              {defeats} {defeats === 1 ? 'defeat' : 'defeats'}
              {draws > 0 && (
                <>
                  {' '}
                  · {draws} {draws === 1 ? 'draw' : 'draws'}
                </>
              )}
            </p>
            <p className="history-score-key" id="history-score-key">
              Scores shown as you–opponent.
            </p>
          </div>
        )}

        <div className="history-body">
          <div className="history-grid-wrap">
            {loading ? (
              <p className="history-empty" role="status">
                Loading history…
              </p>
            ) : error ? (
              <p className="history-empty" role="alert">
                History could not be loaded. Close and reopen to retry.
              </p>
            ) : games.length === 0 ? (
              <p className="history-empty">
                No games recorded yet. Finish a game to see it here.
              </p>
            ) : (
              <div
                className="history-table-scroll"
                role="region"
                aria-label="Game history"
                tabIndex={0}
              >
                <table
                  className="history-table"
                  aria-describedby="history-score-key"
                >
                  <thead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <tr key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <th
                            key={header.id}
                            scope="col"
                            aria-sort={
                              header.column.getIsSorted() === 'asc'
                                ? 'ascending'
                                : header.column.getIsSorted() === 'desc'
                                  ? 'descending'
                                  : undefined
                            }
                          >
                            {header.column.getCanSort() ? (
                              <button
                                type="button"
                                className="history-sort-button"
                                onClick={header.column.getToggleSortingHandler()}
                              >
                                {flexRender(
                                  header.column.columnDef.header,
                                  header.getContext()
                                )}
                                <span
                                  className="history-sort-indicator"
                                  aria-hidden="true"
                                >
                                  {header.column.getIsSorted() === 'asc'
                                    ? ' ↑'
                                    : header.column.getIsSorted() === 'desc'
                                      ? ' ↓'
                                      : ' ↕'}
                                </span>
                              </button>
                            ) : (
                              flexRender(
                                header.column.columnDef.header,
                                header.getContext()
                              )
                            )}
                          </th>
                        ))}
                      </tr>
                    ))}
                  </thead>
                  <tbody>
                    {table.getRowModel().rows.map((row) => (
                      <tr key={row.id}>
                        {row.getVisibleCells().map((cell) => (
                          <td key={cell.id}>
                            {flexRender(
                              cell.column.columnDef.cell,
                              cell.getContext()
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
