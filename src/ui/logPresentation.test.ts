import { describe, expect, it } from 'vitest';

import type { GameLogEntry } from '../engine/types';
import {
  formatLogSummary,
  groupLogEntriesByTurn,
  seedSummaryValue,
  suitCodeToSuit,
} from './logPresentation';

describe('groupLogEntriesByTurn', () => {
  it('groups reverse-chronological entries and uses the oldest turn-owner player for the header', () => {
    const entries: GameLogEntry[] = [
      makeLogEntry(3, 'PlayerA', 'income choice 6:Moons'),
      makeLogEntry(3, 'PlayerB', 'Income PlayerB none'),
      makeLogEntry(3, 'PlayerB', 'Roll d10 7/4 (income 7)'),
      makeLogEntry(2, 'PlayerA', 'end turn'),
    ];

    expect(groupLogEntriesByTurn(entries)).toEqual([
      {
        turn: 3,
        player: 'PlayerB',
        entries: entries.slice(0, 3),
      },
      {
        turn: 2,
        player: 'PlayerA',
        entries: entries.slice(3),
      },
    ]);
  });
});

describe('formatLogSummary', () => {
  it('adds property rank and suit codes to card actions', () => {
    expect(formatLogSummary('develop 6')).toBe('Develop 2 mo kn (6)');
    expect(formatLogSummary('[PlayerB] sell 7')).toBe(
      '[PlayerB] Sell 2 su wy (7)'
    );
  });

  it('formats income choices and standalone suit names', () => {
    expect(formatLogSummary('income choice 8:Leaves')).toBe(
      'Income choice 2 wa le (8):le'
    );
    expect(formatLogSummary('Tax Moons (PlayerA -2)')).toBe(
      'Tax mo (PlayerA -2)'
    );
  });
});

describe('log metadata helpers', () => {
  it('extracts only seed-prefixed summaries', () => {
    expect(seedSummaryValue('Seed nightly-run')).toBe('nightly-run');
    expect(seedSummaryValue('Income PlayerA none')).toBeNull();
  });

  it('maps suit codes back to suit names', () => {
    expect(suitCodeToSuit('mo')).toBe('Moons');
    expect(suitCodeToSuit('kn')).toBe('Knots');
  });
});

function makeLogEntry(
  turn: number,
  player: GameLogEntry['player'],
  summary: string
): GameLogEntry {
  return {
    turn,
    player,
    phase: 'ActionWindow',
    summary,
  };
}
