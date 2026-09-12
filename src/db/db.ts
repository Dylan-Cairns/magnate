import Dexie, { type Table } from 'dexie';

import type { Ruleset } from '../engine/types';

export type WinnerOutcome = 'player' | 'bot' | 'draw';
export type WinnerDecider = 'districts' | 'rank-total' | 'resources' | 'draw';

export interface GameRecord {
  id?: number;
  sessionId?: string;
  timestamp: number;
  winner: WinnerOutcome;
  decidedBy: WinnerDecider;
  botProfileId: string;
  botLabel: string;
  ruleset?: Ruleset;
  playerDistricts: number;
  botDistricts: number;
  playerRankTotal: number;
  botRankTotal: number;
  playerResources: number;
  botResources: number;
}

class MagnateDb extends Dexie {
  games!: Table<GameRecord>;

  constructor() {
    super('magnate');
    this.version(1).stores({
      games: '++id, timestamp, winner, botProfileId',
      achievements: '++id, achievementKey, gameId',
    });
    this.version(2).stores({
      games: '++id, &sessionId, timestamp, winner, botProfileId',
      achievements: '++id, achievementKey, gameId',
    });
    this.version(3).stores({
      games: '++id, &sessionId, timestamp, winner, botProfileId',
      achievements: null,
    });
    this.version(4)
      .stores({
        games: '++id, &sessionId, timestamp, winner, botProfileId',
      })
      .upgrade((tx) =>
        tx
          .table('games')
          .toCollection()
          .modify((game: { ruleset?: string }) => {
            if (game.ruleset === 'regular') game.ruleset = 'standard';
          })
      );
  }
}

export const db = new MagnateDb();
