import { WinnerDecider } from '../engine/values';
import Dexie, { type Table } from 'dexie';

import { Ruleset } from '../engine/types';

import type { WinnerOutcome } from './values';
export { WinnerOutcome } from './values';

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
            if (game.ruleset === 'regular') game.ruleset = Ruleset.Standard;
          })
      );
  }
}

export const db = new MagnateDb();

export { WinnerDecider } from '../engine/values';
