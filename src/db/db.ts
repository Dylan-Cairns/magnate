import Dexie, { type Table } from 'dexie';

export type WinnerOutcome = 'player' | 'bot' | 'draw';
export type WinnerDecider = 'districts' | 'rank-total' | 'resources' | 'draw';
export type AchievementKey =
  | 'shutout'
  | 'hard_mode'
  | 'hard_mode_streak'
  | 'tactician';

export interface GameRecord {
  id?: number;
  timestamp: number;
  winner: WinnerOutcome;
  decidedBy: WinnerDecider;
  botProfileId: string;
  botLabel: string;
  playerDistricts: number;
  botDistricts: number;
  playerRankTotal: number;
  botRankTotal: number;
  playerResources: number;
  botResources: number;
}

export interface AchievementRecord {
  id?: number;
  achievementKey: AchievementKey;
  gameId: number;
  unlockedAt: number;
}

class MagnateDb extends Dexie {
  games!: Table<GameRecord>;
  achievements!: Table<AchievementRecord>;

  constructor() {
    super('magnate');
    this.version(1).stores({
      games: '++id, timestamp, winner, botProfileId',
      achievements: '++id, achievementKey, gameId',
    });
  }
}

export const db = new MagnateDb();
