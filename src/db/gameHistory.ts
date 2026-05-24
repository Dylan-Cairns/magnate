import type { FinalScore, PlayerId } from '../engine/types';
import {
  db,
  type AchievementKey,
  type AchievementRecord,
  type GameRecord,
  type WinnerOutcome,
} from './db';
import { checkNewAchievements, computeStats } from './historyLogic';

export interface RecordGameParams {
  score: FinalScore;
  humanPlayerId: PlayerId;
  botProfileId: string;
  botLabel: string;
}

export interface AchievementWithGame {
  achievement: AchievementRecord;
  game: GameRecord | undefined;
}

export const ACHIEVEMENT_META: Record<
  AchievementKey,
  { label: string; description: string }
> = {
  shutout: {
    label: 'Shutout',
    description: 'Win all 5 districts',
  },
  hard_mode: {
    label: 'Hard Mode',
    description: 'Beat the hard bot',
  },
  hard_mode_streak: {
    label: 'Hard Mode Streak',
    description: 'Beat the hard bot 3 times in a row',
  },
  tactician: {
    label: 'Tactician',
    description: 'Win with more districts but less property points',
  },
};

export async function recordGame(params: RecordGameParams): Promise<void> {
  const { score, humanPlayerId, botProfileId, botLabel } = params;
  const botPlayerId: PlayerId =
    humanPlayerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';

  const winner: WinnerOutcome =
    score.winner === humanPlayerId
      ? 'player'
      : score.winner === 'Draw'
        ? 'draw'
        : 'bot';

  const record: GameRecord = {
    timestamp: Date.now(),
    winner,
    decidedBy: score.decidedBy,
    botProfileId,
    botLabel,
    playerDistricts: score.districtPoints[humanPlayerId],
    botDistricts: score.districtPoints[botPlayerId],
    playerRankTotal: score.rankTotals[humanPlayerId],
    botRankTotal: score.rankTotals[botPlayerId],
    playerResources: score.resourceTotals[humanPlayerId],
    botResources: score.resourceTotals[botPlayerId],
  };

  const gameId = (await db.games.add(record)) as number;

  const [allGames, existingAchievements] = await Promise.all([
    db.games.orderBy('timestamp').toArray(),
    db.achievements.toArray(),
  ]);

  const newKeys = checkNewAchievements(
    { ...record, id: gameId },
    allGames,
    existingAchievements
  );

  if (newKeys.length > 0) {
    const now = Date.now();
    await db.achievements.bulkAdd(
      newKeys.map((key) => ({
        achievementKey: key,
        gameId,
        unlockedAt: now,
      }))
    );
  }
}

export async function getStats() {
  const games = await db.games.toArray();
  return computeStats(games);
}

export async function getGames(): Promise<GameRecord[]> {
  return db.games.orderBy('timestamp').reverse().toArray();
}

export async function getAchievements(): Promise<AchievementWithGame[]> {
  const achievements = await db.achievements.toArray();
  const gameIds = [...new Set(achievements.map((a) => a.gameId))];
  const games = await db.games.where('id').anyOf(gameIds).toArray();
  const gamesById = new Map(games.map((g) => [g.id!, g]));
  return achievements.map((a) => ({
    achievement: a,
    game: gamesById.get(a.gameId),
  }));
}
