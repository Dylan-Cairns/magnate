import type { AchievementKey, AchievementRecord, GameRecord } from './db';

export const HARD_BOT_PROFILE_ID = 'rollout-search-v2-hard';

export interface Stats {
  gamesPlayed: number;
  currentStreak: number;
  longestStreak: number;
}

export function computeStats(games: GameRecord[]): Stats {
  const sorted = [...games].sort((a, b) => a.timestamp - b.timestamp);

  let longestStreak = 0;
  let runningStreak = 0;

  for (const game of sorted) {
    if (game.winner === 'player') {
      runningStreak++;
      if (runningStreak > longestStreak) longestStreak = runningStreak;
    } else {
      runningStreak = 0;
    }
  }

  return {
    gamesPlayed: games.length,
    currentStreak: runningStreak,
    longestStreak,
  };
}

// Returns achievement keys newly earned by newGame.
// allGames must be sorted ascending by timestamp and must include newGame.
// existingAchievements is the full set already stored (used to prevent re-earning one-time unlocks).
export function checkNewAchievements(
  newGame: GameRecord & { id: number },
  allGames: GameRecord[],
  existingAchievements: AchievementRecord[]
): AchievementKey[] {
  const alreadyEarned = new Set(existingAchievements.map((a) => a.achievementKey));
  const unlocked: AchievementKey[] = [];

  const earn = (key: AchievementKey) => {
    if (!alreadyEarned.has(key)) unlocked.push(key);
  };

  if (newGame.winner !== 'player') return unlocked;

  if (newGame.playerDistricts === 5) {
    earn('shutout');
  }

  if (newGame.playerRankTotal < newGame.botRankTotal) {
    earn('tactician');
  }

  if (newGame.botProfileId === HARD_BOT_PROFILE_ID) {
    earn('hard_mode');

    // 3 consecutive games overall must all be hard-bot wins; any other bot breaks the streak
    const lastThree = allGames.slice(-3);
    if (
      lastThree.length === 3 &&
      lastThree.every(
        (g) => g.botProfileId === HARD_BOT_PROFILE_ID && g.winner === 'player'
      )
    ) {
      earn('hard_mode_streak');
    }
  }

  return unlocked;
}
