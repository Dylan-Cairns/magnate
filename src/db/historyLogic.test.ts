import { describe, expect, it } from 'vitest';
import { checkNewAchievements, computeStats, HARD_BOT_PROFILE_ID } from './historyLogic';
import type { GameRecord } from './db';

const EASY_BOT = 'rollout-search-v2-easy';

function makeGame(overrides: Partial<GameRecord> & { id: number }): GameRecord & { id: number } {
  return {
    timestamp: overrides.timestamp ?? Date.now(),
    winner: overrides.winner ?? 'player',
    decidedBy: overrides.decidedBy ?? 'districts',
    botProfileId: overrides.botProfileId ?? EASY_BOT,
    botLabel: overrides.botLabel ?? 'V2 Easy',
    playerDistricts: overrides.playerDistricts ?? 3,
    botDistricts: overrides.botDistricts ?? 2,
    playerRankTotal: overrides.playerRankTotal ?? 20,
    botRankTotal: overrides.botRankTotal ?? 15,
    playerResources: overrides.playerResources ?? 5,
    botResources: overrides.botResources ?? 3,
    ...overrides,
  };
}

// ─── computeStats ────────────────────────────────────────────────────────────

describe('computeStats', () => {
  it('returns zeros for empty history', () => {
    expect(computeStats([])).toEqual({
      gamesPlayed: 0,
      currentStreak: 0,
      longestStreak: 0,
    });
  });

  it('counts games played', () => {
    const games = [
      makeGame({ id: 1, winner: 'player', timestamp: 1 }),
      makeGame({ id: 2, winner: 'bot', timestamp: 2 }),
      makeGame({ id: 3, winner: 'player', timestamp: 3 }),
    ];
    expect(computeStats(games).gamesPlayed).toBe(3);
  });

  it('computes current streak as consecutive wins at the end', () => {
    const games = [
      makeGame({ id: 1, winner: 'bot', timestamp: 1 }),
      makeGame({ id: 2, winner: 'player', timestamp: 2 }),
      makeGame({ id: 3, winner: 'player', timestamp: 3 }),
      makeGame({ id: 4, winner: 'player', timestamp: 4 }),
    ];
    expect(computeStats(games).currentStreak).toBe(3);
  });

  it('resets current streak after a loss', () => {
    const games = [
      makeGame({ id: 1, winner: 'player', timestamp: 1 }),
      makeGame({ id: 2, winner: 'player', timestamp: 2 }),
      makeGame({ id: 3, winner: 'bot', timestamp: 3 }),
    ];
    expect(computeStats(games).currentStreak).toBe(0);
  });

  it('tracks longest streak independently of current streak', () => {
    const games = [
      makeGame({ id: 1, winner: 'player', timestamp: 1 }),
      makeGame({ id: 2, winner: 'player', timestamp: 2 }),
      makeGame({ id: 3, winner: 'player', timestamp: 3 }),
      makeGame({ id: 4, winner: 'bot', timestamp: 4 }),
      makeGame({ id: 5, winner: 'player', timestamp: 5 }),
    ];
    const stats = computeStats(games);
    expect(stats.longestStreak).toBe(3);
    expect(stats.currentStreak).toBe(1);
  });

  it('handles a draw as a streak-breaker', () => {
    const games = [
      makeGame({ id: 1, winner: 'player', timestamp: 1 }),
      makeGame({ id: 2, winner: 'draw', timestamp: 2 }),
      makeGame({ id: 3, winner: 'player', timestamp: 3 }),
    ];
    const stats = computeStats(games);
    expect(stats.longestStreak).toBe(1);
    expect(stats.currentStreak).toBe(1);
  });
});

// ─── checkNewAchievements ────────────────────────────────────────────────────

describe('checkNewAchievements', () => {
  it('returns no achievements for a bot win', () => {
    const game = makeGame({ id: 1, winner: 'bot' });
    expect(checkNewAchievements(game, [game], [])).toEqual([]);
  });

  describe('shutout', () => {
    it('unlocks when player wins all 5 districts', () => {
      const game = makeGame({ id: 1, playerDistricts: 5, botDistricts: 0 });
      expect(checkNewAchievements(game, [game], [])).toContain('shutout');
    });

    it('does not unlock when player wins fewer than 5 districts', () => {
      const game = makeGame({ id: 1, playerDistricts: 4 });
      expect(checkNewAchievements(game, [game], [])).not.toContain('shutout');
    });

    it('does not re-earn if already unlocked', () => {
      const game = makeGame({ id: 1, playerDistricts: 5 });
      const existing = [{ id: 1, achievementKey: 'shutout' as const, gameId: 99, unlockedAt: 0 }];
      expect(checkNewAchievements(game, [game], existing)).not.toContain('shutout');
    });
  });

  describe('tactician', () => {
    it('unlocks when player wins with lower rank total', () => {
      const game = makeGame({ id: 1, playerRankTotal: 10, botRankTotal: 20 });
      expect(checkNewAchievements(game, [game], [])).toContain('tactician');
    });

    it('does not unlock when player has higher rank total', () => {
      const game = makeGame({ id: 1, playerRankTotal: 25, botRankTotal: 20 });
      expect(checkNewAchievements(game, [game], [])).not.toContain('tactician');
    });

    it('does not unlock when rank totals are equal', () => {
      const game = makeGame({ id: 1, playerRankTotal: 20, botRankTotal: 20 });
      expect(checkNewAchievements(game, [game], [])).not.toContain('tactician');
    });
  });

  describe('hard_mode', () => {
    it('unlocks on first win against hard bot', () => {
      const game = makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID });
      expect(checkNewAchievements(game, [game], [])).toContain('hard_mode');
    });

    it('does not unlock against other bots', () => {
      const game = makeGame({ id: 1, botProfileId: EASY_BOT });
      expect(checkNewAchievements(game, [game], [])).not.toContain('hard_mode');
    });

    it('does not re-earn if already unlocked', () => {
      const game = makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID });
      const existing = [{ id: 1, achievementKey: 'hard_mode' as const, gameId: 99, unlockedAt: 0 }];
      expect(checkNewAchievements(game, [game], existing)).not.toContain('hard_mode');
    });
  });

  describe('hard_mode_streak', () => {
    it('unlocks after 3 consecutive hard bot wins', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 1 }),
        makeGame({ id: 2, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 2 }),
        makeGame({ id: 3, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 3 }),
      ];
      expect(checkNewAchievements(games[2], games, [])).toContain('hard_mode_streak');
    });

    it('does not unlock with only 2 hard bot wins', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 1 }),
        makeGame({ id: 2, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 2 }),
      ];
      expect(checkNewAchievements(games[1], games, [])).not.toContain('hard_mode_streak');
    });

    it('does not unlock if one of the last 3 hard games was a loss', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, winner: 'bot', timestamp: 1 }),
        makeGame({ id: 2, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 2 }),
        makeGame({ id: 3, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 3 }),
      ];
      expect(checkNewAchievements(games[2], games, [])).not.toContain('hard_mode_streak');
    });

    it('a game against a different bot breaks the streak', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 1 }),
        makeGame({ id: 2, botProfileId: EASY_BOT, winner: 'bot', timestamp: 2 }),
        makeGame({ id: 3, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 3 }),
        makeGame({ id: 4, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 4 }),
      ];
      // Last 3 games: easy loss (2), hard win (3), hard win (4) — streak broken
      expect(checkNewAchievements(games[3], games, [])).not.toContain('hard_mode_streak');
    });

    it('a hard bot loss breaks the streak', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 1 }),
        makeGame({ id: 2, botProfileId: HARD_BOT_PROFILE_ID, winner: 'bot', timestamp: 2 }),
        makeGame({ id: 3, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 3 }),
        makeGame({ id: 4, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 4 }),
      ];
      // Last 3 games: hard loss (2), hard win (3), hard win (4) — streak broken
      expect(checkNewAchievements(games[3], games, [])).not.toContain('hard_mode_streak');
    });

    it('does not re-earn if already unlocked', () => {
      const games = [
        makeGame({ id: 1, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 1 }),
        makeGame({ id: 2, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 2 }),
        makeGame({ id: 3, botProfileId: HARD_BOT_PROFILE_ID, timestamp: 3 }),
      ];
      const existing = [{ id: 1, achievementKey: 'hard_mode_streak' as const, gameId: 99, unlockedAt: 0 }];
      expect(checkNewAchievements(games[2], games, existing)).not.toContain('hard_mode_streak');
    });
  });

  it('can unlock multiple achievements in a single game', () => {
    const game = makeGame({
      id: 1,
      botProfileId: HARD_BOT_PROFILE_ID,
      playerDistricts: 5,
      botDistricts: 0,
      playerRankTotal: 10,
      botRankTotal: 20,
    });
    const keys = checkNewAchievements(game, [game], []);
    expect(keys).toContain('shutout');
    expect(keys).toContain('tactician');
    expect(keys).toContain('hard_mode');
  });
});
