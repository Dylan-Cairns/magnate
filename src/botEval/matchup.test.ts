import { describe, expect, it } from 'vitest';

import { testHeadToHeadConfig } from './__tests__/fixtures';
import { runHeadToHead } from './matchup';

describe('paired TypeScript bot matchups', () => {
  it('uses paired seeds, swaps seats, and alternates the first player seat', async () => {
    const completedPairs: number[] = [];
    const run = await runHeadToHead(testHeadToHeadConfig(2), {
      onProgress(progress) {
        if (progress.type === 'pair-completed') {
          completedPairs.push(progress.completedPairs);
        }
      },
    });

    expect(run.games).toHaveLength(4);
    expect(run.games.map((game) => game.seed)).toEqual([
      'matchup-test-0001',
      'matchup-test-0001',
      'matchup-test-0002',
      'matchup-test-0002',
    ]);
    expect(run.games.map((game) => game.firstPlayer)).toEqual([
      'PlayerA',
      'PlayerA',
      'PlayerB',
      'PlayerB',
    ]);
    expect(run.games[0].botBySeat).toEqual({
      PlayerA: 'heuristic-candidate',
      PlayerB: 'random-opponent',
    });
    expect(run.games[1].botBySeat).toEqual({
      PlayerA: 'random-opponent',
      PlayerB: 'heuristic-candidate',
    });
    expect(run.summary.totalGames).toBe(4);
    expect(
      run.summary.latencyByBotId['heuristic-candidate'].actions
    ).toBeGreaterThan(0);
    expect(
      run.summary.multiChoiceLatencyByBotId['heuristic-candidate'].actions
    ).toBe(
      run.games
        .flatMap((game) => game.transcript)
        .filter(
          (decision) =>
            decision.botId === 'heuristic-candidate' &&
            decision.legalActionCount > 1
        ).length
    );
    expect(
      run.summary.multiChoiceLatencyByBotId['heuristic-candidate'].actions
    ).toBeLessThan(run.summary.latencyByBotId['heuristic-candidate'].actions);
    expect(completedPairs).toEqual([1, 2]);
  });
});
