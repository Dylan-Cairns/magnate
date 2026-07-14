import { describe, expect, it } from 'vitest';

import { testHeadToHeadConfig } from './__tests__/fixtures';
import { runHeadToHead } from './matchup';

describe('paired TypeScript bot matchups', () => {
  it('uses paired seeds, swaps seats, and alternates the first player seat', async () => {
    const completedPairs: number[] = [];
    const completedGameIds: string[] = [];
    const run = await runHeadToHead(testHeadToHeadConfig(2), {
      onProgress(progress) {
        if (progress.type === 'game-completed') {
          completedGameIds.push(progress.game.gameId);
        } else if (progress.type === 'pair-completed') {
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
    expect(completedGameIds).toEqual([
      'pair-0001-candidate-as-a',
      'pair-0001-candidate-as-b',
      'pair-0002-candidate-as-a',
      'pair-0002-candidate-as-b',
    ]);
  });

  it(
    'runs paired seeds in child processes and preserves deterministic artifact ordering',
    async () => {
      const config = testHeadToHeadConfig(2);
      const sequential = await runHeadToHead(config);
      const completedGameIds: string[] = [];
      const parallel = await runHeadToHead(config, {
        workers: 2,
        onProgress(progress) {
          if (progress.type === 'game-completed') {
            completedGameIds.push(progress.game.gameId);
          }
        },
      });

      expect(parallel.execution).toEqual({
        requestedWorkers: 2,
        workers: 2,
        parallelUnit: 'paired-seed',
        latencyMode: 'loaded',
      });
      expect(parallel.games.map(gameResult)).toEqual(
        sequential.games.map(gameResult)
      );
      expect([...completedGameIds].sort()).toEqual([
        'pair-0001-candidate-as-a',
        'pair-0001-candidate-as-b',
        'pair-0002-candidate-as-a',
        'pair-0002-candidate-as-b',
      ]);
    },
    15_000
  );

  it('caps workers at the number of paired seeds', async () => {
    const run = await runHeadToHead(testHeadToHeadConfig(1), { workers: 4 });

    expect(run.execution).toEqual({
      requestedWorkers: 4,
      workers: 1,
      parallelUnit: 'paired-seed',
      latencyMode: 'isolated',
    });
  });

  it('rejects invalid worker counts', async () => {
    await expect(
      runHeadToHead(testHeadToHeadConfig(), { workers: 0 })
    ).rejects.toThrow('workers must be a positive integer.');
  });

  it(
    'fails the parent matchup when a child pair fails',
    async () => {
      await expect(
        runHeadToHead(
          {
            ...testHeadToHeadConfig(2),
            maxDecisionsPerGame: 1,
          },
          { workers: 2 }
        )
      ).rejects.toThrow(/Pair worker \d+ failed on pair \d+: Game .* exceeded/);
    },
    10_000
  );
});

function gameResult(
  game: Awaited<ReturnType<typeof runHeadToHead>>['games'][number]
) {
  return {
    gameId: game.gameId,
    seed: game.seed,
    firstPlayer: game.firstPlayer,
    botBySeat: game.botBySeat,
    actionKeys: game.transcript.map((decision) => decision.actionKey),
    finalScore: game.finalScore,
    turns: game.turns,
  };
}
