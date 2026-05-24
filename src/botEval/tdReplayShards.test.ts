import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import {
  collectAndWriteShardedTdReplayArtifacts,
  planContiguousTdReplayShards,
} from './tdReplayShards';
import type {
  TdReplayConfig,
  TdReplayValueTransitionPayload,
} from './types';

describe('TypeScript TD replay contiguous sharding', () => {
  const cleanupPaths: string[] = [];

  afterEach(async () => {
    await Promise.all(
      cleanupPaths
        .splice(0)
        .map((entry) => rm(entry, { recursive: true, force: true }))
    );
  });

  it('plans contiguous ranges with remainders on earlier shards', () => {
    expect(planContiguousTdReplayShards({ games: 10, workers: 3 })).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 4 },
      { shardIndex: 1, gameIndexStart: 4, games: 3 },
      { shardIndex: 2, gameIndexStart: 7, games: 3 },
    ]);
  });

  it('caps shard count at the number of games', () => {
    expect(planContiguousTdReplayShards({ games: 2, workers: 8 })).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 1 },
      { shardIndex: 1, gameIndexStart: 1, games: 1 },
    ]);
  });

  it('plans one-game shards when shardGames is one', () => {
    const shards = planContiguousTdReplayShards({
      games: 60,
      workers: 6,
      shardGames: 1,
    });
    expect(shards).toHaveLength(60);
    expect(shards.slice(0, 3)).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 1 },
      { shardIndex: 1, gameIndexStart: 1, games: 1 },
      { shardIndex: 2, gameIndexStart: 2, games: 1 },
    ]);
    expect(shards.at(-1)).toEqual({
      shardIndex: 59,
      gameIndexStart: 59,
      games: 1,
    });
  });

  it('plans fixed-size contiguous shards with a smaller final shard', () => {
    expect(
      planContiguousTdReplayShards({
        games: 62,
        workers: 6,
        shardGames: 10,
      })
    ).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 10 },
      { shardIndex: 1, gameIndexStart: 10, games: 10 },
      { shardIndex: 2, gameIndexStart: 20, games: 10 },
      { shardIndex: 3, gameIndexStart: 30, games: 10 },
      { shardIndex: 4, gameIndexStart: 40, games: 10 },
      { shardIndex: 5, gameIndexStart: 50, games: 10 },
      { shardIndex: 6, gameIndexStart: 60, games: 2 },
    ]);
  });

  it('rejects invalid planner inputs', () => {
    expect(() =>
      planContiguousTdReplayShards({ games: 0, workers: 1 })
    ).toThrow('games must be a positive integer');
    expect(() =>
      planContiguousTdReplayShards({ games: 1, workers: 0 })
    ).toThrow('workers must be a positive integer');
    expect(() =>
      planContiguousTdReplayShards({
        games: 1,
        workers: 1,
        shardGames: 0,
      })
    ).toThrow('shardGames must be a positive integer');
  });

  it('writes independent shard artifacts with global game ids and seeds', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-td-replay-sharded-')
    );
    cleanupPaths.push(outputDirectory);

    const written = await collectAndWriteShardedTdReplayArtifacts(
      tdReplayConfig({ games: 3 }),
      outputDirectory,
      {
        workers: 2,
        shardGames: 1,
        generatedAtUtc: '2026-06-04T00:00:00.000Z',
        git: { commit: 'test-commit', dirty: false },
        nodeVersion: 'test-node',
        runBaseName: 'sharded-smoke',
      }
    );

    expect(written.runDirectory).toBe(
      path.join(outputDirectory, 'sharded-smoke')
    );
    expect(written.summary.execution).toEqual({
      requestedWorkers: 2,
      workers: 2,
      parallelUnit: 'contiguous-game-range',
      shards: 3,
      shardGames: 1,
    });
    expect(written.summary.results.games).toBe(3);
    expect(written.summary.shards.map((shard) => shard.games)).toEqual([
      1, 1, 1,
    ]);
    expect(written.summary.shards.map((shard) => shard.gameIndexStart)).toEqual(
      [0, 1, 2]
    );
    expect(written.summary.games.map((game) => game.gameId)).toEqual([
      'game-0000',
      'game-0001',
      'game-0002',
    ]);
    expect(written.summary.games.map((game) => game.seed)).toEqual([
      'td-replay-sharded-test-0',
      'td-replay-sharded-test-1',
      'td-replay-sharded-test-2',
    ]);

    for (const shard of written.summary.shards) {
      expect(
        shard.valuePath.endsWith(`shard-00${shard.shardIndex}.value.jsonl`)
      ).toBe(true);
      expect(
        shard.opponentPath.endsWith(
          `shard-00${shard.shardIndex}.opponent.jsonl`
        )
      ).toBe(true);
      expect(
        shard.summaryPath.endsWith(`shard-00${shard.shardIndex}.summary.json`)
      ).toBe(true);
    }

    const valueRows = await readJsonl<TdReplayValueTransitionPayload>(
      written.summary.shards[0].valuePath
    );
    expect(new Set(valueRows.map((row) => row.episodeId))).toEqual(
      new Set(['td-replay-sharded-test-0'])
    );
    const manifest = JSON.parse(
      await readFile(written.summaryPath, 'utf8')
    ) as unknown;
    expect(manifest).toMatchObject({
      artifactType: 'ts-td-replay-sharded',
      results: {
        games: 3,
      },
    });
  }, 30_000);
});

function tdReplayConfig({ games }: { games: number }): TdReplayConfig {
  return {
    schemaVersion: 1,
    runLabel: 'td-replay-sharded-test',
    seedPrefix: 'td-replay-sharded-test',
    games,
    playerA: {
      id: 'heuristic-a',
      kind: 'heuristic',
    },
    playerB: {
      id: 'heuristic-b',
      kind: 'heuristic',
    },
  };
}

async function readJsonl<T>(inputPath: string): Promise<T[]> {
  const raw = await readFile(inputPath, 'utf8');
  return raw
    .trim()
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as T);
}
