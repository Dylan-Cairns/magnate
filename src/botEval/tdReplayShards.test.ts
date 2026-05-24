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
    expect(planContiguousTdReplayShards(10, 3)).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 4 },
      { shardIndex: 1, gameIndexStart: 4, games: 3 },
      { shardIndex: 2, gameIndexStart: 7, games: 3 },
    ]);
  });

  it('caps shard count at the number of games', () => {
    expect(planContiguousTdReplayShards(2, 8)).toEqual([
      { shardIndex: 0, gameIndexStart: 0, games: 1 },
      { shardIndex: 1, gameIndexStart: 1, games: 1 },
    ]);
  });

  it('rejects invalid planner inputs', () => {
    expect(() => planContiguousTdReplayShards(0, 1)).toThrow(
      'games must be a positive integer'
    );
    expect(() => planContiguousTdReplayShards(1, 0)).toThrow(
      'workers must be a positive integer'
    );
  });

  it('writes independent shard artifacts with global game ids and seeds', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-td-replay-sharded-')
    );
    cleanupPaths.push(outputDirectory);

    const written = await collectAndWriteShardedTdReplayArtifacts(
      tdReplayConfig({ games: 2 }),
      outputDirectory,
      {
        workers: 2,
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
    });
    expect(written.summary.results.games).toBe(2);
    expect(written.summary.shards.map((shard) => shard.games)).toEqual([1, 1]);
    expect(written.summary.shards.map((shard) => shard.gameIndexStart)).toEqual(
      [0, 1]
    );
    expect(written.summary.games.map((game) => game.gameId)).toEqual([
      'game-0000',
      'game-0001',
    ]);
    expect(written.summary.games.map((game) => game.seed)).toEqual([
      'td-replay-sharded-test-0',
      'td-replay-sharded-test-1',
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
        games: 2,
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
