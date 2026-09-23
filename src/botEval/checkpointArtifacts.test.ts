import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { testHeadToHeadConfig } from './__tests__/fixtures';
import {
  clearHeadToHeadCheckpoint,
  createHeadToHeadCheckpoint,
  headToHeadCheckpointPath,
  headToHeadConfigsMatch,
  loadHeadToHeadCheckpoint,
  writeHeadToHeadCheckpoint,
} from './checkpointArtifacts';
import { runHeadToHead } from './matchup';
import type { HeadToHeadCheckpoint } from './types';

describe('head-to-head checkpoints', () => {
  const cleanupPaths: string[] = [];

  afterEach(async () => {
    await Promise.all(
      cleanupPaths
        .splice(0)
        .map((entry) => rm(entry, { recursive: true, force: true }))
    );
  });

  async function temporaryDirectory(): Promise<string> {
    const directory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-bot-eval-checkpoint-')
    );
    cleanupPaths.push(directory);
    return directory;
  }

  it('round-trips a checkpoint through disk', async () => {
    const outputDirectory = await temporaryDirectory();
    const config = testHeadToHeadConfig(2);
    const complete = await runHeadToHead(config);
    const checkpoint = createHeadToHeadCheckpoint(
      config,
      [{ pairIndex: 0, games: [complete.games[0], complete.games[1]] }],
      12_345
    );
    const checkpointPath = headToHeadCheckpointPath(outputDirectory);

    writeHeadToHeadCheckpoint(checkpoint, checkpointPath);
    const loaded = await loadHeadToHeadCheckpoint(checkpointPath);

    expect(loaded).toEqual(checkpoint);
  });
  it('rejects unsupported schema versions and artifact types', async () => {
    const outputDirectory = await temporaryDirectory();
    const checkpointPath = headToHeadCheckpointPath(outputDirectory);
    const checkpoint = createHeadToHeadCheckpoint(
      testHeadToHeadConfig(),
      [],
      0
    );

    await writeFile(
      checkpointPath,
      `${JSON.stringify({ ...checkpoint, schemaVersion: 99 })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'Unsupported head-to-head checkpoint schemaVersion=99.'
    );

    await writeFile(
      checkpointPath,
      `${JSON.stringify({ ...checkpoint, artifactType: 'other' })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'Unsupported head-to-head checkpoint type=other.'
    );
  });

  it('rejects malformed elapsed time and results', async () => {
    const outputDirectory = await temporaryDirectory();
    const checkpointPath = headToHeadCheckpointPath(outputDirectory);
    const config = testHeadToHeadConfig();
    const complete = await runHeadToHead(config);
    const checkpoint = createHeadToHeadCheckpoint(
      config,
      [{ pairIndex: 0, games: [complete.games[0], complete.games[1]] }],
      500
    );

    await writeFile(
      checkpointPath,
      `${JSON.stringify({ ...checkpoint, elapsedMs: -1 })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'head-to-head checkpoint.elapsedMs must be a finite number >= 0.'
    );

    await writeFile(
      checkpointPath,
      `${JSON.stringify({
        ...checkpoint,
        results: [{ pairIndex: -1, games: checkpoint.results[0]?.games }],
      })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'head-to-head checkpoint.results[0].pairIndex must be a nonnegative integer.'
    );

    await writeFile(
      checkpointPath,
      `${JSON.stringify({
        ...checkpoint,
        results: [
          { pairIndex: 0, games: [checkpoint.results[0]?.games[0]] },
        ],
      })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'head-to-head checkpoint.results[0].games must contain exactly two played games.'
    );

    const firstGame = checkpoint.results[0]?.games[0];
    await writeFile(
      checkpointPath,
      `${JSON.stringify({
        ...checkpoint,
        results: [
          {
            pairIndex: 0,
            games: [{ ...firstGame, botBySeat: undefined }, firstGame],
          },
        ],
      })}\n`,
      'utf8'
    );
    await expect(loadHeadToHeadCheckpoint(checkpointPath)).rejects.toThrow(
      'head-to-head checkpoint.results[0].games[0].botBySeat must be an object.'
    );
  });

  it('clears checkpoints idempotently', async () => {
    const outputDirectory = await temporaryDirectory();
    const checkpointPath = headToHeadCheckpointPath(outputDirectory);
    const checkpoint = createHeadToHeadCheckpoint(
      testHeadToHeadConfig(),
      [],
      0
    );

    writeHeadToHeadCheckpoint(checkpoint, checkpointPath);
    clearHeadToHeadCheckpoint(checkpointPath);
    await expect(readFile(checkpointPath, 'utf8')).rejects.toThrow();
    expect(() => clearHeadToHeadCheckpoint(checkpointPath)).not.toThrow();
  });

  it('matches configs canonically and rejects mismatches', () => {
    const config = testHeadToHeadConfig(2);

    expect(headToHeadConfigsMatch(config, structuredClone(config))).toBe(true);
    expect(
      headToHeadConfigsMatch(config, { ...config, seedPrefix: 'other' })
    ).toBe(false);
    expect(headToHeadConfigsMatch(config, { ...config, gamesPerSide: 3 })).toBe(
      false
    );
    expect(
      headToHeadConfigsMatch(config, {
        ...config,
        candidate: { id: 'heuristic-candidate', kind: 'heuristic' },
        opponent: { id: 'random-opponent', kind: 'random' },
      })
    ).toBe(true);
    expect(
      headToHeadConfigsMatch(config, {
        ...config,
        candidate: { id: 'heuristic-candidate', kind: 'random' },
      })
    ).toBe(false);
  });

  it('resumes from a checkpoint without replaying completed pairs', async () => {
    const outputDirectory = await temporaryDirectory();
    const config = testHeadToHeadConfig(2);
    const complete = await runHeadToHead(config);
    const checkpointPath = headToHeadCheckpointPath(outputDirectory);
    writeHeadToHeadCheckpoint(
      createHeadToHeadCheckpoint(
        config,
        [{ pairIndex: 0, games: [complete.games[0], complete.games[1]] }],
        7_000
      ),
      checkpointPath
    );

    const loaded: HeadToHeadCheckpoint =
      await loadHeadToHeadCheckpoint(checkpointPath);
    const completedGameIds: string[] = [];
    const resumed = await runHeadToHead(config, {
      initialResults: loaded.results,
      initialElapsedMs: loaded.elapsedMs,
      onProgress(progress) {
        if (progress.type === 'game-completed') {
          completedGameIds.push(progress.game.gameId);
        }
      },
    });

    expect(completedGameIds).toEqual([
      'pair-0002-candidate-as-a',
      'pair-0002-candidate-as-b',
    ]);
    expect(resumed.games.map(gameSummary)).toEqual(
      complete.games.map(gameSummary)
    );
    expect(resumed.summary.elapsedMs).toBeGreaterThanOrEqual(7_000);
  });
});

function gameSummary(
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
