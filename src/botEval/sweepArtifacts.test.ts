import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { loadHeadToHeadArtifact } from './artifacts';
import { runRolloutSearchSweep } from './sweep';
import {
  loadRolloutSearchSweepArtifact,
  writeRolloutSearchSweepArtifacts,
} from './sweepArtifacts';
import type { RolloutSearchSweepConfig } from './types';

describe('rollout-search sweep artifacts', () => {
  const cleanupPaths: string[] = [];

  afterEach(async () => {
    await Promise.all(
      cleanupPaths
        .splice(0)
        .map((entry) => rm(entry, { recursive: true, force: true }))
    );
  });

  it('writes aggregate JSON, escaped CSV, Markdown, and child matchup artifacts', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-bot-sweep-')
    );
    cleanupPaths.push(outputDirectory);
    const run = await runRolloutSearchSweep(sweepConfig());

    const written = await writeRolloutSearchSweepArtifacts(
      run,
      outputDirectory,
      {
        generatedAtUtc: '2026-06-01T00:00:00.000Z',
        git: { commit: 'test-commit', dirty: false },
        nodeVersion: 'test-node',
      }
    );
    const loaded = await loadRolloutSearchSweepArtifact(written.artifactPath);
    const csv = await readFile(written.csvPath, 'utf8');
    const markdown = await readFile(written.summaryPath, 'utf8');
    const child = await loadHeadToHeadArtifact(
      path.join(outputDirectory, loaded.rows[0].matchupArtifactPath)
    );

    expect(loaded).toEqual(written.artifact);
    expect(csv).toContain('"search,candidate"');
    expect(markdown).toContain(
      '# TypeScript Rollout Search Sweep: artifact-test'
    );
    expect(markdown).toContain('## Root-Action Latency: search,candidate');
    expect(child.config.candidate.id).toBe('search,candidate');
  });
});

function sweepConfig(): RolloutSearchSweepConfig {
  return {
    schemaVersion: 1,
    runLabel: 'artifact-test',
    seedPrefix: 'artifact-test',
    gamesPerSide: 1,
    opponent: {
      id: 'random-opponent',
      kind: 'random',
    },
    candidates: [
      {
        id: 'search,candidate',
        kind: 'search',
        config: {
          worlds: 1,
          rollouts: 1,
          depth: 1,
          maxRootActions: 1,
          rolloutEpsilon: 0,
        },
      },
    ],
  };
}
