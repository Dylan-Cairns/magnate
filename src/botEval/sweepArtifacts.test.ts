import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
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
    expect(csv).toContain('workers,latencyMode');
    expect(markdown).toContain(
      '# TypeScript Rollout Search Sweep: artifact-test'
    );
    expect(markdown).toContain(
      'Execution: workers=1 requestedWorkers=1 parallelUnit=paired-seed latencyMode=isolated'
    );
    expect(markdown).toContain('## Root-Action Latency: search,candidate');
    expect(child.execution?.latencyMode).toBe('isolated');
    expect(child.config.candidate.id).toBe('search,candidate');

    const legacy = structuredClone(written.artifact);
    legacy.schemaVersion = 1;
    delete legacy.execution;
    for (const row of legacy.rows) {
      delete (row as Partial<typeof row>).execution;
    }
    const legacyPath = path.join(outputDirectory, 'legacy-sweep.json');
    await writeFile(legacyPath, `${JSON.stringify(legacy)}\n`, 'utf8');
    await expect(loadRolloutSearchSweepArtifact(legacyPath)).resolves.toEqual(
      legacy
    );
  });

  it('refreshes partial running artifacts before sweep completion', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-bot-sweep-partial-')
    );
    cleanupPaths.push(outputDirectory);
    const completeRun = await runRolloutSearchSweep(sweepConfig());
    const options = {
      generatedAtUtc: '2026-06-01T00:00:00.000Z',
      git: { commit: 'test-commit', dirty: false },
      nodeVersion: 'test-node',
    };

    await writeRolloutSearchSweepArtifacts(
      {
        config: completeRun.config,
        execution: completeRun.execution,
        matchups: [],
      },
      outputDirectory,
      {
        ...options,
        status: 'running',
        matchupIndicesToWrite: [],
      }
    );
    const initial = await loadRolloutSearchSweepArtifact(
      path.join(outputDirectory, 'sweep.json')
    );
    expect(initial.status).toBe('running');
    expect(initial.completedCandidates).toBe(0);

    const written = await writeRolloutSearchSweepArtifacts(
      completeRun,
      outputDirectory,
      {
        ...options,
        status: 'running',
        matchupIndicesToWrite: [0],
      }
    );
    const partial = await loadRolloutSearchSweepArtifact(written.artifactPath);
    const markdown = await readFile(written.summaryPath, 'utf8');

    expect(partial.status).toBe('running');
    expect(partial.completedCandidates).toBe(1);
    expect(partial.totalCandidates).toBe(1);
    expect(markdown).toContain('Status: running (1/1 candidates)');
    await expect(
      loadHeadToHeadArtifact(
        path.join(outputDirectory, partial.rows[0].matchupArtifactPath)
      )
    ).resolves.toBeDefined();
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
