import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { testHeadToHeadConfig } from './__tests__/fixtures';
import { runHeadToHead } from './matchup';
import {
  createHeadToHeadArtifact,
  loadHeadToHeadArtifact,
  writeHeadToHeadArtifacts,
} from './artifacts';

describe('TypeScript bot evaluation artifacts', () => {
  const cleanupPaths: string[] = [];

  afterEach(async () => {
    await Promise.all(
      cleanupPaths
        .splice(0)
        .map((entry) => rm(entry, { recursive: true, force: true }))
    );
  });

  it('writes loadable JSON and a readable Markdown summary', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-bot-eval-')
    );
    cleanupPaths.push(outputDirectory);
    const run = await runHeadToHead(testHeadToHeadConfig());
    const artifact = createHeadToHeadArtifact(run, {
      generatedAtUtc: '2026-06-01T00:00:00.000Z',
      git: { commit: 'test-commit', dirty: false },
      nodeVersion: 'test-node',
    });

    const written = await writeHeadToHeadArtifacts(artifact, outputDirectory);
    const loaded = await loadHeadToHeadArtifact(written.artifactPath);
    const markdown = await readFile(written.summaryPath, 'utf8');

    expect(loaded).toEqual(artifact);
    expect(markdown).toContain('# TypeScript Bot Evaluation: matchup-test');
    expect(markdown).toContain(
      'Execution: workers=1 requestedWorkers=1 parallelUnit=paired-seed latencyMode=isolated'
    );
    expect(markdown).toContain('heuristic-candidate');
  });

  it('loads schema-v2 artifacts without execution metadata', async () => {
    const outputDirectory = await mkdtemp(
      path.join(os.tmpdir(), 'magnate-bot-eval-v2-')
    );
    cleanupPaths.push(outputDirectory);
    const artifact = createHeadToHeadArtifact(
      await runHeadToHead(testHeadToHeadConfig())
    );
    artifact.schemaVersion = 2;
    delete artifact.execution;
    const artifactPath = path.join(outputDirectory, 'matchup.json');
    await writeFile(artifactPath, `${JSON.stringify(artifact)}\n`, 'utf8');

    await expect(loadHeadToHeadArtifact(artifactPath)).resolves.toEqual(
      artifact
    );
  });
});
