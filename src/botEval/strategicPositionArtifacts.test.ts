import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import type { ActionPolicy } from '../policies/types';
import { createStrategicPositionCatalogV0 } from './strategicPositionCatalog';
import {
  createStrategicPositionArtifactV0,
  renderStrategicPositionSummaryV0,
  writeStrategicPositionArtifactsV0,
} from './strategicPositionArtifacts';
import { runStrategicPositionComparisonV0 } from './strategicPositionComparison';

describe('strategic position artifacts', () => {
  it('writes schema-versioned JSON and a diagnostic Markdown summary', async () => {
    const outputDirectory = await mkdtemp(
      path.join(tmpdir(), 'magnate-strategic-position-')
    );
    try {
      const run = await runStrategicPositionComparisonV0({
        positions: [
          createStrategicPositionCatalogV0().find(
            (position) => position.id === 'minimum-winning-coalition'
          )!,
        ],
        variants: [
          {
            descriptor: {
              kind: 'custom',
              id: 'first-legal',
              label: 'First legal',
              implementationId: 'test:first-legal-v1',
            },
            policy: firstLegalPolicy,
          },
        ],
        now: () => 0,
      });
      const artifact = createStrategicPositionArtifactV0(run, {
        generatedAtUtc: '2026-07-13T00:00:00.000Z',
        git: { commit: 'test', dirty: false },
        nodeVersion: 'v20.19.0',
      });
      const written = await writeStrategicPositionArtifactsV0(
        artifact,
        outputDirectory
      );

      const json = JSON.parse(await readFile(written.artifactPath, 'utf8'));
      const markdown = await readFile(written.summaryPath, 'utf8');
      expect(json).toMatchObject({
        schemaVersion: 1,
        artifactType: 'ts-strategic-position-comparison',
      });
      expect(markdown).toBe(renderStrategicPositionSummaryV0(artifact));
      expect(markdown).toContain('diagnostic characterization');
      expect(markdown).toContain('minimum-winning-coalition');
    } finally {
      await rm(outputDirectory, { recursive: true, force: true });
    }
  });
});

const firstLegalPolicy: ActionPolicy = {
  selectAction(context) {
    return context.legalActions[0];
  },
};
