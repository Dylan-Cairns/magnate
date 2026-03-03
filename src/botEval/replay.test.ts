import { describe, expect, it } from 'vitest';

import { testHeadToHeadConfig } from './__tests__/fixtures';
import { createHeadToHeadArtifact } from './artifacts';
import { runHeadToHead } from './matchup';
import { replayArtifactGame } from './replay';
import type { DecisionRecord } from './types';

describe('TypeScript bot evaluation replay', () => {
  it('replays a recorded game exactly', async () => {
    const artifact = await makeArtifact();
    const result = await replayArtifactGame(artifact, artifact.games[0].gameId);

    expect(result.matched).toBe(true);
    expect(result.decisions).toBe(artifact.games[0].transcript.length);
  });

  it('reports the first modified transcript action', async () => {
    const artifact = await makeArtifact();
    artifact.games[0].transcript[0].actionKey = 'intentionally-modified';

    await expect(
      replayArtifactGame(artifact, artifact.games[0].gameId)
    ).rejects.toThrow('at decision 0');
  });

  it('rejects artifacts from a different policy RNG scheme', async () => {
    const artifact = await makeArtifact();
    artifact.policyRandomSchemeVersion = 'future-scheme';

    await expect(
      replayArtifactGame(artifact, artifact.games[0].gameId)
    ).rejects.toThrow('RNG scheme mismatch');
  });

  it('replays schema-v1 artifacts without diagnostics', async () => {
    const artifact = await makeArtifact();
    artifact.schemaVersion = 1;
    for (const game of artifact.games) {
      for (const decision of game.transcript) {
        delete (decision as Partial<DecisionRecord>).legalActionCount;
        delete decision.searchDiagnostics;
      }
    }

    const result = await replayArtifactGame(artifact, artifact.games[0].gameId);

    expect(result.matched).toBe(true);
  });
});

async function makeArtifact() {
  const run = await runHeadToHead(testHeadToHeadConfig());
  return createHeadToHeadArtifact(run, {
    generatedAtUtc: '2026-06-01T00:00:00.000Z',
    git: { commit: 'test-commit', dirty: false },
    nodeVersion: 'test-node',
  });
}
