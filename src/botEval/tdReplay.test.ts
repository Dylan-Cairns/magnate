import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { legalActions as engineLegalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { heuristicPolicy } from '../policies/heuristicPolicy';
import {
  ACTION_FEATURE_DIM,
  OBSERVATION_DIM,
} from '../policies/trainingEncoding';
import type { ActionPolicy } from '../policies/types';
import { collectTdReplay } from './tdReplay';
import { writeTdReplayArtifacts } from './tdReplayArtifacts';
import type {
  TdReplayConfig,
  TdReplayOpponentSamplePayload,
  TdReplayValueTransitionPayload,
} from './types';

describe('TypeScript TD replay collection', () => {
  const cleanupPaths: string[] = [];

  afterEach(async () => {
    await Promise.all(
      cleanupPaths
        .splice(0)
        .map((entry) => rm(entry, { recursive: true, force: true }))
    );
  });

  it(
    'collects full games with Python-compatible row dimensions and contiguous value timesteps',
    async () => {
      const run = await collectTdReplay(tdReplayConfig({ games: 2 }));

      expect(run.games.map((game) => game.seed)).toEqual([
        'td-replay-test-0',
        'td-replay-test-1',
      ]);
      expect(run.games.map((game) => game.firstPlayer)).toEqual([
        'PlayerA',
        'PlayerB',
      ]);
      expect(run.valueTransitions.length).toBeGreaterThan(0);
      expect(run.opponentSamples.length).toBeGreaterThan(0);

      for (const row of run.valueTransitions) {
        expect(row.observation).toHaveLength(OBSERVATION_DIM);
        if (row.done) {
          expect(row.nextObservation).toBeNull();
        } else {
          expect(row.nextObservation).toHaveLength(OBSERVATION_DIM);
        }
      }
      for (const row of run.opponentSamples) {
        expect(row.observation).toHaveLength(OBSERVATION_DIM);
        expect(row.actionFeatures.length).toBeGreaterThan(0);
        expect(row.actionIndex).toBeGreaterThanOrEqual(0);
        expect(row.actionIndex).toBeLessThan(row.actionFeatures.length);
        for (const features of row.actionFeatures) {
          expect(features).toHaveLength(ACTION_FEATURE_DIM);
        }
      }
      assertContiguousValueSequences(run.valueTransitions);
    },
    15_000
  );

  it(
    'records selected action indexes against the same legal-action order used for encoding',
    async () => {
      const run = await collectTdReplay(tdReplayConfig({ games: 1 }));

      for (const game of run.games) {
        for (const decision of game.decisions) {
          expect(decision.actionIndex).toBeGreaterThanOrEqual(0);
          expect(decision.actionIndex).toBeLessThan(decision.legalActionCount);
          expect(decision.indexedActionKey).toBe(decision.actionKey);
        }
      }
    },
    10_000
  );

  it(
    'passes bridge-canonical legal-action order to replay policies',
    async () => {
      let checkedDecisions = 0;
      let sawRawOrderDifference = false;
      await collectTdReplay(tdReplayConfig({ games: 1 }), {
        createPolicy(): ActionPolicy {
          return {
            selectAction(context) {
              const actualKeys = context.legalActions.map(actionStableKey);
              const canonicalKeys = toKeyedActions(
                engineLegalActions(context.state)
              ).map((entry) => entry.actionKey);
              const rawKeys = engineLegalActions(context.state).map(
                actionStableKey
              );

              expect(actualKeys).toEqual(canonicalKeys);
              if (actualKeys.join('\0') !== rawKeys.join('\0')) {
                sawRawOrderDifference = true;
              }
              checkedDecisions += 1;
              return heuristicPolicy.selectAction(context);
            },
          };
        },
      });

      expect(checkedDecisions).toBeGreaterThan(0);
      expect(sawRawOrderDifference).toBe(true);
    },
    10_000
  );

  it(
    'is deterministic for the same config and seeds',
    async () => {
      const config = tdReplayConfig({ games: 1 });
      const first = await collectTdReplay(config);
      const second = await collectTdReplay(config);

      expect(first.valueTransitions).toEqual(second.valueTransitions);
      expect(first.opponentSamples).toEqual(second.opponentSamples);
      expect(first.games.map(gameResult)).toEqual(second.games.map(gameResult));
    },
    15_000
  );

  it(
    'writes JSONL artifacts and a summary with Python-compatible payload keys',
    async () => {
      const outputDirectory = await mkdtemp(
        path.join(os.tmpdir(), 'magnate-td-replay-')
      );
      cleanupPaths.push(outputDirectory);
      const run = await collectTdReplay(tdReplayConfig({ games: 1 }));

      const written = await writeTdReplayArtifacts(run, outputDirectory, {
        generatedAtUtc: '2026-06-04T00:00:00.000Z',
        git: { commit: 'test-commit', dirty: false },
        nodeVersion: 'test-node',
      });
      const valueRows = await readJsonl<TdReplayValueTransitionPayload>(
        written.valuePath
      );
      const opponentRows =
        await readJsonl<TdReplayOpponentSamplePayload>(written.opponentPath);
      const summary = JSON.parse(
        await readFile(written.summaryPath, 'utf8')
      ) as unknown;

      expect(valueRows).toEqual(run.valueTransitions);
      expect(opponentRows).toEqual(run.opponentSamples);
      expect(Object.keys(valueRows[0]).sort()).toEqual([
        'done',
        'episodeId',
        'nextObservation',
        'observation',
        'playerId',
        'reward',
        'timestep',
      ]);
      expect(Object.keys(opponentRows[0]).sort()).toEqual([
        'actionFeatures',
        'actionIndex',
        'observation',
        'playerId',
      ]);
      expect(summary).toMatchObject({
        schemaVersion: 1,
        artifactType: 'ts-td-replay',
        generatedAtUtc: '2026-06-04T00:00:00.000Z',
        runtime: { nodeVersion: 'test-node' },
        git: { commit: 'test-commit', dirty: false },
        encoding: {
          observationDim: OBSERVATION_DIM,
          actionFeatureDim: ACTION_FEATURE_DIM,
        },
        results: {
          games: 1,
          valueTransitions: run.valueTransitions.length,
          opponentSamples: run.opponentSamples.length,
        },
      });
    },
    10_000
  );
});

function tdReplayConfig({ games }: { games: number }): TdReplayConfig {
  return {
    schemaVersion: 1,
    runLabel: 'td-replay-test',
    seedPrefix: 'td-replay-test',
    games,
    playerA: {
      id: 'tiny-search-a',
      kind: 'search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 1,
        maxRootActions: 2,
        rolloutEpsilon: 0,
      },
    },
    playerB: {
      id: 'heuristic-b',
      kind: 'heuristic',
    },
  };
}

function assertContiguousValueSequences(
  rows: readonly TdReplayValueTransitionPayload[]
): void {
  const bySequence = new Map<string, TdReplayValueTransitionPayload[]>();
  for (const row of rows) {
    const key = `${row.episodeId}:${row.playerId}`;
    const sequence = bySequence.get(key) ?? [];
    sequence.push(row);
    bySequence.set(key, sequence);
  }

  for (const sequence of bySequence.values()) {
    sequence.sort((left, right) => left.timestep - right.timestep);
    sequence.forEach((row, index) => {
      expect(row.timestep).toBe(index);
      expect(row.done).toBe(index === sequence.length - 1);
    });
  }
}

function gameResult(game: Awaited<ReturnType<typeof collectTdReplay>>['games'][number]) {
  return {
    seed: game.seed,
    firstPlayer: game.firstPlayer,
    botBySeat: game.botBySeat,
    actionKeys: game.decisions.map((decision) => decision.actionKey),
    finalScore: game.finalScore,
    turns: game.turns,
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
