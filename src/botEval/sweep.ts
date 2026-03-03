import { performance } from 'node:perf_hooks';

import {
  runHeadToHead,
  type HeadToHeadDependencies,
  type HeadToHeadProgress,
} from './matchup';
import { resolveEvaluationExecution } from './execution';
import type {
  HeadToHeadConfig,
  HeadToHeadRun,
  RolloutSearchSweepConfig,
  RolloutSearchSweepRun,
} from './types';

export interface RolloutSearchSweepDependencies {
  runMatchup?: (
    config: HeadToHeadConfig,
    dependencies?: HeadToHeadDependencies
  ) => Promise<HeadToHeadRun>;
  now?: () => number;
  workers?: number;
  progressIntervalMs?: number;
  onProgress?: (progress: RolloutSearchSweepProgress) => void;
  onCandidateCompleted?: (
    completed: RolloutSearchSweepCandidateCompleted
  ) => Promise<void> | void;
}

export type RolloutSearchSweepProgress =
  | {
      type: 'sweep-started';
      candidates: number;
      gamesPerSide: number;
      totalGames: number;
      workers: number;
    }
  | {
      type: 'candidate-started';
      candidateIndex: number;
      totalCandidates: number;
      candidateId: string;
      configLabel: string;
      configProxyCost: number;
    }
  | ({
      candidateIndex: number;
      totalCandidates: number;
    } & HeadToHeadProgress)
  | {
      type: 'candidate-completed';
      candidateIndex: number;
      totalCandidates: number;
      candidateId: string;
      matchup: HeadToHeadRun;
      elapsedMs: number;
    }
  | {
      type: 'sweep-completed';
      candidates: number;
      elapsedMs: number;
    };

export interface RolloutSearchSweepCandidateCompleted {
  run: RolloutSearchSweepRun;
  candidateIndex: number;
  totalCandidates: number;
  matchup: HeadToHeadRun;
}

export async function runRolloutSearchSweep(
  config: RolloutSearchSweepConfig,
  dependencies: RolloutSearchSweepDependencies = {}
): Promise<RolloutSearchSweepRun> {
  const runMatchup = dependencies.runMatchup ?? runHeadToHead;
  const now = dependencies.now ?? (() => performance.now());
  const matchups: HeadToHeadRun[] = [];
  const execution = resolveEvaluationExecution(
    dependencies.workers ?? 1,
    config.gamesPerSide
  );
  const startedAt = now();
  const totalCandidates = config.candidates.length;
  dependencies.onProgress?.({
    type: 'sweep-started',
    candidates: totalCandidates,
    gamesPerSide: config.gamesPerSide,
    totalGames: totalCandidates * config.gamesPerSide * 2,
    workers: execution.workers,
  });
  for (let index = 0; index < config.candidates.length; index += 1) {
    const candidate = config.candidates[index];
    const candidateIndex = index + 1;
    dependencies.onProgress?.({
      type: 'candidate-started',
      candidateIndex,
      totalCandidates,
      candidateId: candidate.id,
      configLabel: searchConfigLabel(candidate.config),
      configProxyCost: configProxyCost(candidate.config),
    });
    const matchup = await runMatchup(
      {
        schemaVersion: 1,
        runLabel: `${config.runLabel}-${candidate.id}`,
        seedPrefix: config.seedPrefix,
        gamesPerSide: config.gamesPerSide,
        candidate,
        opponent: config.opponent,
        maxDecisionsPerGame: config.maxDecisionsPerGame,
      },
      {
        workers: execution.requestedWorkers,
        progressIntervalMs: dependencies.progressIntervalMs,
        onProgress(progress) {
          dependencies.onProgress?.({
            ...progress,
            candidateIndex,
            totalCandidates,
          });
        },
      }
    );
    matchups.push(matchup);
    await dependencies.onCandidateCompleted?.({
      run: partialRun(config, execution, matchups),
      candidateIndex,
      totalCandidates,
      matchup,
    });
    dependencies.onProgress?.({
      type: 'candidate-completed',
      candidateIndex,
      totalCandidates,
      candidateId: candidate.id,
      matchup,
      elapsedMs: now() - startedAt,
    });
  }
  const run = partialRun(config, execution, matchups);
  dependencies.onProgress?.({
    type: 'sweep-completed',
    candidates: totalCandidates,
    elapsedMs: now() - startedAt,
  });
  return run;
}

function partialRun(
  config: RolloutSearchSweepConfig,
  execution: RolloutSearchSweepRun['execution'],
  matchups: readonly HeadToHeadRun[]
): RolloutSearchSweepRun {
  return {
    config: structuredClone(config),
    execution: structuredClone(execution),
    matchups: [...matchups],
  };
}

function configProxyCost(
  config: RolloutSearchSweepConfig['candidates'][number]['config']
): number {
  return config.worlds * config.rollouts * config.maxRootActions * config.depth;
}

function searchConfigLabel(
  config: RolloutSearchSweepConfig['candidates'][number]['config']
): string {
  return `${config.worlds}w/${config.rollouts}r/${config.depth}d/${config.maxRootActions}a/e${config.rolloutEpsilon}`;
}
