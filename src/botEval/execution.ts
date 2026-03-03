import type { EvaluationExecution } from './types';

export function resolveEvaluationExecution(
  requestedWorkers: number,
  gamesPerSide: number
): EvaluationExecution {
  validateWorkerCount(requestedWorkers);
  if (!Number.isInteger(gamesPerSide) || gamesPerSide <= 0) {
    throw new Error('gamesPerSide must be a positive integer.');
  }

  const workers = Math.min(requestedWorkers, gamesPerSide);
  return {
    requestedWorkers,
    workers,
    parallelUnit: 'paired-seed',
    latencyMode: workers === 1 ? 'isolated' : 'loaded',
  };
}

export function validateWorkerCount(workers: number): void {
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new Error('workers must be a positive integer.');
  }
}
