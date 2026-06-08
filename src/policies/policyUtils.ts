import type { GameAction } from '../engine/types';

export function cachedAsyncLoader<T>(
  loader: () => Promise<T>
): () => Promise<T> {
  let promise: Promise<T> | null = null;
  return () => {
    if (promise === null) {
      promise = loader();
    }
    return promise;
  };
}

export function forcedAction(
  candidateActions: readonly GameAction[]
): GameAction | undefined | null {
  if (candidateActions.length === 0) {
    return undefined;
  }
  if (candidateActions.length === 1) {
    return candidateActions[0];
  }
  return null;
}
