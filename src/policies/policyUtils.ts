import type { GameAction } from '../engine/types';
import { CARD_BY_ID } from '../engine/cards';
import { ActionId, CardKind } from '../engine/values';

export function botCandidateActions(
  actions: readonly GameAction[]
): GameAction[] {
  // Buying a rank-2 deed spends the outright-build cost without completing it.
  return actions.filter((action) => {
    if (action.type !== ActionId.BuyDeed) return true;
    const card = CARD_BY_ID[action.cardId];
    return card.kind !== CardKind.Property || card.rank !== 2;
  });
}

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
