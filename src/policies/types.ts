import type { GameAction, PlayerView } from '../engine/types';

export interface ActionSelectionContext {
  view: PlayerView;
  legalActions: readonly GameAction[];
  random: () => number;
}

export type MaybePromise<T> = T | Promise<T>;

export interface ActionPolicy {
  selectAction(context: ActionSelectionContext): MaybePromise<GameAction | undefined>;
}
