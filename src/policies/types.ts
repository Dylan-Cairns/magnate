import type { GameAction, GameState, PlayerView } from '../engine/types';

export interface ActionSelectionContext {
  state: GameState;
  view: PlayerView;
  legalActions: readonly GameAction[];
  random: () => number;
}

export type MaybePromise<T> = T | Promise<T>;

export interface ActionPolicy {
  selectAction(context: ActionSelectionContext): MaybePromise<GameAction | undefined>;
}
