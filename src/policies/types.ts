import type { GameAction, PlayerView } from '../engine/types';

export interface ActionSelectionContext {
  view: PlayerView;
  legalActions: readonly GameAction[];
  random: () => number;
}

export interface ActionPolicy {
  selectAction(context: ActionSelectionContext): GameAction | undefined;
}
