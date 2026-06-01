import type { GameAction, GameState, PlayerView } from '../engine/types';

export interface SearchDecisionDiagnostics {
  kind: 'search';
  legalRootActions: number;
  expandedRootActions: number;
  rootVisitBudget: number;
  configProxyCost: number;
  maxSimulatedActionSteps: number;
  simulatedActionSteps: number;
  terminalRollouts: number;
}

export interface ActionSelectionContext {
  state: GameState;
  view: PlayerView;
  legalActions: readonly GameAction[];
  random: () => number;
  onSearchDiagnostics?: (diagnostics: SearchDecisionDiagnostics) => void;
  onProgress?: () => void;
}

export type MaybePromise<T> = T | Promise<T>;

export interface ActionPolicy {
  selectAction(
    context: ActionSelectionContext
  ): MaybePromise<GameAction | undefined>;
}
