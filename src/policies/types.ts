import type { GameAction, GameState, PlayerView } from '../engine/types';
import type { SearchHeuristicVersion } from './searchConfig';

export interface SearchRootActionDiagnostics {
  actionKey: string;
  visits: number;
  meanValue: number;
  terminalRollouts: number;
  terminalRate: number;
  prior: number;
}

export interface SearchDecisionDiagnostics {
  kind: 'search';
  heuristic?: SearchHeuristicVersion;
  stochasticSimulation?: 'common-random-scenarios-v1';
  legalRootActions: number;
  expandedRootActions: number;
  rootVisitBudget: number;
  configProxyCost: number;
  maxSimulatedActionSteps: number;
  simulatedActionSteps: number;
  terminalRollouts: number;
  terminalRate: number;
  selectedActionKey: string;
  selectedActionVisits: number;
  selectedActionMeanValue: number;
  selectedActionTerminalRollouts: number;
  selectedActionTerminalRate: number;
  rootActions: readonly SearchRootActionDiagnostics[];
  parallelWorkers?: number;
  parallelBatches?: number;
  parallelBatchSize?: number;
}

export interface ActionSelectionContext {
  state: GameState;
  view: PlayerView;
  legalActions: readonly GameAction[];
  random: () => number;
  randomSeed?: string;
  onSearchDiagnostics?: (diagnostics: SearchDecisionDiagnostics) => void;
  onProgress?: () => void;
}

export type MaybePromise<T> = T | Promise<T>;

export interface ActionPolicy {
  selectAction(
    context: ActionSelectionContext
  ): MaybePromise<GameAction | undefined>;
}
