import { legalActions } from '../engine/actionBuilders';
import { createDevFixtureSession, type DevFixtureId } from '../dev/fixtures';
import { createSession } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
} from '../engine/types';
export {
  policyRandomForState as botRandomForState,
  policyRandomSeedForState as botRandomSeedForState,
} from '../policies/policyRandom';

export function makeBrowserSessionSeed(now = Date.now()): string {
  return `seed-${now}`;
}

export function createBrowserSession(
  seed: string,
  humanPlayerId: PlayerId,
  devFixtureId: DevFixtureId | null = null
): GameState {
  if (devFixtureId && import.meta.env.DEV) {
    return createDevFixtureSession(devFixtureId, humanPlayerId);
  }
  return createSession(seed, humanPlayerId);
}

export function withSeedLogPrefix(
  state: GameState,
  entries: readonly GameLogEntry[],
  fallbackPlayerId: PlayerId
): ReadonlyArray<GameLogEntry> {
  const seedSummary = `Seed ${state.seed}`;
  if (entries[0]?.summary === seedSummary) {
    return [...entries];
  }

  return [
    {
      turn: state.turn,
      player: activePlayerIdForState(state, fallbackPlayerId),
      phase: state.phase,
      summary: seedSummary,
    },
    ...entries,
  ];
}

export function activePlayerIdForState(
  state: GameState,
  fallbackPlayerId: PlayerId
): PlayerId {
  return state.players[state.activePlayerIndex]?.id ?? fallbackPlayerId;
}

export function humanActionsAcceptingInputForState({
  state,
  humanPlayerId,
  humanInputReady,
}: {
  state: GameState;
  humanPlayerId: PlayerId;
  humanInputReady: boolean;
}): readonly GameAction[] {
  if (isTerminal(state) || !humanInputReady) {
    return [];
  }

  const actions = legalActions(state);
  if (state.phase === 'CollectIncome') {
    return incomeChoiceActionsForPlayer(actions, humanPlayerId);
  }

  if (activePlayerIdForState(state, humanPlayerId) !== humanPlayerId) {
    return [];
  }
  return actions;
}

export function humanDecisionWindowKeyForState(
  state: GameState,
  humanPlayerId: PlayerId
): string | null {
  if (isTerminal(state)) {
    return null;
  }

  const actions = legalActions(state);
  if (state.phase === 'CollectIncome') {
    return incomeChoiceActionsForPlayer(actions, humanPlayerId).length > 0
      ? `income:${String(state.turn)}:${humanPlayerId}`
      : null;
  }
  return activePlayerIdForState(state, humanPlayerId) === humanPlayerId &&
    actions.length > 0
    ? `action:${String(state.turn)}:${humanPlayerId}`
    : null;
}

export function transitionOpensHumanDecisionWindow(
  previousState: GameState,
  nextState: GameState,
  humanPlayerId: PlayerId
): boolean {
  const previousWindow = humanDecisionWindowKeyForState(
    previousState,
    humanPlayerId
  );
  const nextWindow = humanDecisionWindowKeyForState(nextState, humanPlayerId);
  return nextWindow !== null && nextWindow !== previousWindow;
}

export function incomeChoiceActionsForPlayer(
  actions: readonly GameAction[],
  playerId: PlayerId
): readonly Extract<GameAction, { type: 'choose-income-suit' }>[] {
  return actions.filter(
    (action): action is Extract<GameAction, { type: 'choose-income-suit' }> =>
      action.type === 'choose-income-suit' && action.playerId === playerId
  );
}

export function shouldScheduleBotAction({
  terminal,
  activePlayerId,
  botPlayerId,
  isIncomeChoicePhase,
  botIncomeActionCount,
  startupPreloadReady,
}: {
  terminal: boolean;
  activePlayerId: PlayerId;
  botPlayerId: PlayerId;
  isIncomeChoicePhase: boolean;
  botIncomeActionCount: number;
  startupPreloadReady: boolean;
}): boolean {
  const hasBotIncomeAction = botIncomeActionCount > 0;
  if (terminal || !startupPreloadReady) {
    return false;
  }
  if (isIncomeChoicePhase) {
    return hasBotIncomeAction;
  }
  return activePlayerId === botPlayerId;
}

export function botDecisionResultIsCurrent({
  cancelled,
  decisionGeneration,
  currentGeneration,
  decisionState,
  currentState,
}: {
  cancelled: boolean;
  decisionGeneration: number;
  currentGeneration: number;
  decisionState: GameState;
  currentState: GameState;
}): boolean {
  return (
    !cancelled &&
    decisionGeneration === currentGeneration &&
    decisionState === currentState
  );
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
