import { legalActions } from '../engine/actionBuilders';
import {
  createDevFixtureSession,
  type DevFixtureId,
} from '../dev/fixtures';
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
  actionCommitPending,
  allowHumanActionsWhileCommitPending,
}: {
  state: GameState;
  humanPlayerId: PlayerId;
  actionCommitPending: boolean;
  allowHumanActionsWhileCommitPending: boolean;
}): readonly GameAction[] {
  if (isTerminal(state)) {
    return [];
  }

  const actions = legalActions(state);
  if (state.phase === 'CollectIncome') {
    if (actionCommitPending && !allowHumanActionsWhileCommitPending) {
      return [];
    }
    return incomeChoiceActionsForPlayer(actions, humanPlayerId);
  }

  if (activePlayerIdForState(state, humanPlayerId) !== humanPlayerId) {
    return [];
  }
  if (!actionCommitPending) {
    return actions;
  }
  if (!allowHumanActionsWhileCommitPending) {
    return [];
  }
  return actions.filter((action) => action.type === 'choose-income-suit');
}

export function incomeChoiceActionsForPlayer(
  actions: readonly GameAction[],
  playerId: PlayerId
): readonly Extract<GameAction, { type: 'choose-income-suit' }>[] {
  return actions.filter(
    (
      action
    ): action is Extract<GameAction, { type: 'choose-income-suit' }> =>
      action.type === 'choose-income-suit' && action.playerId === playerId
  );
}

export function shouldScheduleBotAction({
  terminal,
  activePlayerId,
  botPlayerId,
  actionCommitPending,
  allowIncomeChoiceWhileCommitPending,
  botIncomeActionCount,
  startupPreloadReady,
}: {
  terminal: boolean;
  activePlayerId: PlayerId;
  botPlayerId: PlayerId;
  actionCommitPending: boolean;
  allowIncomeChoiceWhileCommitPending: boolean;
  botIncomeActionCount: number;
  startupPreloadReady: boolean;
}): boolean {
  const hasBotIncomeAction = botIncomeActionCount > 0;
  if (terminal || !startupPreloadReady) {
    return false;
  }
  if (
    actionCommitPending &&
    !(allowIncomeChoiceWhileCommitPending && hasBotIncomeAction)
  ) {
    return false;
  }
  return (
    activePlayerId === botPlayerId ||
    hasBotIncomeAction
  );
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
