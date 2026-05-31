import { legalActions } from '../engine/actionBuilders';
import { rngFromSeed } from '../engine/rng';
import { createSession } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
} from '../engine/types';

export function makeBrowserSessionSeed(now = Date.now()): string {
  return `seed-${now}`;
}

export function createBrowserSession(
  seed: string,
  humanPlayerId: PlayerId
): GameState {
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

export function botRandomForState(
  state: GameState,
  profileId: string
): () => number {
  return rngFromSeed(
    `${state.seed}:bot:${profileId}:turn:${state.turn}:phase:${state.phase}:log:${state.log.length}:actor:${state.activePlayerIndex}`
  );
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
  if (
    isTerminal(state) ||
    activePlayerIdForState(state, humanPlayerId) !== humanPlayerId
  ) {
    return [];
  }

  const actions = legalActions(state);
  if (!actionCommitPending) {
    return actions;
  }
  if (!allowHumanActionsWhileCommitPending) {
    return [];
  }
  return actions.filter((action) => action.type === 'choose-income-suit');
}

export function shouldScheduleBotAction({
  terminal,
  activePlayerId,
  botPlayerId,
  actionCommitPending,
  startupPreloadReady,
}: {
  terminal: boolean;
  activePlayerId: PlayerId;
  botPlayerId: PlayerId;
  actionCommitPending: boolean;
  startupPreloadReady: boolean;
}): boolean {
  return (
    !terminal &&
    activePlayerId === botPlayerId &&
    !actionCommitPending &&
    startupPreloadReady
  );
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
