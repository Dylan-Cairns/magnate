import { legalActions } from '../engine/actionBuilders';
import {
  toDecisionPlayerView,
  turnOwnerIdForState,
} from '../engine/decisionActor';
import {
  createDevFixtureSession,
  DEV_FIXTURES_ENABLED,
  type DevFixtureId,
} from '../dev/fixtures';
import { createSession } from '../engine/session';
import { newGame } from '../engine/game';
import { isTerminal } from '../engine/scoring';
import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
  PlayerView,
  Ruleset,
} from '../engine/types';
import { toPlayerView } from '../engine/view';
import { initialTurnCycleLogEntries } from './logTimeline';
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
  devFixtureId: DevFixtureId | null = null,
  ruleset: Ruleset = 'standard'
): GameState {
  // Both the parsed id and the static build flag must allow fixtures; keeping
  // the flag in this branch lets the bundler drop the fixture builders (and the
  // late-game rollout) entirely from the default build.
  if (devFixtureId && DEV_FIXTURES_ENABLED) {
    return createDevFixtureSession(devFixtureId, humanPlayerId);
  }
  return createSession(seed, humanPlayerId, ruleset);
}

export function rulesetLabel(ruleset: Ruleset): string {
  return ruleset === 'extended' ? 'Extended' : 'Standard';
}

export function withSeedLogPrefix(
  state: GameState,
  entries: readonly GameLogEntry[],
  fallbackPlayerId: PlayerId,
  botProfileLabel?: string
): ReadonlyArray<GameLogEntry> {
  const seedSummary = `Seed: ${state.seed}`;
  if (entries[0]?.summary === seedSummary) {
    return [...entries];
  }

  const prefix: GameLogEntry[] = [
    {
      turn: state.turn,
      player: activePlayerIdForState(state, fallbackPlayerId),
      phase: state.phase,
      summary: seedSummary,
    },
    {
      turn: state.turn,
      player: activePlayerIdForState(state, fallbackPlayerId),
      phase: state.phase,
      summary: `Ruleset: ${rulesetLabel(state.ruleset)}`,
      details: { ruleset: state.ruleset },
    },
  ];
  if (botProfileLabel) {
    prefix.push({
      turn: state.turn,
      player: activePlayerIdForState(state, fallbackPlayerId),
      phase: state.phase,
      summary: `Opponent: ${botProfileLabel}`,
    });
  }
  return [...prefix, ...entries];
}

export function initialBrowserTimelineLog(
  state: GameState,
  humanPlayerId: PlayerId,
  botProfileLabel: string
): ReadonlyArray<GameLogEntry> {
  const initialState = newGame(state.seed, {
    firstPlayer: humanPlayerId,
    ruleset: state.ruleset,
  });
  return withSeedLogPrefix(
    state,
    initialTurnCycleLogEntries(initialState, state, humanPlayerId),
    humanPlayerId,
    botProfileLabel
  );
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

export type BotDecisionPlan = {
  /** The bot owns the current decision and has a legal action to submit. */
  applicable: boolean;
  incomeChoicePhase: boolean;
  actions: readonly GameAction[];
  /** Turn owner for a normal decision; null during simultaneous income choices. */
  actingPlayerId: PlayerId | null;
  /** Decision view for the bot, or null when the bot cannot act. */
  view: PlayerView | null;
};

/**
 * Shape the bot's next decision from canonical state. Pure: it reads legality
 * and actor ownership but never mutates or selects randomness, so it can be
 * tested without a scheduler or policy.
 */
export function planBotDecision(
  state: GameState,
  botPlayerId: PlayerId
): BotDecisionPlan {
  const incomeChoicePhase = state.phase === 'CollectIncome';
  const actions = legalActions(state);
  const incomeChoiceActions = incomeChoiceActionsForPlayer(actions, botPlayerId);
  const actingPlayerId = incomeChoicePhase
    ? null
    : (turnOwnerIdForState(state) ?? null);
  const applicable =
    !isTerminal(state) &&
    (incomeChoicePhase
      ? incomeChoiceActions.length > 0
      : actingPlayerId === botPlayerId);

  if (!applicable) {
    return {
      applicable: false,
      incomeChoicePhase,
      actions: incomeChoicePhase ? incomeChoiceActions : actions,
      actingPlayerId,
      view: null,
    };
  }

  return {
    applicable: true,
    incomeChoicePhase,
    actions: incomeChoicePhase ? incomeChoiceActions : actions,
    actingPlayerId,
    view: incomeChoicePhase
      ? toDecisionPlayerView(state, botPlayerId)
      : toPlayerView(state, botPlayerId),
  };
}

/**
 * The player who should be recorded as acting for a bot selection: the income
 * choice owner for simultaneous income, otherwise the turn owner.
 */
export function resolveBotActingPlayerId(
  choice: GameAction,
  actingPlayerId: PlayerId | null,
  botPlayerId: PlayerId
): PlayerId {
  return choice.type === 'choose-income-suit'
    ? choice.playerId
    : (actingPlayerId ?? botPlayerId);
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
