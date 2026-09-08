import { createSession, stepToDecision } from '../engine/session';
import { isTerminal } from '../engine/scoring';
import type { GameLogEntry, GameState, PlayerId } from '../engine/types';
import { resolveBotProfile, type BotProfileId } from '../policies/catalog';
import type { BugReportActionEntry } from './bugReport';
import {
  humanDecisionWindowKeyForState,
  transitionOpensHumanDecisionWindow,
} from './gameControllerModel';
import type { DeferredIncomeLogContext } from './logTimeline';
import { transitionLogUpdate } from './logTimeline';

export const SAVED_GAME_KEY = 'magnate:savedGame';

export interface SavedGame {
  version: 1;
  gameId: string;
  humanPlayerId: PlayerId;
  botProfileId: BotProfileId;
  state: GameState;
  timelineLog: readonly GameLogEntry[];
  actionHistory: readonly BugReportActionEntry[];
  deferredIncomeLogContext: DeferredIncomeLogContext | null;
}

export type SavedGameLoad = {
  save: SavedGame | null;
  error: string | null;
};

// Compare JSON data independently of object key order and omitted undefined fields.
function canonicalJson(value: unknown): string {
  return JSON.stringify(value, (_key, entry: unknown) => {
    if (typeof entry !== 'object' || entry === null || Array.isArray(entry))
      return entry;
    return Object.fromEntries(
      Object.entries(entry).sort(([a], [b]) => a.localeCompare(b))
    );
  });
}

export function parseSavedGame(
  raw: string,
  humanPlayerId: PlayerId
): SavedGame {
  const save = JSON.parse(raw) as SavedGame;
  if (
    !save ||
    save.version !== 1 ||
    typeof save.gameId !== 'string' ||
    !save.gameId ||
    save.humanPlayerId !== humanPlayerId ||
    typeof save.state?.seed !== 'string' ||
    !Array.isArray(save.actionHistory) ||
    !Array.isArray(save.timelineLog)
  ) {
    throw new Error('Invalid or incompatible saved game.');
  }
  resolveBotProfile(save.botProfileId);
  let state = createSession(save.state.seed, humanPlayerId);
  let previousState: GameState | null = null;
  let deferredIncomeLogContext: DeferredIncomeLogContext | null = null;
  for (const entry of save.actionHistory) {
    if (
      !entry ||
      entry.turn !== state.turn ||
      entry.phase !== state.phase ||
      entry.actingPlayerId !==
        (entry.action?.type === 'choose-income-suit'
          ? entry.action.playerId
          : state.players[state.activePlayerIndex]?.id)
    ) {
      throw new Error('Invalid saved action history.');
    }
    previousState = state;
    state = stepToDecision(state, entry.action);
    deferredIncomeLogContext = transitionLogUpdate(
      previousState,
      state,
      entry.action,
      humanPlayerId,
      deferredIncomeLogContext
    ).deferredIncomeLogContext;
  }
  if (
    canonicalJson(state) !== canonicalJson(save.state) ||
    canonicalJson(deferredIncomeLogContext) !==
      canonicalJson(save.deferredIncomeLogContext) ||
    (!isTerminal(state) &&
      (!humanDecisionWindowKeyForState(state, humanPlayerId) ||
        (previousState &&
          !transitionOpensHumanDecisionWindow(
            previousState,
            state,
            humanPlayerId
          ))))
  ) {
    throw new Error(
      'Saved game does not match a supported decision checkpoint.'
    );
  }
  const phases = [
    'StartTurn',
    'TaxCheck',
    'CollectIncome',
    'ActionWindow',
    'DrawCard',
    'GameOver',
  ];
  if (
    save.timelineLog.some(
      (entry) =>
        !entry ||
        typeof entry.summary !== 'string' ||
        !Number.isSafeInteger(entry.turn) ||
        !phases.includes(entry.phase) ||
        !['PlayerA', 'PlayerB'].includes(entry.player)
    )
  ) {
    throw new Error('Invalid saved timeline.');
  }
  return { ...save, state, deferredIncomeLogContext };
}

export function loadSavedGame(humanPlayerId: PlayerId): SavedGameLoad {
  if (typeof window === 'undefined') return { save: null, error: null };
  try {
    const raw = window.localStorage.getItem(SAVED_GAME_KEY);
    return {
      save: raw === null ? null : parseSavedGame(raw, humanPlayerId),
      error: null,
    };
  } catch {
    return {
      save: null,
      error:
        'Your saved game could not be loaded. Autosave is paused to preserve it. Choose New Game to replace it.',
    };
  }
}

export function writeSavedGame(save: SavedGame): string | null {
  if (typeof window === 'undefined') return null;
  try {
    window.localStorage.setItem(SAVED_GAME_KEY, JSON.stringify(save));
    return null;
  } catch {
    return 'Your progress could not be saved in this browser. Reloading may lose recent progress.';
  }
}
