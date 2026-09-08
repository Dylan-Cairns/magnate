import { legalActions } from '../../engine/actionBuilders';
import { createSession, stepToDecision } from '../../engine/session';
import type { GameAction } from '../../engine/types';
import {
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from '../../policies/catalog';
import { initialBrowserTimelineLog } from '../gameControllerModel';
import { transitionLogUpdate } from '../logTimeline';
import type { SavedGame } from '../savedGame';

export function initialSave(seed = 'save-test'): SavedGame {
  const state = createSession(seed, 'PlayerA');
  return {
    version: 1,
    gameId: 'test-session',
    humanPlayerId: 'PlayerA',
    botProfileId: DEFAULT_BOT_PROFILE_ID,
    state,
    timelineLog: initialBrowserTimelineLog(
      state,
      'PlayerA',
      resolveBotProfile(DEFAULT_BOT_PROFILE_ID).selected.label
    ),
    actionHistory: [],
    deferredIncomeLogContext: null,
  };
}

export function advanceSave(save: SavedGame, action: GameAction): SavedGame {
  const state = stepToDecision(save.state, action);
  const timeline = transitionLogUpdate(
    save.state,
    state,
    action,
    'PlayerA',
    save.deferredIncomeLogContext
  );
  return {
    ...save,
    state,
    timelineLog: [...save.timelineLog, ...timeline.entries],
    deferredIncomeLogContext: timeline.deferredIncomeLogContext,
    actionHistory: [
      ...save.actionHistory,
      {
        turn: save.state.turn,
        phase: save.state.phase,
        actingPlayerId:
          action.type === 'choose-income-suit'
            ? action.playerId
            : save.state.players[save.state.activePlayerIndex].id,
        action,
      },
    ],
  };
}

// Cheap deterministic play with incomplete deeds to exercise shared income windows.
export function nextTestAction(save: SavedGame): GameAction {
  const actions = legalActions(save.state);
  const action =
    actions.find((a) => a.type === 'end-turn') ??
    actions.find((a) => a.type === 'choose-income-suit') ??
    actions.find((a) => a.type === 'buy-deed') ??
    actions.find((a) => a.type === 'sell-card');
  if (!action) throw new Error('No test action');
  return action;
}
