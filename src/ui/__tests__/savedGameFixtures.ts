import { PlayerId, ActionId } from '../../engine/values';
import { legalActions } from '../../engine/actionBuilders';
import { createSession, stepToDecision } from '../../engine/session';
import { type GameAction, Ruleset } from '../../engine/types';
import {
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from '../../policies/catalog';
import { initialBrowserTimelineLog } from '../gameControllerModel';
import { transitionLogUpdate } from '../logTimeline';
import type { SavedGame } from '../savedGame';

export function initialSave(
  seed = 'save-test',
  ruleset: Ruleset = Ruleset.Standard
): SavedGame {
  const state = createSession(seed, PlayerId.PlayerA, ruleset);
  return {
    version: 1,
    gameId: 'test-session',
    humanPlayerId: PlayerId.PlayerA,
    botProfileId: DEFAULT_BOT_PROFILE_ID,
    state,
    timelineLog: initialBrowserTimelineLog(
      state,
      PlayerId.PlayerA,
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
    PlayerId.PlayerA,
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
          action.type === ActionId.ChooseIncomeSuit
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
    actions.find((a) => a.type === ActionId.EndTurn) ??
    actions.find((a) => a.type === ActionId.ChooseIncomeSuit) ??
    actions.find((a) => a.type === ActionId.BuyDeed) ??
    actions.find((a) => a.type === ActionId.SellCard);
  if (!action) throw new Error('No test action');
  return action;
}
