import { legalActions } from './actionBuilders';
import type { GameAction, GameState, PlayerId, PlayerView } from './types';
import { toPlayerView } from './view';

export function turnOwnerIdForState(state: GameState): PlayerId | undefined {
  return state.players[state.activePlayerIndex]?.id;
}

export function decisionPlayerIdForState(
  state: GameState
): PlayerId | undefined {
  if (state.phase === 'CollectIncome') {
    const pendingChoice = state.pendingIncomeChoices?.find(
      (choice) =>
        !state.submittedIncomeChoices?.some(
          (submitted) =>
            submitted.playerId === choice.playerId &&
            submitted.districtId === choice.districtId &&
            submitted.cardId === choice.cardId
        )
    );
    if (pendingChoice) {
      return pendingChoice.playerId;
    }
  }

  return turnOwnerIdForState(state);
}

export function legalActionsForDecisionPlayer(
  state: GameState,
  playerId = decisionPlayerIdForState(state)
): readonly GameAction[] {
  const actions = legalActions(state);
  if (state.phase !== 'CollectIncome' || playerId === undefined) {
    return actions;
  }

  return actions.filter(
    (action) =>
      action.type === 'choose-income-suit' && action.playerId === playerId
  );
}

export function toDecisionPlayerView(
  state: GameState,
  playerId = decisionPlayerIdForState(state)
): PlayerView {
  if (playerId === undefined) {
    throw new Error('Could not resolve decision player for view.');
  }

  const view = toPlayerView(state, playerId);
  return view.activePlayerId === playerId
    ? view
    : {
        ...view,
        activePlayerId: playerId,
      };
}
