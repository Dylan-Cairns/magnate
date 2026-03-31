import { newGame } from './game';
import { applyAction, applyKnownLegalAction } from './reducer';
import { advanceToDecision } from './turnFlow';
import type { GameAction, GameState, PlayerId } from './types';

export function createSession(seed: string, firstPlayer: PlayerId): GameState {
  return advanceToDecision(newGame(seed, { firstPlayer }));
}

export function stepToDecision(
  state: GameState,
  action: GameAction
): GameState {
  return advanceToDecision(applyAction(state, action));
}

export function stepKnownLegalActionToDecision(
  state: GameState,
  action: GameAction
): GameState {
  return advanceToDecision(applyKnownLegalAction(state, action), {
    assumeActionWindowDecision: true,
  });
}
