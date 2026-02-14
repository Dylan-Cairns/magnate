import { newGame } from './game';
import { applyAction } from './reducer';
import { advanceToDecision } from './turnFlow';
import type { GameAction, GameState, PlayerId } from './types';

export function createSession(seed: string, firstPlayer: PlayerId): GameState {
  return advanceToDecision(newGame(seed, { firstPlayer }));
}

export function stepToDecision(state: GameState, action: GameAction): GameState {
  return advanceToDecision(applyAction(state, action));
}
