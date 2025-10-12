import type { GameState, PlayerId } from '../engine/types';

export interface TurnResetAnchor {
  turn: number;
  playerId: PlayerId;
  state: GameState;
}

export function shouldCaptureTurnResetAnchor(
  state: GameState,
  activePlayerId: PlayerId,
  humanPlayerId: PlayerId,
  anchor: TurnResetAnchor | null
): boolean {
  if (activePlayerId !== humanPlayerId) {
    return false;
  }
  if (state.phase !== 'ActionWindow') {
    return false;
  }
  if (state.cardPlayedThisTurn) {
    return false;
  }
  if (!anchor) {
    return true;
  }
  return anchor.turn !== state.turn || anchor.playerId !== activePlayerId;
}

export function canUseTurnReset(
  state: GameState,
  activePlayerId: PlayerId,
  humanPlayerId: PlayerId,
  anchor: TurnResetAnchor | null
): boolean {
  if (!anchor) {
    return false;
  }
  if (activePlayerId !== humanPlayerId) {
    return false;
  }
  if (state.phase !== 'ActionWindow') {
    return false;
  }
  if (anchor.turn !== state.turn || anchor.playerId !== activePlayerId) {
    return false;
  }
  return state !== anchor.state;
}
