import type { PlayerId, Winner } from '../engine/types';

export function playerDisplayName(
  playerId: PlayerId,
  humanPlayerId: PlayerId
): string {
  return playerId === humanPlayerId ? 'You' : 'Bot';
}

export function winnerDisplayName(
  winner: Winner,
  humanPlayerId: PlayerId
): string {
  if (winner === 'Draw') return 'Draw';
  return playerDisplayName(winner, humanPlayerId);
}
