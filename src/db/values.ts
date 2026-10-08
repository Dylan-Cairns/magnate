export const WinnerOutcome = {
  Player: 'player',
  Bot: 'bot',
  Draw: 'draw',
} as const;
export type WinnerOutcome = (typeof WinnerOutcome)[keyof typeof WinnerOutcome];
