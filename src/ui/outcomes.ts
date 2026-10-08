export const CelebrationOutcome = {
  Win: 'win',
  Loss: 'loss',
  Draw: 'draw',
} as const;
export type CelebrationOutcome =
  (typeof CelebrationOutcome)[keyof typeof CelebrationOutcome];
