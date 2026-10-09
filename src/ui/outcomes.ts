export const CelebrationOutcome = {
  Win: 'win',
  Draw: 'draw',
} as const;
export type CelebrationOutcome =
  (typeof CelebrationOutcome)[keyof typeof CelebrationOutcome];
