import type { HeadToHeadConfig } from '../types';

export function testHeadToHeadConfig(gamesPerSide = 1): HeadToHeadConfig {
  return {
    schemaVersion: 1,
    runLabel: 'matchup-test',
    seedPrefix: 'matchup-test',
    gamesPerSide,
    candidate: {
      id: 'heuristic-candidate',
      kind: 'heuristic',
    },
    opponent: {
      id: 'random-opponent',
      kind: 'random',
    },
  };
}
