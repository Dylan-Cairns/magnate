import { runHeadToHead } from './matchup';
import type {
  HeadToHeadConfig,
  HeadToHeadRun,
  RolloutSearchSweepConfig,
  RolloutSearchSweepRun,
} from './types';

export interface RolloutSearchSweepDependencies {
  runMatchup?: (config: HeadToHeadConfig) => Promise<HeadToHeadRun>;
}

export async function runRolloutSearchSweep(
  config: RolloutSearchSweepConfig,
  dependencies: RolloutSearchSweepDependencies = {}
): Promise<RolloutSearchSweepRun> {
  const runMatchup = dependencies.runMatchup ?? runHeadToHead;
  const matchups: HeadToHeadRun[] = [];
  for (const candidate of config.candidates) {
    matchups.push(
      await runMatchup({
        schemaVersion: 1,
        runLabel: `${config.runLabel}-${candidate.id}`,
        seedPrefix: config.seedPrefix,
        gamesPerSide: config.gamesPerSide,
        candidate,
        opponent: config.opponent,
        maxDecisionsPerGame: config.maxDecisionsPerGame,
      })
    );
  }
  return {
    config: structuredClone(config),
    matchups,
  };
}
