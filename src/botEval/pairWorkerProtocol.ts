import type { PlayGameHeartbeat } from './playGame';
import type { PairedSeedJob, PairedSeedResult } from './pair';
import type { HeadToHeadConfig, PlayedGame } from './types';

export type PairWorkerRequest =
  | {
      type: 'initialize';
      config: HeadToHeadConfig;
      progressIntervalMs: number;
    }
  | {
      type: 'run-pair';
      job: PairedSeedJob;
    }
  | {
      type: 'shutdown';
    };

export type PairWorkerResponse =
  | {
      type: 'ready';
    }
  | {
      type: 'heartbeat';
      pairIndex: number;
      heartbeat: PlayGameHeartbeat;
    }
  | {
      type: 'game-completed';
      pairIndex: number;
      game: PlayedGame;
    }
  | {
      type: 'pair-completed';
      result: PairedSeedResult;
    }
  | {
      type: 'error';
      pairIndex?: number;
      message: string;
      stack?: string;
    };
