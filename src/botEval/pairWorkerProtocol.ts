import type { PlayGameHeartbeat } from './playGame';
import type { PairedSeedJob, PairedSeedResult } from './pair';
import type { HeadToHeadConfig } from './types';

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
      type: 'pair-completed';
      result: PairedSeedResult;
    }
  | {
      type: 'error';
      pairIndex?: number;
      message: string;
      stack?: string;
    };
