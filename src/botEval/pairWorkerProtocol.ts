import { PairWorkerMessageType } from './workerValues';
import type { PlayGameHeartbeat } from './playGame';
import type { PairedSeedJob, PairedSeedResult } from './pair';
import type { HeadToHeadConfig, PlayedGame } from './types';

export type PairWorkerRequest =
  | {
      type: typeof PairWorkerMessageType.Initialize;
      config: HeadToHeadConfig;
      progressIntervalMs: number;
    }
  | {
      type: typeof PairWorkerMessageType.RunPair;
      job: PairedSeedJob;
    }
  | {
      type: typeof PairWorkerMessageType.Shutdown;
    };

export type PairWorkerResponse =
  | {
      type: typeof PairWorkerMessageType.Ready;
    }
  | {
      type: typeof PairWorkerMessageType.Heartbeat;
      pairIndex: number;
      heartbeat: PlayGameHeartbeat;
    }
  | {
      type: typeof PairWorkerMessageType.GameCompleted;
      pairIndex: number;
      game: PlayedGame;
    }
  | {
      type: typeof PairWorkerMessageType.PairCompleted;
      result: PairedSeedResult;
    }
  | {
      type: typeof PairWorkerMessageType.Error;
      pairIndex?: number;
      message: string;
      stack?: string;
    };
