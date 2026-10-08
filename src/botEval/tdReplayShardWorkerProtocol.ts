import { TdReplayShardWorkerMessageType } from './workerValues';
import type { TdReplayProgress } from './tdReplay';
import type { GitMetadata, TdReplayConfig, TdReplaySummary } from './types';

export interface TdReplayShardPlan {
  shardIndex: number;
  gameIndexStart: number;
  games: number;
}

export interface TdReplayShardWrittenArtifacts {
  summary: TdReplaySummary;
  valuePath: string;
  opponentPath: string;
  summaryPath: string;
}

export interface TdReplayShardResult {
  shard: TdReplayShardPlan;
  written: TdReplayShardWrittenArtifacts;
}

export type TdReplayShardWorkerRequest =
  | {
      type: typeof TdReplayShardWorkerMessageType.RunShard;
      config: TdReplayConfig;
      shard: TdReplayShardPlan;
      gameIndexTotal: number;
      outputDirectory: string;
      progressIntervalMs: number;
      generatedAtUtc: string;
      git: GitMetadata;
      nodeVersion: string;
    }
  | {
      type: typeof TdReplayShardWorkerMessageType.Shutdown;
    };

export type TdReplayShardWorkerResponse =
  | {
      type: typeof TdReplayShardWorkerMessageType.Ready;
    }
  | {
      type: typeof TdReplayShardWorkerMessageType.Progress;
      shardIndex: number;
      progress: TdReplayProgress;
    }
  | {
      type: typeof TdReplayShardWorkerMessageType.ShardCompleted;
      result: TdReplayShardResult;
    }
  | {
      type: typeof TdReplayShardWorkerMessageType.Error;
      shardIndex?: number;
      message: string;
      stack?: string;
    };
