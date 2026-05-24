import type { TdReplayProgress } from './tdReplay';
import type {
  GitMetadata,
  TdReplayConfig,
  TdReplaySummary,
} from './types';

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
      type: 'run-shard';
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
      type: 'shutdown';
    };

export type TdReplayShardWorkerResponse =
  | {
      type: 'ready';
    }
  | {
      type: 'progress';
      shardIndex: number;
      progress: TdReplayProgress;
    }
  | {
      type: 'shard-completed';
      result: TdReplayShardResult;
    }
  | {
      type: 'error';
      shardIndex?: number;
      message: string;
      stack?: string;
    };
