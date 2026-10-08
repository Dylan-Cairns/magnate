// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const PairWorkerMessageType = {
  Initialize: 'initialize',
  RunPair: 'run-pair',
  Shutdown: 'shutdown',
  Ready: 'ready',
  Heartbeat: 'heartbeat',
  GameCompleted: 'game-completed',
  PairCompleted: 'pair-completed',
  Error: 'error',
} as const;
export type PairWorkerMessageType =
  (typeof PairWorkerMessageType)[keyof typeof PairWorkerMessageType];

export const TdReplayShardWorkerMessageType = {
  RunShard: 'run-shard',
  Shutdown: 'shutdown',
  Ready: 'ready',
  Progress: 'progress',
  ShardCompleted: 'shard-completed',
  Error: 'error',
} as const;
export type TdReplayShardWorkerMessageType =
  (typeof TdReplayShardWorkerMessageType)[keyof typeof TdReplayShardWorkerMessageType];
