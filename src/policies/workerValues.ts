// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const SearchWorkerExecutionMode = {
  Legacy: 'legacy',
  ResumableScalar: 'resumable-scalar',
  ResumablePairedTd: 'resumable-paired-td',
} as const;
export type SearchWorkerExecutionMode =
  (typeof SearchWorkerExecutionMode)[keyof typeof SearchWorkerExecutionMode];

export const BotWorkerMessageType = {
  SelectAction: 'select-action',
  Cancel: 'cancel',
  Shutdown: 'shutdown',
  SelectedAction: 'selected-action',
  Error: 'error',
} as const;
export type BotWorkerMessageType =
  (typeof BotWorkerMessageType)[keyof typeof BotWorkerMessageType];

export const SearchWorkerMessageType = {
  RunBatch: 'run-batch',
  InitializeRolloutSearch: 'initialize-rollout-search',
  Shutdown: 'shutdown',
  Initialized: 'initialized',
  BatchResult: 'batch-result',
  Error: 'error',
} as const;
export type SearchWorkerMessageType =
  (typeof SearchWorkerMessageType)[keyof typeof SearchWorkerMessageType];

export const EffectiveSearchExecutionMode = {
  ...SearchWorkerExecutionMode,
  Synchronous: 'synchronous',
} as const;
export type EffectiveSearchExecutionMode =
  (typeof EffectiveSearchExecutionMode)[keyof typeof EffectiveSearchExecutionMode];

export const TdSearchExecutor = {
  Legacy: 'legacy',
  Paired: 'paired',
} as const;
export type TdSearchExecutor =
  (typeof TdSearchExecutor)[keyof typeof TdSearchExecutor];
