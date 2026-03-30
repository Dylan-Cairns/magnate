import type { FinalScore, GamePhase, PlayerId } from '../engine/types';
import type { BotSpec, SearchBotSpec } from '../policies/botSpec';
import type { SearchDecisionDiagnostics } from '../policies/types';

export const HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION = 1;
export const HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION = 3;
export const HEAD_TO_HEAD_ARTIFACT_TYPE = 'ts-bot-head-to-head';
export const ROLLOUT_SEARCH_SWEEP_CONFIG_SCHEMA_VERSION = 1;
export const ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION = 2;
export const ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE = 'ts-rollout-search-sweep';
export const TD_REPLAY_CONFIG_SCHEMA_VERSION = 1;
export const TD_REPLAY_SUMMARY_SCHEMA_VERSION = 1;
export const TD_REPLAY_ARTIFACT_TYPE = 'ts-td-replay';
export const ROOT_ACTION_COUNT_BUCKETS = [
  '2-4',
  '5-8',
  '9-16',
  '17-32',
  '33-64',
  '65+',
] as const;

export type HeadToHeadArtifactSchemaVersion = 1 | 2 | 3;
export type RolloutSearchSweepArtifactSchemaVersion = 1 | 2;
export type RootActionCountBucket = (typeof ROOT_ACTION_COUNT_BUCKETS)[number];

export interface EvaluationExecution {
  requestedWorkers: number;
  workers: number;
  parallelUnit: 'paired-seed';
  latencyMode: 'isolated' | 'loaded';
}

export interface HeadToHeadConfig {
  schemaVersion: typeof HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION;
  runLabel: string;
  seedPrefix: string;
  gamesPerSide: number;
  candidate: BotSpec;
  opponent: BotSpec;
  maxDecisionsPerGame?: number;
}

export interface RolloutSearchSweepConfig {
  schemaVersion: typeof ROLLOUT_SEARCH_SWEEP_CONFIG_SCHEMA_VERSION;
  runLabel: string;
  seedPrefix: string;
  gamesPerSide: number;
  opponent: BotSpec;
  candidates: SearchBotSpec[];
  maxDecisionsPerGame?: number;
}

export interface TdReplayConfig {
  schemaVersion: typeof TD_REPLAY_CONFIG_SCHEMA_VERSION;
  runLabel: string;
  seedPrefix: string;
  games: number;
  playerA: BotSpec;
  playerB: BotSpec;
  maxDecisionsPerGame?: number;
}

export interface TdReplayValueTransitionPayload {
  observation: number[];
  reward: number;
  done: boolean;
  nextObservation: number[] | null;
  playerId: PlayerId;
  episodeId: string;
  timestep: number;
}

export interface TdReplayOpponentSamplePayload {
  observation: number[];
  actionFeatures: number[][];
  actionIndex: number;
  playerId: PlayerId;
}

export interface TdReplayDecisionRecord {
  decisionIndex: number;
  turn: number;
  phase: GamePhase;
  activePlayerId: PlayerId;
  botId: string;
  actionKey: string;
  actionIndex: number;
  indexedActionKey: string;
  legalActionCount: number;
}

export interface CollectedTdReplayGame {
  gameId: string;
  seed: string;
  firstPlayer: PlayerId;
  botBySeat: Record<PlayerId, string>;
  decisions: TdReplayDecisionRecord[];
  finalScore: FinalScore;
  turns: number;
  elapsedMs: number;
  valueTransitions: TdReplayValueTransitionPayload[];
  opponentSamples: TdReplayOpponentSamplePayload[];
}

export interface TdReplayRun {
  config: TdReplayConfig;
  games: CollectedTdReplayGame[];
  valueTransitions: TdReplayValueTransitionPayload[];
  opponentSamples: TdReplayOpponentSamplePayload[];
  elapsedMs: number;
}

export interface DecisionRecord {
  decisionIndex: number;
  turn: number;
  phase: GamePhase;
  activePlayerId: PlayerId;
  botId: string;
  actionKey: string;
  legalActionCount: number;
  latencyMs: number;
  searchDiagnostics?: SearchDecisionDiagnostics;
}

export interface PlayedGame {
  gameId: string;
  seed: string;
  firstPlayer: PlayerId;
  botBySeat: Record<PlayerId, string>;
  transcript: DecisionRecord[];
  finalScore: FinalScore;
  turns: number;
  elapsedMs: number;
}

export interface ConfidenceInterval {
  low: number;
  high: number;
}

export interface LatencySummary {
  actions: number;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  maxMs: number;
}

export interface SearchWorkSummary {
  searchedDecisions: number;
  rootVisits: number;
  configProxyCost: number;
  maxSimulatedActionSteps: number;
  simulatedActionSteps: number;
  stepUtilization: number;
  meanSimulatedActionSteps: number;
  terminalRollouts: number;
  terminalRate: number;
  meanSelectedActionValue: number;
  meanSelectedActionVisits: number;
  meanSelectedActionTerminalRate: number;
}

export interface HeadToHeadSummary {
  gamesPerSide: number;
  totalGames: number;
  candidateId: string;
  opponentId: string;
  candidateWins: number;
  opponentWins: number;
  draws: number;
  candidateWinRate: number;
  candidateWinRateCi95: ConfidenceInterval;
  candidateWinRateAsPlayerA: number;
  candidateWinRateAsPlayerB: number;
  candidateWinRateMovingFirst: number;
  candidateWinRateMovingSecond: number;
  sideGap: number;
  averageTurns: number;
  elapsedMs: number;
  gamesPerMinute: number;
  finalScoreDeciders: Record<FinalScore['decidedBy'], number>;
  latencyByBotId: Record<string, LatencySummary>;
  multiChoiceLatencyByBotId: Record<string, LatencySummary>;
  searchWorkByBotId: Record<string, SearchWorkSummary>;
  searchLatencyByRootActionCountByBotId: Record<
    string,
    Record<RootActionCountBucket, LatencySummary>
  >;
}

export interface HeadToHeadRun {
  config: HeadToHeadConfig;
  execution: EvaluationExecution;
  summary: HeadToHeadSummary;
  games: PlayedGame[];
}

export interface RolloutSearchSweepRun {
  config: RolloutSearchSweepConfig;
  execution: EvaluationExecution;
  matchups: HeadToHeadRun[];
}

export interface GitMetadata {
  commit: string | null;
  dirty: boolean | null;
}

export interface HeadToHeadArtifact {
  schemaVersion: HeadToHeadArtifactSchemaVersion;
  artifactType: typeof HEAD_TO_HEAD_ARTIFACT_TYPE;
  generatedAtUtc: string;
  policyRandomSchemeVersion: string;
  runtime: {
    nodeVersion: string;
  };
  git: GitMetadata;
  execution?: EvaluationExecution;
  config: HeadToHeadConfig;
  summary: HeadToHeadSummary;
  games: PlayedGame[];
}

export interface RolloutSearchSweepArtifactRow {
  candidate: SearchBotSpec;
  execution: EvaluationExecution;
  summary: HeadToHeadSummary;
  matchupArtifactPath: string;
}

export type RolloutSearchSweepArtifactStatus = 'running' | 'completed';

export interface RolloutSearchSweepArtifact {
  schemaVersion: RolloutSearchSweepArtifactSchemaVersion;
  artifactType: typeof ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE;
  generatedAtUtc: string;
  runtime: {
    nodeVersion: string;
  };
  git: GitMetadata;
  execution?: EvaluationExecution;
  status: RolloutSearchSweepArtifactStatus;
  completedCandidates: number;
  totalCandidates: number;
  config: RolloutSearchSweepConfig;
  rows: RolloutSearchSweepArtifactRow[];
}

export interface TdReplaySummaryGame {
  gameId: string;
  seed: string;
  firstPlayer: PlayerId;
  botBySeat: Record<PlayerId, string>;
  winner: FinalScore['winner'];
  finalScore: FinalScore;
  turns: number;
  decisions: number;
  valueTransitions: number;
  opponentSamples: number;
  elapsedMs: number;
}

export interface TdReplaySummary {
  schemaVersion: typeof TD_REPLAY_SUMMARY_SCHEMA_VERSION;
  artifactType: typeof TD_REPLAY_ARTIFACT_TYPE;
  generatedAtUtc: string;
  policyRandomSchemeVersion: string;
  runtime: {
    nodeVersion: string;
  };
  git: GitMetadata;
  config: TdReplayConfig;
  encoding: {
    encodingVersion: number;
    observationDim: number;
    actionFeatureDim: number;
  };
  results: {
    games: number;
    winners: Record<FinalScore['winner'], number>;
    averageTurns: number;
    decisions: number;
    valueTransitions: number;
    opponentSamples: number;
    elapsedMs: number;
  };
  artifacts: {
    valueTransitions: string;
    opponentSamples: string;
    summary: string;
  };
  games: TdReplaySummaryGame[];
}
