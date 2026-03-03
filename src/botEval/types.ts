import type { FinalScore, GamePhase, PlayerId } from '../engine/types';
import type { BotSpec, SearchBotSpec } from '../policies/botSpec';
import type { SearchDecisionDiagnostics } from '../policies/types';

export const HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION = 1;
export const HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION = 2;
export const HEAD_TO_HEAD_ARTIFACT_TYPE = 'ts-bot-head-to-head';
export const ROLLOUT_SEARCH_SWEEP_CONFIG_SCHEMA_VERSION = 1;
export const ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION = 1;
export const ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE = 'ts-rollout-search-sweep';
export const ROOT_ACTION_COUNT_BUCKETS = [
  '2-4',
  '5-8',
  '9-16',
  '17-32',
  '33-64',
  '65+',
] as const;

export type HeadToHeadArtifactSchemaVersion = 1 | 2;
export type RootActionCountBucket = (typeof ROOT_ACTION_COUNT_BUCKETS)[number];

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
  summary: HeadToHeadSummary;
  games: PlayedGame[];
}

export interface RolloutSearchSweepRun {
  config: RolloutSearchSweepConfig;
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
  config: HeadToHeadConfig;
  summary: HeadToHeadSummary;
  games: PlayedGame[];
}

export interface RolloutSearchSweepArtifactRow {
  candidate: SearchBotSpec;
  summary: HeadToHeadSummary;
  matchupArtifactPath: string;
}

export type RolloutSearchSweepArtifactStatus = 'running' | 'completed';

export interface RolloutSearchSweepArtifact {
  schemaVersion: typeof ROLLOUT_SEARCH_SWEEP_ARTIFACT_SCHEMA_VERSION;
  artifactType: typeof ROLLOUT_SEARCH_SWEEP_ARTIFACT_TYPE;
  generatedAtUtc: string;
  runtime: {
    nodeVersion: string;
  };
  git: GitMetadata;
  status: RolloutSearchSweepArtifactStatus;
  completedCandidates: number;
  totalCandidates: number;
  config: RolloutSearchSweepConfig;
  rows: RolloutSearchSweepArtifactRow[];
}
