import type { FinalScore, GamePhase, PlayerId } from '../engine/types';
import type { BotSpec } from '../policies/botSpec';

export const HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION = 1;
export const HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION = 1;
export const HEAD_TO_HEAD_ARTIFACT_TYPE = 'ts-bot-head-to-head';

export interface HeadToHeadConfig {
  schemaVersion: typeof HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION;
  runLabel: string;
  seedPrefix: string;
  gamesPerSide: number;
  candidate: BotSpec;
  opponent: BotSpec;
  maxDecisionsPerGame?: number;
}

export interface DecisionRecord {
  decisionIndex: number;
  turn: number;
  phase: GamePhase;
  activePlayerId: PlayerId;
  botId: string;
  actionKey: string;
  latencyMs: number;
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
}

export interface HeadToHeadRun {
  config: HeadToHeadConfig;
  summary: HeadToHeadSummary;
  games: PlayedGame[];
}

export interface GitMetadata {
  commit: string | null;
  dirty: boolean | null;
}

export interface HeadToHeadArtifact {
  schemaVersion: typeof HEAD_TO_HEAD_ARTIFACT_SCHEMA_VERSION;
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
