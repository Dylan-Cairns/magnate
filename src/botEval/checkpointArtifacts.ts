import { mkdirSync, rmSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { requiredRecord, writeAtomicSync } from './artifactUtils';
import { parseHeadToHeadConfig } from './config';
import type { PairedSeedResult } from './pair';
import {
  HEAD_TO_HEAD_CHECKPOINT_SCHEMA_VERSION,
  HEAD_TO_HEAD_CHECKPOINT_TYPE,
  type HeadToHeadCheckpoint,
  type HeadToHeadConfig,
  type PlayedGame,
} from './types';

export const HEAD_TO_HEAD_CHECKPOINT_FILE_NAME = 'checkpoint.json';

export function headToHeadCheckpointPath(outputDirectory: string): string {
  return path.join(outputDirectory, HEAD_TO_HEAD_CHECKPOINT_FILE_NAME);
}

export function createHeadToHeadCheckpoint(
  config: HeadToHeadConfig,
  results: readonly PairedSeedResult[],
  elapsedMs: number
): HeadToHeadCheckpoint {
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
    throw new Error(
      'Head-to-head checkpoint elapsedMs must be a finite number >= 0.'
    );
  }
  return {
    schemaVersion: HEAD_TO_HEAD_CHECKPOINT_SCHEMA_VERSION,
    artifactType: HEAD_TO_HEAD_CHECKPOINT_TYPE,
    config: structuredClone(config),
    elapsedMs,
    results: results.map((result) => structuredClone(result)),
  };
}

export function writeHeadToHeadCheckpoint(
  checkpoint: HeadToHeadCheckpoint,
  checkpointPath: string
): void {
  mkdirSync(path.dirname(checkpointPath), { recursive: true });
  writeAtomicSync(checkpointPath, `${JSON.stringify(checkpoint, null, 2)}\n`);
}

export async function loadHeadToHeadCheckpoint(
  checkpointPath: string
): Promise<HeadToHeadCheckpoint> {
  const payload: unknown = JSON.parse(await readFile(checkpointPath, 'utf8'));
  const source = requiredRecord(payload, 'head-to-head checkpoint');
  if (source.schemaVersion !== HEAD_TO_HEAD_CHECKPOINT_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported head-to-head checkpoint schemaVersion=${String(source.schemaVersion)}.`
    );
  }
  if (source.artifactType !== HEAD_TO_HEAD_CHECKPOINT_TYPE) {
    throw new Error(
      `Unsupported head-to-head checkpoint type=${String(source.artifactType)}.`
    );
  }
  const config = parseHeadToHeadConfig(source.config);
  const elapsedMs = source.elapsedMs;
  if (
    typeof elapsedMs !== 'number' ||
    !Number.isFinite(elapsedMs) ||
    elapsedMs < 0
  ) {
    throw new Error(
      'head-to-head checkpoint.elapsedMs must be a finite number >= 0.'
    );
  }
  if (!Array.isArray(source.results)) {
    throw new Error('head-to-head checkpoint.results must be an array.');
  }
  return {
    schemaVersion: HEAD_TO_HEAD_CHECKPOINT_SCHEMA_VERSION,
    artifactType: HEAD_TO_HEAD_CHECKPOINT_TYPE,
    config,
    elapsedMs,
    results: source.results.map((result, index) =>
      parseCheckpointResult(result, index)
    ),
  };
}

export function clearHeadToHeadCheckpoint(checkpointPath: string): void {
  rmSync(checkpointPath, { force: true });
}

export function headToHeadConfigsMatch(
  left: HeadToHeadConfig,
  right: HeadToHeadConfig
): boolean {
  return canonicalJson(left) === canonicalJson(right);
}

function parseCheckpointResult(value: unknown, index: number): PairedSeedResult {
  const label = `head-to-head checkpoint.results[${String(index)}]`;
  const source = requiredRecord(value, label);
  const pairIndex = source.pairIndex;
  if (
    typeof pairIndex !== 'number' ||
    !Number.isInteger(pairIndex) ||
    pairIndex < 0
  ) {
    throw new Error(`${label}.pairIndex must be a nonnegative integer.`);
  }
  if (!Array.isArray(source.games)) {
    throw new Error(`${label}.games must be an array.`);
  }
  const games = source.games.map((game, gameIndex) =>
    parseCheckpointGame(game, `${label}.games[${String(gameIndex)}]`)
  );
  const candidateAsA = games[0];
  const candidateAsB = games[1];
  if (games.length !== 2 || !candidateAsA || !candidateAsB) {
    throw new Error(`${label}.games must contain exactly two played games.`);
  }
  return {
    pairIndex,
    games: [candidateAsA, candidateAsB],
  };
}

function parseCheckpointGame(value: unknown, label: string): PlayedGame {
  const source = requiredRecord(value, label);
  requiredCheckpointString(source.gameId, `${label}.gameId`);
  requiredCheckpointString(source.seed, `${label}.seed`);
  if (source.firstPlayer !== 'PlayerA' && source.firstPlayer !== 'PlayerB') {
    throw new Error(`${label}.firstPlayer must be PlayerA or PlayerB.`);
  }
  const botBySeat = requiredRecord(source.botBySeat, `${label}.botBySeat`);
  requiredCheckpointString(botBySeat.PlayerA, `${label}.botBySeat.PlayerA`);
  requiredCheckpointString(botBySeat.PlayerB, `${label}.botBySeat.PlayerB`);
  if (!Array.isArray(source.transcript)) {
    throw new Error(`${label}.transcript must be an array.`);
  }
  const finalScore = requiredRecord(source.finalScore, `${label}.finalScore`);
  if (
    finalScore.winner !== 'PlayerA' &&
    finalScore.winner !== 'PlayerB' &&
    finalScore.winner !== 'Draw'
  ) {
    throw new Error(`${label}.finalScore.winner is invalid.`);
  }
  if (
    finalScore.decidedBy !== 'districts' &&
    finalScore.decidedBy !== 'rank-total' &&
    finalScore.decidedBy !== 'resources' &&
    finalScore.decidedBy !== 'draw'
  ) {
    throw new Error(`${label}.finalScore.decidedBy is invalid.`);
  }
  requiredCheckpointNumber(source.turns, `${label}.turns`);
  requiredCheckpointNumber(source.elapsedMs, `${label}.elapsedMs`);
  return value as PlayedGame;
}

function requiredCheckpointString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function requiredCheckpointNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a finite number >= 0.`);
  }
  return value;
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value) ?? 'null';
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>).sort(
    ([left], [right]) => (left < right ? -1 : left > right ? 1 : 0)
  );
  return `{${entries
    .map(([key, entryValue]) => `${JSON.stringify(key)}:${canonicalJson(entryValue)}`)
    .join(',')}}`;
}
