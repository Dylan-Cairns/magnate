import { mkdir, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import {
  ACTION_FEATURE_DIM,
  ENCODING_VERSION,
  OBSERVATION_DIM,
} from '../policies/trainingEncoding';
import { collectGitMetadata } from './gitMetadata';
import {
  TD_REPLAY_ARTIFACT_TYPE,
  TD_REPLAY_SUMMARY_SCHEMA_VERSION,
  type TdReplayOpponentSamplePayload,
  type TdReplayRun,
  type TdReplaySummary,
  type TdReplayValueTransitionPayload,
} from './types';

export interface TdReplayArtifactOptions {
  cwd?: string;
  generatedAtUtc?: string;
  git?: TdReplaySummary['git'];
  nodeVersion?: string;
  runBaseName?: string;
}

export interface WrittenTdReplayArtifacts {
  summary: TdReplaySummary;
  valuePath: string;
  opponentPath: string;
  summaryPath: string;
}

export async function writeTdReplayArtifacts(
  run: TdReplayRun,
  outputDirectory: string,
  options: TdReplayArtifactOptions = {}
): Promise<WrittenTdReplayArtifacts> {
  const generatedAtUtc = options.generatedAtUtc ?? new Date().toISOString();
  const runBaseName =
    options.runBaseName ??
    defaultTdReplayRunBaseName(run.config.runLabel, generatedAtUtc);
  const valuePath = path.join(outputDirectory, `${runBaseName}.value.jsonl`);
  const opponentPath = path.join(
    outputDirectory,
    `${runBaseName}.opponent.jsonl`
  );
  const summaryPath = path.join(outputDirectory, `${runBaseName}.summary.json`);
  const summary = createTdReplaySummary(run, {
    ...options,
    generatedAtUtc,
    artifacts: {
      valueTransitions: valuePath,
      opponentSamples: opponentPath,
      summary: summaryPath,
    },
  });

  await mkdir(outputDirectory, { recursive: true });
  await writeAtomic(valuePath, renderValueTransitionsJsonl(run.valueTransitions));
  await writeAtomic(opponentPath, renderOpponentSamplesJsonl(run.opponentSamples));
  await writeAtomic(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  return {
    summary,
    valuePath,
    opponentPath,
    summaryPath,
  };
}

export function createTdReplaySummary(
  run: TdReplayRun,
  options: TdReplayArtifactOptions & {
    artifacts: TdReplaySummary['artifacts'];
  }
): TdReplaySummary {
  const winners: TdReplaySummary['results']['winners'] = {
    PlayerA: 0,
    PlayerB: 0,
    Draw: 0,
  };
  let turnTotal = 0;
  let decisionTotal = 0;
  for (const game of run.games) {
    winners[game.finalScore.winner] += 1;
    turnTotal += game.turns;
    decisionTotal += game.decisions.length;
  }

  return {
    schemaVersion: TD_REPLAY_SUMMARY_SCHEMA_VERSION,
    artifactType: TD_REPLAY_ARTIFACT_TYPE,
    generatedAtUtc: options.generatedAtUtc ?? new Date().toISOString(),
    policyRandomSchemeVersion: POLICY_RANDOM_SCHEME_VERSION,
    runtime: {
      nodeVersion: options.nodeVersion ?? process.version,
    },
    git: options.git ?? collectGitMetadata(options.cwd),
    config: structuredClone(run.config),
    encoding: {
      encodingVersion: ENCODING_VERSION,
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
    },
    results: {
      games: run.games.length,
      winners,
      averageTurns: run.games.length > 0 ? turnTotal / run.games.length : 0,
      decisions: decisionTotal,
      valueTransitions: run.valueTransitions.length,
      opponentSamples: run.opponentSamples.length,
      elapsedMs: run.elapsedMs,
    },
    artifacts: structuredClone(options.artifacts),
    games: run.games.map((game) => ({
      gameId: game.gameId,
      seed: game.seed,
      firstPlayer: game.firstPlayer,
      botBySeat: structuredClone(game.botBySeat),
      winner: game.finalScore.winner,
      finalScore: structuredClone(game.finalScore),
      turns: game.turns,
      decisions: game.decisions.length,
      valueTransitions: game.valueTransitions.length,
      opponentSamples: game.opponentSamples.length,
      elapsedMs: game.elapsedMs,
    })),
  };
}

export function defaultTdReplayOutputDirectory(): string {
  return path.join('artifacts', 'td_replay');
}

export function defaultTdReplayRunBaseName(
  runLabel: string,
  generatedAtUtc = new Date().toISOString()
): string {
  const stamp = new Date(generatedAtUtc)
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, 'Z');
  return `${stamp}-${slug(runLabel)}`;
}

function renderValueTransitionsJsonl(
  rows: readonly TdReplayValueTransitionPayload[]
): string {
  return renderJsonl(rows, validateValueTransition);
}

function renderOpponentSamplesJsonl(
  rows: readonly TdReplayOpponentSamplePayload[]
): string {
  return renderJsonl(rows, validateOpponentSample);
}

function renderJsonl<T>(
  rows: readonly T[],
  validate: (row: T, rowNumber: number) => void
): string {
  const lines: string[] = [];
  for (let index = 0; index < rows.length; index += 1) {
    const rowNumber = index + 1;
    validate(rows[index], rowNumber);
    lines.push(JSON.stringify(rows[index]));
  }
  return lines.length > 0 ? `${lines.join('\n')}\n` : '';
}

function validateValueTransition(
  row: TdReplayValueTransitionPayload,
  rowNumber: number
): void {
  validateVector(row.observation, OBSERVATION_DIM, `value row ${rowNumber}`);
  if (row.done) {
    if (row.nextObservation !== null) {
      throw new Error(
        `Invalid value row ${String(rowNumber)}: done=true requires nextObservation=null.`
      );
    }
  } else if (row.nextObservation === null) {
    throw new Error(
      `Invalid value row ${String(rowNumber)}: done=false requires nextObservation.`
    );
  } else {
    validateVector(
      row.nextObservation,
      OBSERVATION_DIM,
      `value row ${rowNumber} nextObservation`
    );
  }
  if (row.playerId !== 'PlayerA' && row.playerId !== 'PlayerB') {
    throw new Error(
      `Invalid value row ${String(rowNumber)}: playerId must be PlayerA or PlayerB.`
    );
  }
  if (row.episodeId.trim() === '') {
    throw new Error(
      `Invalid value row ${String(rowNumber)}: episodeId must be non-empty.`
    );
  }
  if (!Number.isInteger(row.timestep) || row.timestep < 0) {
    throw new Error(
      `Invalid value row ${String(rowNumber)}: timestep must be an integer >= 0.`
    );
  }
  if (typeof row.reward !== 'number' || !Number.isFinite(row.reward)) {
    throw new Error(
      `Invalid value row ${String(rowNumber)}: reward must be finite.`
    );
  }
}

function validateOpponentSample(
  row: TdReplayOpponentSamplePayload,
  rowNumber: number
): void {
  validateVector(row.observation, OBSERVATION_DIM, `opponent row ${rowNumber}`);
  if (row.playerId !== 'PlayerA' && row.playerId !== 'PlayerB') {
    throw new Error(
      `Invalid opponent row ${String(rowNumber)}: playerId must be PlayerA or PlayerB.`
    );
  }
  if (row.actionFeatures.length === 0) {
    throw new Error(
      `Invalid opponent row ${String(rowNumber)}: actionFeatures must be non-empty.`
    );
  }
  if (
    !Number.isInteger(row.actionIndex) ||
    row.actionIndex < 0 ||
    row.actionIndex >= row.actionFeatures.length
  ) {
    throw new Error(
      `Invalid opponent row ${String(rowNumber)}: actionIndex is out of bounds.`
    );
  }
  for (let index = 0; index < row.actionFeatures.length; index += 1) {
    validateVector(
      row.actionFeatures[index],
      ACTION_FEATURE_DIM,
      `opponent row ${rowNumber} actionFeatures[${String(index)}]`
    );
  }
}

function validateVector(
  vector: readonly number[],
  expectedLength: number,
  label: string
): void {
  if (vector.length !== expectedLength) {
    throw new Error(
      `Invalid ${label}: expected length ${String(expectedLength)}, actual ${String(vector.length)}.`
    );
  }
  if (!vector.every((value) => Number.isFinite(value))) {
    throw new Error(`Invalid ${label}: all entries must be finite numbers.`);
  }
}

async function writeAtomic(
  targetPath: string,
  contents: string
): Promise<void> {
  const tempPath = `${targetPath}.tmp`;
  await writeFile(tempPath, contents, 'utf8');
  await rename(tempPath, targetPath);
}

function slug(value: string): string {
  return (
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'run'
  );
}
