import { randomUUID } from 'node:crypto';
import { mkdir, open, rename, rm, writeFile } from 'node:fs/promises';
import type { FileHandle } from 'node:fs/promises';
import path from 'node:path';

import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import {
  ACTION_FEATURE_DIM,
  ENCODING_VERSION,
  OBSERVATION_DIM,
} from '../policies/trainingEncoding';
import { collectGitMetadata } from './gitMetadata';
import {
  collectTdReplayGames,
  validateTdReplayConfig,
  type TdReplayDependencies,
} from './tdReplay';
import {
  TD_REPLAY_ARTIFACT_TYPE,
  TD_REPLAY_SUMMARY_SCHEMA_VERSION,
  type CollectedTdReplayGame,
  type TdReplayConfig,
  type TdReplayOpponentSamplePayload,
  type TdReplayRun,
  type TdReplaySummary,
  type TdReplaySummaryGame,
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

export type CollectTdReplayArtifactOptions = TdReplayArtifactOptions &
  TdReplayDependencies;

interface TdReplaySummarySource {
  config: TdReplayConfig;
  games: readonly TdReplaySummaryGame[];
  elapsedMs: number;
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
  await writeAtomic(
    valuePath,
    renderValueTransitionsJsonl(run.valueTransitions)
  );
  await writeAtomic(
    opponentPath,
    renderOpponentSamplesJsonl(run.opponentSamples)
  );
  await writeAtomic(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  return {
    summary,
    valuePath,
    opponentPath,
    summaryPath,
  };
}

export async function collectAndWriteTdReplayArtifacts(
  config: TdReplayConfig,
  outputDirectory: string,
  options: CollectTdReplayArtifactOptions = {}
): Promise<WrittenTdReplayArtifacts> {
  validateTdReplayConfig(config);
  const writer = await StreamingTdReplayArtifactWriter.open(
    config,
    outputDirectory,
    options
  );
  try {
    const result = await collectTdReplayGames(config, options, (game) =>
      writer.writeGame(game)
    );
    return await writer.close(result.elapsedMs);
  } catch (error) {
    await writer.abort();
    throw error;
  }
}

export function createTdReplaySummary(
  run: TdReplayRun,
  options: TdReplayArtifactOptions & {
    artifacts: TdReplaySummary['artifacts'];
  }
): TdReplaySummary {
  return createTdReplaySummaryFromGames(
    {
      config: run.config,
      games: run.games.map(summarizeTdReplayGame),
      elapsedMs: run.elapsedMs,
    },
    options
  );
}

export function createTdReplaySummaryFromGames(
  source: TdReplaySummarySource,
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
  let valueTransitionTotal = 0;
  let opponentSampleTotal = 0;
  for (const game of source.games) {
    winners[game.winner] += 1;
    turnTotal += game.turns;
    decisionTotal += game.decisions;
    valueTransitionTotal += game.valueTransitions;
    opponentSampleTotal += game.opponentSamples;
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
    config: structuredClone(source.config),
    encoding: {
      encodingVersion: ENCODING_VERSION,
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
    },
    results: {
      games: source.games.length,
      winners,
      averageTurns:
        source.games.length > 0 ? turnTotal / source.games.length : 0,
      decisions: decisionTotal,
      valueTransitions: valueTransitionTotal,
      opponentSamples: opponentSampleTotal,
      elapsedMs: source.elapsedMs,
    },
    artifacts: structuredClone(options.artifacts),
    games: source.games.map((game) => structuredClone(game)),
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

function renderJsonlChunk<T>(
  rows: readonly T[],
  validate: (row: T, rowNumber: number) => void,
  firstRowNumber: number
): string {
  const lines: string[] = [];
  for (let index = 0; index < rows.length; index += 1) {
    const rowNumber = firstRowNumber + index;
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
  if (row.actionProbs.length !== row.actionFeatures.length) {
    throw new Error(
      `Invalid opponent row ${String(rowNumber)}: actionProbs length must match actionFeatures.`
    );
  }
  let probTotal = 0;
  for (const value of row.actionProbs) {
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(
        `Invalid opponent row ${String(rowNumber)}: actionProbs must be finite non-negative numbers.`
      );
    }
    probTotal += value;
  }
  if (!Number.isFinite(probTotal) || probTotal <= 0) {
    throw new Error(
      `Invalid opponent row ${String(rowNumber)}: actionProbs must sum to > 0.`
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
  const tempPath = `${targetPath}.${String(process.pid)}.${randomUUID()}.tmp`;
  await writeFile(tempPath, contents, 'utf8');
  await rename(tempPath, targetPath);
}

class StreamingTdReplayArtifactWriter {
  private valueHandle: FileHandle | null;
  private opponentHandle: FileHandle | null;
  private valueRowsWritten = 0;
  private opponentRowsWritten = 0;
  private readonly games: TdReplaySummaryGame[] = [];
  private closed = false;

  private constructor(
    private readonly config: TdReplayConfig,
    private readonly options: TdReplayArtifactOptions,
    private readonly valuePath: string,
    private readonly opponentPath: string,
    private readonly summaryPath: string,
    private readonly valueTempPath: string,
    private readonly opponentTempPath: string,
    private readonly summaryTempPath: string,
    valueHandle: FileHandle,
    opponentHandle: FileHandle
  ) {
    this.valueHandle = valueHandle;
    this.opponentHandle = opponentHandle;
  }

  static async open(
    config: TdReplayConfig,
    outputDirectory: string,
    options: TdReplayArtifactOptions
  ): Promise<StreamingTdReplayArtifactWriter> {
    const generatedAtUtc = options.generatedAtUtc ?? new Date().toISOString();
    const runBaseName =
      options.runBaseName ??
      defaultTdReplayRunBaseName(config.runLabel, generatedAtUtc);
    const valuePath = path.join(outputDirectory, `${runBaseName}.value.jsonl`);
    const opponentPath = path.join(
      outputDirectory,
      `${runBaseName}.opponent.jsonl`
    );
    const summaryPath = path.join(
      outputDirectory,
      `${runBaseName}.summary.json`
    );
    const valueTempPath = tempArtifactPath(valuePath);
    const opponentTempPath = tempArtifactPath(opponentPath);
    const summaryTempPath = tempArtifactPath(summaryPath);

    await mkdir(outputDirectory, { recursive: true });
    const valueHandle = await open(valueTempPath, 'w');
    const opponentHandle = await open(opponentTempPath, 'w');
    return new StreamingTdReplayArtifactWriter(
      structuredClone(config),
      {
        ...options,
        generatedAtUtc,
      },
      valuePath,
      opponentPath,
      summaryPath,
      valueTempPath,
      opponentTempPath,
      summaryTempPath,
      valueHandle,
      opponentHandle
    );
  }

  async writeGame(game: CollectedTdReplayGame): Promise<void> {
    if (
      this.closed ||
      this.valueHandle === null ||
      this.opponentHandle === null
    ) {
      throw new Error(
        'Cannot write TD replay game after artifact writer is closed.'
      );
    }
    const valueChunk = renderJsonlChunk(
      game.valueTransitions,
      validateValueTransition,
      this.valueRowsWritten + 1
    );
    if (valueChunk.length > 0) {
      await this.valueHandle.writeFile(valueChunk, 'utf8');
    }
    this.valueRowsWritten += game.valueTransitions.length;

    const opponentChunk = renderJsonlChunk(
      game.opponentSamples,
      validateOpponentSample,
      this.opponentRowsWritten + 1
    );
    if (opponentChunk.length > 0) {
      await this.opponentHandle.writeFile(opponentChunk, 'utf8');
    }
    this.opponentRowsWritten += game.opponentSamples.length;
    this.games.push(summarizeTdReplayGame(game));
  }

  async close(elapsedMs: number): Promise<WrittenTdReplayArtifacts> {
    if (this.closed) {
      throw new Error('TD replay artifact writer is already closed.');
    }
    this.closed = true;
    await this.closeOpenHandles();

    const summary = createTdReplaySummaryFromGames(
      {
        config: this.config,
        games: this.games,
        elapsedMs,
      },
      {
        ...this.options,
        artifacts: {
          valueTransitions: this.valuePath,
          opponentSamples: this.opponentPath,
          summary: this.summaryPath,
        },
      }
    );
    await writeFile(
      this.summaryTempPath,
      `${JSON.stringify(summary, null, 2)}\n`,
      'utf8'
    );

    const renamed: string[] = [];
    try {
      await rename(this.valueTempPath, this.valuePath);
      renamed.push(this.valuePath);
      await rename(this.opponentTempPath, this.opponentPath);
      renamed.push(this.opponentPath);
      await rename(this.summaryTempPath, this.summaryPath);
      renamed.push(this.summaryPath);
    } catch (error) {
      await Promise.all(
        renamed.map((targetPath) => rm(targetPath, { force: true }))
      );
      await this.removeTempFiles();
      throw error;
    }

    return {
      summary,
      valuePath: this.valuePath,
      opponentPath: this.opponentPath,
      summaryPath: this.summaryPath,
    };
  }

  async abort(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      await this.closeOpenHandles();
    }
    await this.removeTempFiles();
  }

  private async closeOpenHandles(): Promise<void> {
    const handles = [this.valueHandle, this.opponentHandle];
    this.valueHandle = null;
    this.opponentHandle = null;
    await Promise.all(handles.map((handle) => handle?.close()));
  }

  private async removeTempFiles(): Promise<void> {
    await Promise.all(
      [this.valueTempPath, this.opponentTempPath, this.summaryTempPath].map(
        (targetPath) => rm(targetPath, { force: true })
      )
    );
  }
}

function summarizeTdReplayGame(
  game: CollectedTdReplayGame
): TdReplaySummaryGame {
  return {
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
  };
}

function tempArtifactPath(targetPath: string): string {
  return `${targetPath}.${String(process.pid)}.${randomUUID()}.tmp`;
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
