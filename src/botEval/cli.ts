import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  createHeadToHeadArtifact,
  defaultHeadToHeadOutputDirectory,
  loadHeadToHeadArtifact,
  type HeadToHeadArtifactOptions,
  writeHeadToHeadArtifacts,
} from './artifacts';
import {
  parseHeadToHeadConfig,
  parseRolloutSearchSweepConfig,
  parseTdReplayConfig,
} from './config';
import { installLocalPublicFetch } from './localPublicFetch';
import { resolveEvaluationExecution } from './execution';
import { runHeadToHead, type HeadToHeadProgress } from './matchup';
import { replayArtifactGame } from './replay';
import {
  defaultRolloutSearchSweepOutputDirectory,
  writeRolloutSearchSweepArtifacts,
} from './sweepArtifacts';
import {
  runRolloutSearchSweep,
  type RolloutSearchSweepProgress,
} from './sweep';
import type { TdReplayProgress } from './tdReplay';
import {
  collectAndWriteTdReplayArtifacts,
  defaultTdReplayOutputDirectory,
} from './tdReplayArtifacts';
import {
  collectAndWriteShardedTdReplayArtifacts,
  defaultShardedTdReplayOutputDirectory,
  type TdReplayShardProgress,
} from './tdReplayShards';
import type { RolloutSearchSweepRun } from './types';

const DEFAULT_PROGRESS_INTERVAL_SECONDS = 30;

async function main(): Promise<void> {
  const [command, ...args] = process.argv.slice(2);
  switch (command) {
    case 'head-to-head':
      await runHeadToHeadCommand(args);
      return;
    case 'replay':
      await runReplayCommand(args);
      return;
    case 'rollout-search-sweep':
      await runRolloutSearchSweepCommand(args);
      return;
    case 'collect-td-replay':
      await runCollectTdReplayCommand(args);
      return;
    case 'collect-td-replay-sharded':
      await runCollectTdReplayShardedCommand(args);
      return;
    default:
      throw new Error(
        'Usage: yarn bot:eval head-to-head --config <path> [--out-dir <path>] [--workers <positive-integer>] [--progress-interval-seconds <number>] | rollout-search-sweep --config <path> [--out-dir <path>] [--workers <positive-integer>] [--progress-interval-seconds <number>] | collect-td-replay --config <path> [--out-dir <path>] [--progress-interval-seconds <number>] | collect-td-replay-sharded --config <path> [--out-dir <path>] [--workers <positive-integer>] [--shard-games <positive-integer>] [--progress-interval-seconds <number>] | replay --artifact <path> --game-id <id>'
      );
  }
}

async function runHeadToHeadCommand(args: readonly string[]): Promise<void> {
  installLocalPublicFetch();
  const flags = parseFlags(args);
  const configPath = requiredFlag(flags, '--config');
  const config = parseHeadToHeadConfig(
    JSON.parse(await readFile(configPath, 'utf8'))
  );
  const outputDirectory =
    flags.get('--out-dir') ?? defaultHeadToHeadOutputDirectory(config.runLabel);
  const progressIntervalMs = parseProgressIntervalMs(flags);
  const workers = parseWorkers(flags);
  const execution = resolveEvaluationExecution(workers, config.gamesPerSide);
  process.stderr.write(
    `[matchup] started candidate=${config.candidate.id} opponent=${config.opponent.id} games=${String(config.gamesPerSide * 2)} workers=${String(execution.workers)} requestedWorkers=${String(execution.requestedWorkers)} latencyMode=${execution.latencyMode}\n`
  );
  const run = await runHeadToHead(config, {
    workers,
    progressIntervalMs,
    onProgress: logHeadToHeadProgress,
  });
  const artifact = createHeadToHeadArtifact(run);
  const written = await writeHeadToHeadArtifacts(artifact, outputDirectory);
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        summary: path.resolve(written.summaryPath),
        results: artifact.summary,
      },
      null,
      2
    )}\n`
  );
}

async function runRolloutSearchSweepCommand(
  args: readonly string[]
): Promise<void> {
  const flags = parseFlags(args);
  const configPath = requiredFlag(flags, '--config');
  const config = parseRolloutSearchSweepConfig(
    JSON.parse(await readFile(configPath, 'utf8'))
  );
  const outputDirectory =
    flags.get('--out-dir') ??
    defaultRolloutSearchSweepOutputDirectory(config.runLabel);
  const progressIntervalMs = parseProgressIntervalMs(flags);
  const workers = parseWorkers(flags);
  const execution = resolveEvaluationExecution(workers, config.gamesPerSide);
  const initialRun: RolloutSearchSweepRun = {
    config,
    execution,
    matchups: [],
  };
  let written = await writeRolloutSearchSweepArtifacts(
    initialRun,
    outputDirectory,
    {
      status: 'running',
      matchupIndicesToWrite: [],
    }
  );
  const artifactOptions: HeadToHeadArtifactOptions = {
    generatedAtUtc: written.artifact.generatedAtUtc,
    git: written.artifact.git,
    nodeVersion: written.artifact.runtime.nodeVersion,
  };
  process.stderr.write(`[sweep] artifacts=${path.resolve(outputDirectory)}\n`);
  const run = await runRolloutSearchSweep(config, {
    workers,
    progressIntervalMs,
    onProgress: logRolloutSearchSweepProgress,
    async onCandidateCompleted(completed) {
      written = await writeRolloutSearchSweepArtifacts(
        completed.run,
        outputDirectory,
        {
          ...artifactOptions,
          status: 'running',
          matchupIndicesToWrite: [completed.candidateIndex - 1],
        }
      );
    },
  });
  written = await writeRolloutSearchSweepArtifacts(run, outputDirectory, {
    ...artifactOptions,
    status: 'completed',
    matchupIndicesToWrite: [],
  });
  process.stderr.write(
    `[sweep] artifacts completed json=${path.resolve(written.artifactPath)} csv=${path.resolve(written.csvPath)} summary=${path.resolve(written.summaryPath)}\n`
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        csv: path.resolve(written.csvPath),
        summary: path.resolve(written.summaryPath),
        results: written.artifact.rows,
      },
      null,
      2
    )}\n`
  );
}

async function runCollectTdReplayCommand(
  args: readonly string[]
): Promise<void> {
  const flags = parseFlags(args);
  const configPath = requiredFlag(flags, '--config');
  const config = parseTdReplayConfig(
    JSON.parse(await readFile(configPath, 'utf8'))
  );
  const outputDirectory =
    flags.get('--out-dir') ?? defaultTdReplayOutputDirectory();
  const progressIntervalMs = parseProgressIntervalMs(flags);
  process.stderr.write(
    `[td-replay] started playerA=${config.playerA.id} playerB=${config.playerB.id} games=${String(config.games)}\n`
  );
  const written = await collectAndWriteTdReplayArtifacts(
    config,
    outputDirectory,
    {
      progressIntervalMs,
      onProgress: logTdReplayProgress,
    }
  );
  process.stderr.write(
    `[td-replay] artifacts completed value=${path.resolve(written.valuePath)} opponent=${path.resolve(written.opponentPath)} summary=${path.resolve(written.summaryPath)}\n`
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        valueTransitionsArtifact: path.resolve(written.valuePath),
        opponentSamplesArtifact: path.resolve(written.opponentPath),
        summaryArtifact: path.resolve(written.summaryPath),
        results: written.summary.results,
      },
      null,
      2
    )}\n`
  );
}

async function runCollectTdReplayShardedCommand(
  args: readonly string[]
): Promise<void> {
  const flags = parseFlags(args);
  const configPath = requiredFlag(flags, '--config');
  const config = parseTdReplayConfig(
    JSON.parse(await readFile(configPath, 'utf8'))
  );
  const outputDirectory =
    flags.get('--out-dir') ?? defaultShardedTdReplayOutputDirectory();
  const workers = parseWorkers(flags);
  const shardGames = parseOptionalPositiveInteger(flags, '--shard-games');
  const progressIntervalMs = parseProgressIntervalMs(flags);
  process.stderr.write(
    `[td-replay-sharded] started playerA=${config.playerA.id} playerB=${config.playerB.id} games=${String(config.games)} workers=${String(workers)} shardGames=${formatOptionalShardGames(shardGames)}\n`
  );
  const written = await collectAndWriteShardedTdReplayArtifacts(
    config,
    outputDirectory,
    {
      workers,
      ...(shardGames === undefined ? {} : { shardGames }),
      progressIntervalMs,
      onProgress: logTdReplayShardProgress,
    }
  );
  process.stderr.write(
    `[td-replay-sharded] artifacts completed directory=${path.resolve(written.runDirectory)} shards=${path.resolve(written.shardsDirectory)} summary=${path.resolve(written.summaryPath)}\n`
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        runDirectory: path.resolve(written.runDirectory),
        shardsDirectory: path.resolve(written.shardsDirectory),
        summaryArtifact: path.resolve(written.summaryPath),
        valueTransitionArtifacts: written.summary.shards.map((shard) =>
          path.resolve(shard.valuePath)
        ),
        opponentSampleArtifacts: written.summary.shards.map((shard) =>
          path.resolve(shard.opponentPath)
        ),
        results: written.summary.results,
      },
      null,
      2
    )}\n`
  );
}

async function runReplayCommand(args: readonly string[]): Promise<void> {
  const flags = parseFlags(args);
  const artifactPath = requiredFlag(flags, '--artifact');
  const gameId = requiredFlag(flags, '--game-id');
  const artifact = await loadHeadToHeadArtifact(artifactPath);
  const result = await replayArtifactGame(artifact, gameId);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

function parseFlags(args: readonly string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`Invalid CLI arguments near ${String(key)}.`);
    }
    if (flags.has(key)) {
      throw new Error(`Duplicate CLI flag: ${key}.`);
    }
    flags.set(key, value);
  }
  return flags;
}

function requiredFlag(
  flags: ReadonlyMap<string, string>,
  name: string
): string {
  const value = flags.get(name);
  if (!value) {
    throw new Error(`Missing required flag ${name}.`);
  }
  return value;
}

function parseProgressIntervalMs(flags: ReadonlyMap<string, string>): number {
  const value =
    flags.get('--progress-interval-seconds') ??
    String(DEFAULT_PROGRESS_INTERVAL_SECONDS);
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new Error('--progress-interval-seconds must be a number >= 0.');
  }
  return seconds * 1000;
}

function parseWorkers(flags: ReadonlyMap<string, string>): number {
  const value = flags.get('--workers') ?? '1';
  const workers = Number(value);
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new Error('--workers must be a positive integer.');
  }
  return workers;
}

function parseOptionalPositiveInteger(
  flags: ReadonlyMap<string, string>,
  name: string
): number | undefined {
  const value = flags.get(name);
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function logRolloutSearchSweepProgress(
  progress: RolloutSearchSweepProgress
): void {
  switch (progress.type) {
    case 'sweep-started':
      process.stderr.write(
        `[sweep] started candidates=${String(progress.candidates)} gamesPerSide=${String(progress.gamesPerSide)} totalGames=${String(progress.totalGames)} workers=${String(progress.workers)}\n`
      );
      return;
    case 'candidate-started':
      process.stderr.write(
        `[sweep] candidate ${String(progress.candidateIndex)}/${String(progress.totalCandidates)} started id=${progress.candidateId} config=${progress.configLabel} proxy=${String(progress.configProxyCost)}\n`
      );
      return;
    case 'candidate-completed': {
      const summary = progress.matchup.summary;
      const latency =
        summary.multiChoiceLatencyByBotId[progress.candidateId]?.p95Ms ?? 0;
      process.stderr.write(
        `[sweep] candidate ${String(progress.candidateIndex)}/${String(progress.totalCandidates)} completed id=${progress.candidateId} winRate=${formatDecimal(summary.candidateWinRate)} ci95=[${formatDecimal(summary.candidateWinRateCi95.low)},${formatDecimal(summary.candidateWinRateCi95.high)}] multiP95=${formatMilliseconds(latency)} elapsed=${formatDuration(progress.elapsedMs)}\n`
      );
      return;
    }
    case 'sweep-completed':
      process.stderr.write(
        `[sweep] completed candidates=${String(progress.candidates)} elapsed=${formatDuration(progress.elapsedMs)}\n`
      );
      return;
    default:
      logHeadToHeadProgress(progress);
  }
}

function logTdReplayProgress(progress: TdReplayProgress): void {
  switch (progress.type) {
    case 'game-heartbeat':
      process.stderr.write(
        `[td-replay] heartbeat game=${progress.gameId} ${String(progress.gameNumber)}/${String(progress.totalGames)} turn=${String(progress.turn)} decisions=${String(progress.decisions)} elapsed=${formatDuration(progress.elapsedMs)}\n`
      );
      return;
    case 'game-completed':
      process.stderr.write(
        `[td-replay] game completed=${String(progress.gameNumber)}/${String(progress.totalGames)} game=${progress.game.gameId} winner=${progress.game.finalScore.winner} turns=${String(progress.game.turns)} decisions=${String(progress.game.decisions.length)} valueRows=${String(progress.game.valueTransitions.length)} opponentRows=${String(progress.game.opponentSamples.length)} elapsed=${formatDuration(progress.elapsedMs)} rate=${formatDecimal(progress.gamesPerMinute)} games/min\n`
      );
      return;
  }
}

function logTdReplayShardProgress(progress: TdReplayShardProgress): void {
  switch (progress.type) {
    case 'sharded-started':
      process.stderr.write(
        `[td-replay-sharded] shards started games=${String(progress.games)} shards=${String(progress.shards)} workers=${String(progress.workers)} requestedWorkers=${String(progress.requestedWorkers)} shardGames=${formatOptionalShardGames(progress.shardGames)} directory=${path.resolve(progress.runDirectory)}\n`
      );
      return;
    case 'shard-started':
      process.stderr.write(
        `[td-replay-sharded] shard ${String(progress.shard.shardIndex)} started start=${String(progress.shard.gameIndexStart)} games=${String(progress.shard.games)}\n`
      );
      return;
    case 'shard-progress':
      logTdReplayProgress(progress.progress);
      return;
    case 'shard-completed':
      process.stderr.write(
        `[td-replay-sharded] shard ${String(progress.shard.shardIndex)} completed start=${String(progress.shard.gameIndexStart)} games=${String(progress.shard.games)} valueRows=${String(progress.result.written.summary.results.valueTransitions)} opponentRows=${String(progress.result.written.summary.results.opponentSamples)} elapsed=${formatDuration(progress.result.written.summary.results.elapsedMs)}\n`
      );
      return;
    case 'sharded-completed':
      process.stderr.write(
        `[td-replay-sharded] completed games=${String(progress.summary.results.games)} valueRows=${String(progress.summary.results.valueTransitions)} opponentRows=${String(progress.summary.results.opponentSamples)} elapsed=${formatDuration(progress.summary.results.elapsedMs)}\n`
      );
      return;
  }
}

function logHeadToHeadProgress(progress: HeadToHeadProgress): void {
  switch (progress.type) {
    case 'game-heartbeat':
      process.stderr.write(
        `[matchup] ${progress.candidateId} heartbeat worker=${String(progress.workerId)} pair=${String(progress.pairNumber)} game=${progress.gameId} turn=${String(progress.turn)} decisions=${String(progress.decisions)} elapsed=${formatDuration(progress.elapsedMs)}\n`
      );
      return;
    case 'game-completed':
      process.stderr.write(
        `[matchup] ${progress.candidateId} game completed=${String(progress.completedGames)}/${String(progress.totalGames)} worker=${String(progress.workerId)} pair=${String(progress.pairNumber)} game=${progress.game.gameId} winner=${progress.game.finalScore.winner} turns=${String(progress.game.turns)} decisions=${String(progress.game.transcript.length)} elapsed=${formatDuration(progress.elapsedMs)} rate=${formatDecimal(progress.gamesPerMinute)} games/min\n`
      );
      return;
    case 'pair-completed':
      process.stderr.write(
        `[matchup] ${progress.candidateId} pair completed=${String(progress.completedPairs)}/${String(progress.totalPairs)} pair=${String(progress.pairNumber)} worker=${String(progress.workerId)} games=${String(progress.completedGames)}/${String(progress.totalGames)} elapsed=${formatDuration(progress.elapsedMs)} rate=${formatDecimal(progress.gamesPerMinute)} games/min eta=${formatDuration(progress.etaMs)}\n`
      );
      return;
  }
}

function formatDuration(milliseconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds]
    .map((value) => String(value).padStart(2, '0'))
    .join(':');
}

function formatDecimal(value: number): string {
  return value.toFixed(3);
}

function formatMilliseconds(value: number): string {
  return `${value.toFixed(1)}ms`;
}

function formatOptionalShardGames(value: number | undefined): string {
  return value === undefined ? 'auto' : String(value);
}

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`[bot:eval] ${message}\n`);
  process.exitCode = 1;
});
