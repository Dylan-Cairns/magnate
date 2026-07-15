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
import { installLocalPublicFetch, localPublicUrl } from './localPublicFetch';
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
import {
  createStrategicPositionArtifactV0,
  defaultStrategicPositionOutputDirectoryV0,
  writeStrategicPositionArtifactsV0,
} from './strategicPositionArtifacts';
import {
  createStrategicPositionCatalogV0,
  isStrategicOptionalityPositionV0,
} from './strategicPositionCatalog';
import {
  createDefaultStrategicComparisonVariantsV0,
  createStrategicComparisonVariantCatalogV0,
  runStrategicPositionComparisonV0,
} from './strategicPositionComparison';
import { runStrategicForcedRolloutTraceV0 } from './strategicForcedRolloutTrace';
import {
  createStrategicForcedRolloutTraceArtifactV0,
  defaultStrategicForcedRolloutTraceOutputDirectoryV0,
  writeStrategicForcedRolloutTraceArtifactsV0,
} from './strategicForcedRolloutTraceArtifacts';
import {
  runTdSymmetryAudit,
  sampleOpponentReplayDirectory,
} from './tdSymmetry';
import {
  createTdSymmetryArtifact,
  defaultTdSymmetryOutputDirectory,
  writeTdSymmetryArtifacts,
} from './tdSymmetryArtifacts';
import {
  loadTdRootModelFromIndexUrl,
  parseTdRootModelPackIndex,
} from '../policies/tdRootModelPack';

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
    case 'strategic-positions':
      await runStrategicPositionsCommand(args);
      return;
    case 'strategic-forced-rollouts':
      await runStrategicForcedRolloutsCommand(args);
      return;
    case 'td-symmetry':
      await runTdSymmetryCommand(args);
      return;
    default:
      throw new Error(
        'Usage: yarn bot:eval head-to-head --config <path> [--out-dir <path>] [--workers <positive-integer>] [--progress-interval-seconds <number>] | rollout-search-sweep --config <path> [--out-dir <path>] [--workers <positive-integer>] [--progress-interval-seconds <number>] | collect-td-replay --config <path> [--out-dir <path>] [--progress-interval-seconds <number>] | collect-td-replay-sharded --config <path> [--out-dir <path>] [--workers <positive-integer>] [--shard-games <positive-integer>] [--progress-interval-seconds <number>] | strategic-positions [--out-dir <path>] [--repetitions <positive-integer>] [--start-repetition <nonnegative-integer>] [--positions <comma-separated-ids>] [--variants <comma-separated-ids>] | strategic-forced-rollouts [--out-dir <path>] [--positions <comma-separated-ids>] [--repetitions <comma-separated-nonnegative-integers>] [--scenarios <comma-separated-nonnegative-integers>] | td-symmetry --replay-dir <path> [--sample-size <positive-integer>] [--sampling-seed <text>] [--pack-id <id>] [--model-index-path <public-relative-path>] [--worst-case-limit <nonnegative-integer>] [--out-dir <path>] [--progress-interval-seconds <number>] | replay --artifact <path> --game-id <id>'
      );
  }
}

async function runTdSymmetryCommand(args: readonly string[]): Promise<void> {
  installLocalPublicFetch();
  const flags = parseFlags(args);
  const replayDirectory = requiredFlag(flags, '--replay-dir');
  const sampleSize =
    parseOptionalPositiveInteger(flags, '--sample-size') ?? 10_000;
  const samplingSeed =
    flags.get('--sampling-seed') ?? 'td-symmetry-v2-hard-900-v1';
  if (samplingSeed.trim() === '') {
    throw new Error('--sampling-seed must be non-empty.');
  }
  const modelIndexPath =
    flags.get('--model-index-path') ?? 'model-packs/index.json';
  const indexPayload = JSON.parse(
    await readFile(path.join(process.cwd(), 'public', modelIndexPath), 'utf8')
  ) as unknown;
  const index = parseTdRootModelPackIndex(indexPayload);
  const modelPackId = flags.get('--pack-id') ?? index.defaultPackId;
  if (!modelPackId) {
    throw new Error(
      'The TD model index has no default pack; provide --pack-id explicitly.'
    );
  }
  const modelIndexUrl = new URL(localPublicUrl(modelIndexPath));
  modelIndexUrl.searchParams.set('tdPackId', modelPackId);
  const outputDirectory =
    flags.get('--out-dir') ?? defaultTdSymmetryOutputDirectory();
  const worstCaseLimit =
    parseOptionalNonnegativeInteger(flags, '--worst-case-limit') ?? 25;
  const progressIntervalMs = parseProgressIntervalMs(flags);

  process.stderr.write(
    `[td-symmetry] scanning replay=${path.resolve(replayDirectory)} requestedSample=${String(sampleSize)} seed=${samplingSeed}\n`
  );
  const sampleSet = await sampleOpponentReplayDirectory(
    replayDirectory,
    sampleSize,
    samplingSeed
  );
  process.stderr.write(
    `[td-symmetry] sampled=${String(sampleSet.samples.length)} rowsScanned=${String(sampleSet.rowsScanned)} files=${String(sampleSet.files.length)} loadingPack=${modelPackId}\n`
  );
  const model = await loadTdRootModelFromIndexUrl(modelIndexUrl.toString());
  const startedAt = Date.now();
  let lastProgressAt = startedAt;
  const run = runTdSymmetryAudit({
    samples: sampleSet.samples,
    rowsScanned: sampleSet.rowsScanned,
    sourceFiles: sampleSet.files,
    samplingSeed,
    requestedSampleSize: sampleSize,
    modelPackId,
    modelIndexPath,
    model,
    worstCaseLimit,
    onProgress(progress) {
      const now = Date.now();
      if (
        progress.completedSamples === progress.totalSamples ||
        progressIntervalMs === 0 ||
        now - lastProgressAt >= progressIntervalMs
      ) {
        lastProgressAt = now;
        process.stderr.write(
          `[td-symmetry] decision ${String(progress.completedSamples)}/${String(progress.totalSamples)} elapsed=${formatDuration(now - startedAt)}\n`
        );
      }
    },
  });
  const artifact = createTdSymmetryArtifact(run);
  const written = await writeTdSymmetryArtifacts(artifact, outputDirectory);
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        summary: path.resolve(written.summaryPath),
        sampledRows: run.aggregate.samples,
        rowsScanned: run.replay.rowsScanned,
        topActionMatchRate: run.aggregate.topActionMatchRate,
        samplesWithAnyTopActionFlip: run.aggregate.samplesWithAnyTopActionFlip,
      },
      null,
      2
    )}\n`
  );
}

async function runStrategicForcedRolloutsCommand(
  args: readonly string[]
): Promise<void> {
  installLocalPublicFetch();
  const flags = parseFlags(args);
  const catalog = createStrategicPositionCatalogV0();
  const requestedPositionIds = parseOptionalIds(flags, '--positions');
  const positions = requestedPositionIds
    ? selectStrategicItems(
        catalog,
        requestedPositionIds,
        (position) => position.id,
        'position'
      )
    : catalog.filter(isStrategicOptionalityPositionV0);
  const repetitionIds = parseOptionalNonnegativeIntegerIds(
    flags,
    '--repetitions'
  ) ?? [0];
  const scenarioIndices =
    parseOptionalNonnegativeIntegerIds(flags, '--scenarios') ??
    Array.from({ length: 50 }, (_unused, index) => index);
  const outputDirectory =
    flags.get('--out-dir') ??
    defaultStrategicForcedRolloutTraceOutputDirectoryV0();
  const totalTraces =
    positions.length * repetitionIds.length * scenarioIndices.length * 4;
  const progressTraceInterval = Math.max(4, Math.floor(totalTraces / 80) * 4);
  process.stderr.write(
    `[strategic-forced-rollouts] started positions=${String(positions.length)} repetitions=${String(repetitionIds.length)} scenarios=${String(scenarioIndices.length)} traces=${String(totalTraces)}\n`
  );
  const run = await runStrategicForcedRolloutTraceV0({
    positions,
    repetitionIds,
    scenarioIndices,
    onProgress(progress) {
      if (
        progress.rootFocusActionId === 'overwrite-option' &&
        progress.guide === 'heuristic-v2' &&
        (progress.completedTraces === totalTraces ||
          progress.completedTraces % progressTraceInterval === 0)
      ) {
        process.stderr.write(
          `[strategic-forced-rollouts] scenario ${String(progress.completedTraces)}/${String(progress.totalTraces)} position=${progress.positionId} repetition=${String(progress.repetition)} scenario=${String(progress.scenarioIndex)}\n`
        );
      }
    },
  });
  const artifact = createStrategicForcedRolloutTraceArtifactV0(run);
  const written = await writeStrategicForcedRolloutTraceArtifactsV0(
    artifact,
    outputDirectory
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        summary: path.resolve(written.summaryPath),
        positions: run.positions.length,
        repetitions: run.repetitionIds,
        scenarios: run.scenarioIndices,
        traces: totalTraces,
      },
      null,
      2
    )}\n`
  );
}

async function runStrategicPositionsCommand(
  args: readonly string[]
): Promise<void> {
  installLocalPublicFetch();
  const flags = parseFlags(args);
  const repetitions = parseOptionalPositiveInteger(flags, '--repetitions') ?? 1;
  const repetitionStart =
    parseOptionalNonnegativeInteger(flags, '--start-repetition') ?? 0;
  const positions = selectStrategicItems(
    createStrategicPositionCatalogV0(),
    parseOptionalIds(flags, '--positions'),
    (position) => position.id,
    'position'
  );
  const requestedVariantIds = parseOptionalIds(flags, '--variants');
  const variants = selectStrategicItems(
    requestedVariantIds
      ? createStrategicComparisonVariantCatalogV0()
      : createDefaultStrategicComparisonVariantsV0(),
    requestedVariantIds,
    (variant) => variant.descriptor.id,
    'variant'
  );
  const outputDirectory =
    flags.get('--out-dir') ?? defaultStrategicPositionOutputDirectoryV0();
  process.stderr.write(
    `[strategic-positions] started repetitions=${String(repetitions)} start=${String(repetitionStart)} positions=${String(positions.length)} variants=${String(variants.length)}\n`
  );
  const run = await runStrategicPositionComparisonV0({
    positions,
    variants,
    repetitionStart,
    repetitions,
    onProgress(progress) {
      process.stderr.write(
        `[strategic-positions] decision ${String(progress.completedDecisions)}/${String(progress.totalDecisions)} position=${progress.positionId} repetition=${String(progress.repetition)} variant=${progress.variantId} selected=${progress.selectedActionKey}\n`
      );
    },
  });
  const artifact = createStrategicPositionArtifactV0(run);
  const written = await writeStrategicPositionArtifactsV0(
    artifact,
    outputDirectory
  );
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        summary: path.resolve(written.summaryPath),
        positions: run.positions.length,
        variants: run.variants.map((variant) => variant.id),
        repetitionStart: run.repetitionStart,
        repetitions: run.repetitions,
      },
      null,
      2
    )}\n`
  );
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
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function parseOptionalNonnegativeInteger(
  flags: ReadonlyMap<string, string>,
  name: string
): number | undefined {
  const value = flags.get(name);
  if (value === undefined) {
    return undefined;
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    throw new Error(`${name} must be a nonnegative integer.`);
  }
  return parsed;
}

function parseOptionalIds(
  flags: ReadonlyMap<string, string>,
  name: string
): readonly string[] | undefined {
  const value = flags.get(name);
  if (value === undefined) {
    return undefined;
  }
  const ids = value
    .split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
  if (ids.length === 0 || new Set(ids).size !== ids.length) {
    throw new Error(`${name} must contain unique comma-separated ids.`);
  }
  return ids;
}

function parseOptionalNonnegativeIntegerIds(
  flags: ReadonlyMap<string, string>,
  name: string
): readonly number[] | undefined {
  const values = parseOptionalIds(flags, name);
  if (!values) {
    return undefined;
  }
  const parsed = values.map(Number);
  if (
    parsed.some((value) => !Number.isSafeInteger(value) || value < 0) ||
    new Set(parsed).size !== parsed.length
  ) {
    throw new Error(
      `${name} must contain unique comma-separated nonnegative integers.`
    );
  }
  return parsed;
}

function selectStrategicItems<T>(
  items: readonly T[],
  requestedIds: readonly string[] | undefined,
  idForItem: (item: T) => string,
  label: string
): T[] {
  if (!requestedIds) {
    return [...items];
  }
  const byId = new Map(items.map((item) => [idForItem(item), item]));
  return requestedIds.map((id) => {
    const item = byId.get(id);
    if (!item) {
      throw new Error(`Unknown strategic ${label} id ${id}.`);
    }
    return item;
  });
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
