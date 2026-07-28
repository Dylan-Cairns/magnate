import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  writeFileSync,
} from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  createHeadToHeadArtifact,
  defaultHeadToHeadOutputDirectory,
  writeHeadToHeadArtifacts,
} from '../src/botEval/artifacts';
import {
  installLocalPublicFetch,
  localPublicUrl,
} from '../src/botEval/localPublicFetch';
import { runHeadToHead } from '../src/botEval/matchup';
import type { PairedSeedResult } from '../src/botEval/pair';
import type { HeadToHeadConfig, PlayedGame } from '../src/botEval/types';
import type { PlayerId } from '../src/engine/types';
import type { BotSpec } from '../src/policies/botSpec';
import type { TdRootGuidanceSource } from '../src/policies/tdRootGuidanceConfig';

interface ModelPackIndex {
  defaultPackId: string | null;
  packs: Array<{
    id: string;
    modelType: string;
    manifestPath: string;
  }>;
}

interface Options {
  games: number;
  worlds: number;
  rollouts: number;
  depth: number;
  maxRootActions: number;
  rolloutEpsilon: number;
  tdRoot: TdRootGuidanceSource;
  tdRollout: TdRootGuidanceSource;
  tdLeaf: TdRootGuidanceSource;
  opponent: 'heuristic-v2' | 'td';
  tdPackId?: string;
  tdModelIndexPath: string;
  workers: number;
  maxDecisionsPerGame: number;
  outDir?: string;
  resume: boolean;
  resumeKey?: string;
  dryRun: boolean;
}

const DEFAULT_OPTIONS: Options = {
  games: 10,
  worlds: 10,
  rollouts: 1,
  depth: 40,
  maxRootActions: 16,
  rolloutEpsilon: 0,
  tdRoot: 'td',
  tdRollout: 'td',
  tdLeaf: 'td',
  opponent: 'heuristic-v2',
  tdModelIndexPath: 'model-packs/index.json',
  workers: 1,
  maxDecisionsPerGame: 260,
  resume: false,
  dryRun: false,
};

interface PairCheckpoint {
  schemaVersion: 1;
  runFingerprint: string;
  pairIndex: number;
  result: PairedSeedResult;
}

interface ResumeState {
  schemaVersion: 1;
  runFingerprint: string;
  elapsedMs: number;
  completedPairs: number;
}

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  if (options.games % 2 !== 0) {
    throw new Error(
      '--games must be even because this benchmark side-swaps paired seeds.'
    );
  }

  installLocalPublicFetch();
  const manifestUrl = await resolveTdManifestUrl(options);
  const config = benchmarkConfig(options);
  const outDir =
    options.outDir ?? defaultHeadToHeadOutputDirectory(config.runLabel);
  const resume = prepareResume(options, config, outDir);
  process.stderr.write(
    `[td-vs-v2] games=${String(options.games)} workers=${String(options.workers)} worlds=${String(options.worlds)} depth=${String(options.depth)} maxRootActions=${String(options.maxRootActions)} tdRoot=${options.tdRoot} tdRollout=${options.tdRollout} tdLeaf=${options.tdLeaf} tdManifest=${manifestUrl} resumedPairs=${String(resume.results.length)}\n`
  );
  if (options.dryRun) {
    process.stdout.write(
      `${JSON.stringify(
        {
          status: 'ready',
          dryRun: true,
          outDir: path.resolve(outDir),
          manifestUrl,
          resumedPairs: resume.results.length,
          config,
        },
        null,
        2
      )}\n`
    );
    return;
  }

  const run = await runHeadToHead(config, {
    workers: options.workers,
    progressIntervalMs: 30_000,
    initialResults: resume.results,
    initialElapsedMs: resume.elapsedMs,
    onProgress(progress) {
      if (progress.type === 'game-heartbeat') {
        process.stderr.write(
          `[td-vs-v2] heartbeat pair=${String(progress.pairNumber)} game=${progress.gameId} turn=${String(progress.turn)} decisions=${String(progress.decisions)} elapsed=${formatSeconds(progress.elapsedMs)}\n`
        );
      } else if (progress.type === 'game-completed') {
        process.stderr.write(
          `${formatGameResult(progress.game, config)} completed=${String(progress.completedGames)}/${String(progress.totalGames)} rate=${progress.gamesPerMinute.toFixed(2)} games/min\n`
        );
      } else if (progress.type === 'pair-completed') {
        if (options.resume) {
          writePairCheckpoint(outDir, resume.runFingerprint, progress.result);
          writeResumeState(outDir, {
            schemaVersion: 1,
            runFingerprint: resume.runFingerprint,
            elapsedMs: progress.elapsedMs,
            completedPairs: progress.completedPairs,
          });
        }
        process.stderr.write(
          `[td-vs-v2] completed ${String(progress.completedGames)}/${String(progress.totalGames)} games rate=${progress.gamesPerMinute.toFixed(2)} games/min\n`
        );
      }
    },
  });

  const artifact = createHeadToHeadArtifact(run);
  const written = await writeHeadToHeadArtifacts(artifact, outDir);
  process.stdout.write(
    `${JSON.stringify(
      {
        status: 'completed',
        artifact: path.resolve(written.artifactPath),
        summary: path.resolve(written.summaryPath),
        results: run.summary,
      },
      null,
      2
    )}\n`
  );
}

function benchmarkConfig(options: Options): HeadToHeadConfig {
  const searchConfig = {
    worlds: options.worlds,
    rollouts: options.rollouts,
    depth: options.depth,
    maxRootActions: options.maxRootActions,
    rolloutEpsilon: options.rolloutEpsilon,
    heuristic: 'v2' as const,
  };
  const guidanceLabel = `root-${options.tdRoot}-rollout-${options.tdRollout}-leaf-${options.tdLeaf}`;
  const opponentLabel =
    options.opponent === 'td' ? 'td-root-all-td' : 'heuristic-v2';
  const opponent =
    options.opponent === 'td'
      ? ({
          id: 'td-root-medium-root-td-rollout-td-leaf-td',
          kind: 'td-root-search',
          modelIndexPath: selectedTdModelIndexPath(options),
          config: searchConfig,
          guidance: {
            root: 'td' as const,
            rollout: 'td' as const,
            leaf: 'td' as const,
          },
        } satisfies BotSpec)
      : ({
          id: 'heuristic-v2-medium',
          kind: 'search',
          config: searchConfig,
        } satisfies BotSpec);
  return {
    schemaVersion: 1,
    runLabel: `td-root-${guidanceLabel}-vs-${opponentLabel}-medium`,
    seedPrefix: `td-root-${guidanceLabel}-vs-${opponentLabel}-medium`,
    gamesPerSide: options.games / 2,
    maxDecisionsPerGame: options.maxDecisionsPerGame,
    candidate: {
      id: `td-root-medium-${guidanceLabel}`,
      kind: 'td-root-search',
      modelIndexPath: selectedTdModelIndexPath(options),
      config: searchConfig,
      guidance: {
        root: options.tdRoot,
        rollout: options.tdRollout,
        leaf: options.tdLeaf,
      },
    },
    opponent,
  };
}

function formatGameResult(game: PlayedGame, config: HeadToHeadConfig): string {
  const candidateSeat = seatForBot(game, config.candidate.id);
  const opponentSeat = seatForBot(game, config.opponent.id);
  const candidateResult =
    game.finalScore.winner === 'Draw'
      ? 'draw'
      : game.finalScore.winner === candidateSeat
        ? 'candidate-win'
        : 'opponent-win';
  const winnerBot =
    game.finalScore.winner === 'Draw'
      ? 'Draw'
      : game.botBySeat[game.finalScore.winner];
  return [
    '[td-vs-v2]',
    `game=${game.gameId}`,
    `candidateSeat=${candidateSeat}`,
    `opponentSeat=${opponentSeat}`,
    `result=${candidateResult}`,
    `winner=${winnerBot}`,
    `districts=${game.finalScore.districtPoints.PlayerA}-${game.finalScore.districtPoints.PlayerB}`,
    `ranks=${game.finalScore.rankTotals.PlayerA}-${game.finalScore.rankTotals.PlayerB}`,
    `resources=${game.finalScore.resourceTotals.PlayerA}-${game.finalScore.resourceTotals.PlayerB}`,
    `decidedBy=${game.finalScore.decidedBy}`,
    `turns=${String(game.turns)}`,
    `decisions=${String(game.transcript.length)}`,
    `elapsed=${formatSeconds(game.elapsedMs)}`,
  ].join(' ');
}

function seatForBot(game: PlayedGame, botId: string): PlayerId {
  if (game.botBySeat.PlayerA === botId) {
    return 'PlayerA';
  }
  if (game.botBySeat.PlayerB === botId) {
    return 'PlayerB';
  }
  throw new Error(`Game ${game.gameId} does not include bot ${botId}.`);
}

function formatSeconds(elapsedMs: number): string {
  return `${(elapsedMs / 1000).toFixed(1)}s`;
}

async function resolveTdManifestUrl(options: Options): Promise<string> {
  const publicRoot = path.resolve(process.cwd(), 'public');
  const indexPath = path.resolve(publicRoot, options.tdModelIndexPath);
  if (
    indexPath !== publicRoot &&
    !indexPath.startsWith(`${publicRoot}${path.sep}`)
  ) {
    throw new Error('--td-model-index-path must stay under public/.');
  }
  const index = JSON.parse(await readFile(indexPath, 'utf8')) as ModelPackIndex;
  const selectedPackId = options.tdPackId ?? index.defaultPackId;
  if (!selectedPackId) {
    throw new Error(
      'No TD pack id was provided and public/model-packs/index.json has no defaultPackId.'
    );
  }
  const selected = index.packs.find(
    (pack) =>
      pack.id === selectedPackId && pack.modelType === 'td-root-search-v1'
  );
  if (!selected) {
    throw new Error(
      `Could not find td-root-search-v1 pack id=${selectedPackId}.`
    );
  }
  return localPublicUrl(selected.manifestPath);
}

function selectedTdModelIndexPath(options: Options): string {
  if (!options.tdPackId) {
    return options.tdModelIndexPath;
  }
  const separator = options.tdModelIndexPath.includes('?') ? '&' : '?';
  return `${options.tdModelIndexPath}${separator}tdPackId=${encodeURIComponent(options.tdPackId)}`;
}

function parseOptions(args: readonly string[]): Options {
  const flags = new Map<string, string>();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith('--') || value === undefined) {
      throw new Error(`Invalid argument near ${String(key)}.`);
    }
    flags.set(key, value);
  }
  return {
    ...DEFAULT_OPTIONS,
    games: optionalInt(flags, '--games', DEFAULT_OPTIONS.games),
    worlds: optionalInt(flags, '--worlds', DEFAULT_OPTIONS.worlds),
    rollouts: optionalInt(flags, '--rollouts', DEFAULT_OPTIONS.rollouts),
    depth: optionalInt(flags, '--depth', DEFAULT_OPTIONS.depth),
    maxRootActions: optionalInt(
      flags,
      '--max-root-actions',
      DEFAULT_OPTIONS.maxRootActions
    ),
    rolloutEpsilon: optionalNumber(
      flags,
      '--rollout-epsilon',
      DEFAULT_OPTIONS.rolloutEpsilon
    ),
    tdRoot: optionalTdRootGuidanceSource(
      flags,
      '--td-root',
      DEFAULT_OPTIONS.tdRoot
    ),
    tdRollout: optionalTdRootGuidanceSource(
      flags,
      '--td-rollout',
      DEFAULT_OPTIONS.tdRollout
    ),
    tdLeaf: optionalTdRootGuidanceSource(
      flags,
      '--td-leaf',
      DEFAULT_OPTIONS.tdLeaf
    ),
    opponent: optionalOpponent(flags, '--opponent', DEFAULT_OPTIONS.opponent),
    tdModelIndexPath:
      flags.get('--td-model-index-path') ?? DEFAULT_OPTIONS.tdModelIndexPath,
    workers: optionalInt(flags, '--workers', DEFAULT_OPTIONS.workers),
    maxDecisionsPerGame: optionalInt(
      flags,
      '--max-decisions-per-game',
      DEFAULT_OPTIONS.maxDecisionsPerGame
    ),
    tdPackId: flags.get('--td-pack-id'),
    outDir: flags.get('--out-dir'),
    resume: optionalBoolean(flags, '--resume', DEFAULT_OPTIONS.resume),
    resumeKey: flags.get('--resume-key'),
    dryRun: optionalBoolean(flags, '--dry-run', DEFAULT_OPTIONS.dryRun),
  };
}

function prepareResume(
  options: Options,
  config: HeadToHeadConfig,
  outDir: string
): { runFingerprint: string; results: PairedSeedResult[]; elapsedMs: number } {
  if (!options.resume) {
    return { runFingerprint: '', results: [], elapsedMs: 0 };
  }
  if (!options.resumeKey?.trim()) {
    throw new Error('--resume-key is required when --resume true.');
  }
  const runFingerprint = createHash('sha256')
    .update(
      JSON.stringify({
        config,
        resumeKey: options.resumeKey,
      })
    )
    .digest('hex');
  const checkpointRoot = pairCheckpointRoot(outDir);
  mkdirSync(checkpointRoot, { recursive: true });
  const results = readdirSync(checkpointRoot)
    .filter((name) => /^pair-\d{4}\.json$/.test(name))
    .map((name) => {
      const checkpoint = JSON.parse(
        readFileSync(path.join(checkpointRoot, name), 'utf8')
      ) as PairCheckpoint;
      if (
        checkpoint.schemaVersion !== 1 ||
        checkpoint.runFingerprint !== runFingerprint ||
        checkpoint.pairIndex !== checkpoint.result.pairIndex
      ) {
        throw new Error(`Invalid or stale pair checkpoint: ${name}.`);
      }
      return checkpoint.result;
    });
  const statePath = resumeStatePath(outDir);
  let elapsedMs = 0;
  if (existsSync(statePath)) {
    const state = JSON.parse(readFileSync(statePath, 'utf8')) as ResumeState;
    if (
      state.schemaVersion !== 1 ||
      state.runFingerprint !== runFingerprint ||
      !Number.isFinite(state.elapsedMs) ||
      state.elapsedMs < 0 ||
      !Number.isInteger(state.completedPairs) ||
      state.completedPairs < 0 ||
      state.completedPairs > results.length
    ) {
      throw new Error('Invalid or stale resume-state.json.');
    }
    elapsedMs = state.elapsedMs;
  }
  return { runFingerprint, results, elapsedMs };
}

function writePairCheckpoint(
  outDir: string,
  runFingerprint: string,
  result: PairedSeedResult
): void {
  const pairNumber = result.pairIndex + 1;
  const target = path.join(
    pairCheckpointRoot(outDir),
    `pair-${String(pairNumber).padStart(4, '0')}.json`
  );
  writeJsonAtomic(target, {
    schemaVersion: 1,
    runFingerprint,
    pairIndex: result.pairIndex,
    result,
  } satisfies PairCheckpoint);
}

function writeResumeState(outDir: string, state: ResumeState): void {
  writeJsonAtomic(resumeStatePath(outDir), state);
}

function writeJsonAtomic(target: string, payload: unknown): void {
  mkdirSync(path.dirname(target), { recursive: true });
  const temporary = `${target}.${String(process.pid)}.tmp`;
  writeFileSync(temporary, `${JSON.stringify(payload)}\n`, 'utf8');
  renameSync(temporary, target);
}

function pairCheckpointRoot(outDir: string): string {
  return path.join(outDir, 'pair-checkpoints');
}

function resumeStatePath(outDir: string): string {
  return path.join(outDir, 'resume-state.json');
}

function optionalInt(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: number
): number {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return parsed;
}

function optionalNumber(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: number
): number {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${name} must be a finite number >= 0.`);
  }
  return parsed;
}

function optionalBoolean(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: boolean
): boolean {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === 'true') {
    return true;
  }
  if (raw === 'false') {
    return false;
  }
  throw new Error(`${name} must be true or false.`);
}

function optionalTdRootGuidanceSource(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: TdRootGuidanceSource
): TdRootGuidanceSource {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === 'td' || raw === 'heuristic') {
    return raw;
  }
  throw new Error(`${name} must be td or heuristic.`);
}

function optionalOpponent(
  flags: ReadonlyMap<string, string>,
  name: string,
  fallback: Options['opponent']
): Options['opponent'] {
  const raw = flags.get(name);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === 'heuristic-v2' || raw === 'td') {
    return raw;
  }
  throw new Error(`${name} must be heuristic-v2 or td.`);
}

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`[td-vs-v2] ${message}\n`);
  process.exitCode = 1;
});
