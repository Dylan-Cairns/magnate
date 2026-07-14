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
  tdModelIndexPath,
} from '../src/botEval/localPublicFetch';
import { runHeadToHead } from '../src/botEval/matchup';
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
  workers: number;
  maxDecisionsPerGame: number;
  outDir?: string;
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
  workers: 1,
  maxDecisionsPerGame: 260,
};

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  if (options.games % 2 !== 0) {
    throw new Error('--games must be even because this benchmark side-swaps paired seeds.');
  }

  installLocalPublicFetch();
  const manifestUrl = await resolveTdManifestUrl(options);
  const config = benchmarkConfig(options);
  const outDir =
    options.outDir ?? defaultHeadToHeadOutputDirectory(config.runLabel);
  process.stderr.write(
    `[td-vs-v2] games=${String(options.games)} workers=${String(options.workers)} worlds=${String(options.worlds)} depth=${String(options.depth)} maxRootActions=${String(options.maxRootActions)} tdRoot=${options.tdRoot} tdRollout=${options.tdRollout} tdLeaf=${options.tdLeaf} tdManifest=${manifestUrl}\n`
  );

  const run = await runHeadToHead(config, {
    workers: options.workers,
    progressIntervalMs: 30_000,
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
          modelIndexPath: tdModelIndexPath(options.tdPackId),
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
      modelIndexPath: tdModelIndexPath(options.tdPackId),
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

function formatGameResult(
  game: PlayedGame,
  config: HeadToHeadConfig
): string {
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
  const indexPath = path.join(process.cwd(), 'public', 'model-packs', 'index.json');
  const index = JSON.parse(await readFile(indexPath, 'utf8')) as ModelPackIndex;
  const selectedPackId = options.tdPackId ?? index.defaultPackId;
  if (!selectedPackId) {
    throw new Error('No TD pack id was provided and public/model-packs/index.json has no defaultPackId.');
  }
  const selected = index.packs.find(
    (pack) => pack.id === selectedPackId && pack.modelType === 'td-root-search-v1'
  );
  if (!selected) {
    throw new Error(`Could not find td-root-search-v1 pack id=${selectedPackId}.`);
  }
  return localPublicUrl(selected.manifestPath);
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
    workers: optionalInt(flags, '--workers', DEFAULT_OPTIONS.workers),
    maxDecisionsPerGame: optionalInt(
      flags,
      '--max-decisions-per-game',
      DEFAULT_OPTIONS.maxDecisionsPerGame
    ),
    tdPackId: flags.get('--td-pack-id'),
    outDir: flags.get('--out-dir'),
  };
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
