import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  createHeadToHeadArtifact,
  defaultHeadToHeadOutputDirectory,
  writeHeadToHeadArtifacts,
} from '../src/botEval/artifacts';
import { runHeadToHead } from '../src/botEval/matchup';
import type { HeadToHeadConfig } from '../src/botEval/types';
import { createPolicyFromBotSpec, type BotSpec } from '../src/policies/botSpec';
import { createTdRootSearchPolicy } from '../src/policies/tdRootSearchPolicy';
import { loadTdRootModelFromManifestUrl } from '../src/policies/tdRootModelPack';
import type { LoadedTdGuidanceModel } from '../src/policies/tdGuidanceModel';
import type { TdRootGuidanceSource } from '../src/policies/tdRootGuidanceConfig';
import type { ActionPolicy } from '../src/policies/types';

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
  tdPackId?: string;
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
  maxDecisionsPerGame: 260,
};

async function main(): Promise<void> {
  const options = parseOptions(process.argv.slice(2));
  if (options.games % 2 !== 0) {
    throw new Error('--games must be even because this benchmark side-swaps paired seeds.');
  }

  const manifestUrl = await resolveTdManifestUrl(options);
  const modelCache = new Map<string, Promise<LoadedTdGuidanceModel>>();
  const createPolicy = (spec: BotSpec): ActionPolicy => {
    if (spec.kind !== 'td-root-search') {
      return createPolicyFromBotSpec(spec);
    }
    return createTdRootSearchPolicy({
      ...spec.config,
      guidance: spec.guidance,
      loadModel: () => {
        const cached = modelCache.get(manifestUrl);
        if (cached) {
          return cached;
        }
        const loaded = loadTdRootModelFromManifestUrl(manifestUrl);
        modelCache.set(manifestUrl, loaded);
        return loaded;
      },
    });
  };

  const config = benchmarkConfig(options);
  const outDir =
    options.outDir ?? defaultHeadToHeadOutputDirectory(config.runLabel);
  process.stderr.write(
    `[td-vs-v2] games=${String(options.games)} worlds=${String(options.worlds)} depth=${String(options.depth)} maxRootActions=${String(options.maxRootActions)} tdRoot=${options.tdRoot} tdRollout=${options.tdRollout} tdLeaf=${options.tdLeaf} tdManifest=${manifestUrl}\n`
  );

  const run = await runHeadToHead(config, {
    workers: 1,
    createPolicy,
    progressIntervalMs: 30_000,
    onProgress(progress) {
      if (progress.type === 'pair-completed') {
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
  return {
    schemaVersion: 1,
    runLabel: `td-root-${guidanceLabel}-vs-heuristic-v2-medium`,
    seedPrefix: `td-root-${guidanceLabel}-vs-heuristic-v2-medium`,
    gamesPerSide: options.games / 2,
    maxDecisionsPerGame: options.maxDecisionsPerGame,
    candidate: {
      id: `td-root-medium-${guidanceLabel}`,
      kind: 'td-root-search',
      config: searchConfig,
      guidance: {
        root: options.tdRoot,
        rollout: options.tdRollout,
        leaf: options.tdLeaf,
      },
    },
    opponent: {
      id: 'heuristic-v2-medium',
      kind: 'search',
      config: searchConfig,
    },
  };
}

async function resolveTdManifestUrl(options: Options): Promise<string> {
  installLocalPublicFetch();
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

function installLocalPublicFetch(): void {
  globalThis.fetch = async (input: Parameters<typeof fetch>[0]) => {
    const url = new URL(String(input));
    if (url.protocol !== 'http:' || url.hostname !== 'localhost') {
      throw new Error(`Unexpected benchmark fetch URL: ${url.toString()}`);
    }
    const relativePath = url.pathname.replace(/^\/+/, '');
    const payload = JSON.parse(
      await readFile(path.join(process.cwd(), 'public', relativePath), 'utf8')
    ) as unknown;
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      async json() {
        return payload;
      },
    } as Response;
  };
}

function localPublicUrl(publicRelativePath: string): string {
  return `http://localhost/${publicRelativePath.replace(/^\/+/, '')}`;
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

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`[td-vs-v2] ${message}\n`);
  process.exitCode = 1;
});
