import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  createHeadToHeadArtifact,
  defaultHeadToHeadOutputDirectory,
  loadHeadToHeadArtifact,
  writeHeadToHeadArtifacts,
} from './artifacts';
import { parseHeadToHeadConfig, parseRolloutSearchSweepConfig } from './config';
import { runHeadToHead } from './matchup';
import { replayArtifactGame } from './replay';
import {
  defaultRolloutSearchSweepOutputDirectory,
  writeRolloutSearchSweepArtifacts,
} from './sweepArtifacts';
import { runRolloutSearchSweep } from './sweep';

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
    default:
      throw new Error(
        'Usage: yarn bot:eval head-to-head --config <path> [--out-dir <path>] | rollout-search-sweep --config <path> [--out-dir <path>] | replay --artifact <path> --game-id <id>'
      );
  }
}

async function runHeadToHeadCommand(args: readonly string[]): Promise<void> {
  const flags = parseFlags(args);
  const configPath = requiredFlag(flags, '--config');
  const config = parseHeadToHeadConfig(
    JSON.parse(await readFile(configPath, 'utf8'))
  );
  const outputDirectory =
    flags.get('--out-dir') ?? defaultHeadToHeadOutputDirectory(config.runLabel);
  const run = await runHeadToHead(config);
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
  const run = await runRolloutSearchSweep(config);
  const written = await writeRolloutSearchSweepArtifacts(run, outputDirectory);
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

void main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`[bot:eval] ${message}\n`);
  process.exitCode = 1;
});
