import { rename, writeFile } from 'node:fs/promises';
import path from 'node:path';

import {
  ROOT_ACTION_COUNT_BUCKETS,
  type LatencySummary,
  type RootActionCountBucket,
} from './types';

export function defaultBotEvalOutputDirectory(
  runLabel: string,
  generatedAt = new Date()
): string {
  const stamp = generatedAt
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d{3}Z$/, 'Z');
  return path.join('artifacts', 'ts-bot-evals', `${stamp}-${slug(runLabel)}`);
}

export async function writeAtomic(
  targetPath: string,
  contents: string
): Promise<void> {
  const tempPath = `${targetPath}.tmp`;
  await writeFile(tempPath, contents, 'utf8');
  await rename(tempPath, targetPath);
}

export function appendRootActionLatencyTable(
  lines: string[],
  title: string,
  buckets: Readonly<Record<RootActionCountBucket, LatencySummary>>
): void {
  lines.push(
    '',
    title,
    '',
    '| legal root actions | decisions | mean ms | p50 ms | p95 ms | max ms |',
    '|:---|---:|---:|---:|---:|---:|'
  );
  for (const bucket of ROOT_ACTION_COUNT_BUCKETS) {
    const latency = buckets[bucket];
    lines.push(
      `| ${bucket} | ${latency.actions} | ${format(latency.meanMs)} | ${format(latency.p50Ms)} | ${format(latency.p95Ms)} | ${format(latency.maxMs)} |`
    );
  }
}

export function format(value: number): string {
  return value.toFixed(3);
}

export function slug(value: string): string {
  return (
    value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'run'
  );
}

export function requiredRecord(
  value: unknown,
  label: string
): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}
