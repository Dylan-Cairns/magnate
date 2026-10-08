import { BotKind } from './values';
import { heuristicPolicy } from './heuristicPolicy';
import { randomPolicy } from './randomPolicy';
import {
  createSearchPolicy,
  SearchHeuristicVersion,
  type SearchPolicyConfig,
} from './searchPolicy';
import {
  createTdRootSearchPolicy,
  type TdRootSearchPolicyOptions,
} from './tdRootSearchPolicy';
import type { ActionPolicy } from './types';

export interface RandomBotSpec {
  id: string;
  kind: typeof BotKind.Random;
}

export interface HeuristicBotSpec {
  id: string;
  kind: typeof BotKind.Heuristic;
}

export interface SearchBotSpec {
  id: string;
  kind: typeof BotKind.Search;
  config: SearchPolicyConfig;
}

export interface TdRootSearchBotSpec {
  id: string;
  kind: typeof BotKind.TdRootSearch;
  config: SearchPolicyConfig;
  modelIndexPath?: string;
}

export type BotSpec =
  | RandomBotSpec
  | HeuristicBotSpec
  | SearchBotSpec
  | TdRootSearchBotSpec;

export interface BotPolicyRuntimeOverrides {
  tdRootSearchLoadModel?: TdRootSearchPolicyOptions['loadModel'];
}

export function createPolicyFromBotSpec(
  spec: BotSpec,
  overrides: BotPolicyRuntimeOverrides = {}
): ActionPolicy {
  switch (spec.kind) {
    case BotKind.Random:
      return randomPolicy;
    case BotKind.Heuristic:
      return heuristicPolicy;
    case BotKind.Search:
      return createSearchPolicy(spec.config);
    case BotKind.TdRootSearch:
      return createTdRootSearchPolicy({
        ...spec.config,
        modelIndexPath: spec.modelIndexPath,
        loadModel: overrides.tdRootSearchLoadModel,
      });
  }
}

export function parseBotSpec(value: unknown, label = 'bot spec'): BotSpec {
  const source = requiredRecord(value, label);
  const id = requiredString(source.id, `${label}.id`);
  const kind = requiredString(source.kind, `${label}.kind`);

  switch (kind) {
    case BotKind.Random:
      return { id, kind };
    case BotKind.Heuristic:
      return { id, kind };
    case BotKind.Search:
      return {
        id,
        kind,
        config: parseSearchConfig(source.config, `${label}.config`),
      };
    case BotKind.TdRootSearch:
      return optionalObjectProperties({
        id,
        kind,
        config: parseSearchConfig(source.config, `${label}.config`),
        modelIndexPath: optionalString(
          source.modelIndexPath,
          `${label}.modelIndexPath`
        ),
      });
    default:
      throw new Error(
        `${label}.kind must be random, heuristic, search, or td-root-search.`
      );
  }
}

function parseSearchConfig(value: unknown, label: string): SearchPolicyConfig {
  const source = requiredRecord(value, label);
  const config: SearchPolicyConfig = {
    worlds: requiredPositiveInteger(source.worlds, `${label}.worlds`),
    rollouts: requiredPositiveInteger(source.rollouts, `${label}.rollouts`),
    depth: requiredPositiveInteger(source.depth, `${label}.depth`),
    maxRootActions: requiredPositiveInteger(
      source.maxRootActions,
      `${label}.maxRootActions`
    ),
    rolloutEpsilon: requiredProbability(
      source.rolloutEpsilon,
      `${label}.rolloutEpsilon`
    ),
  };
  const heuristic = optionalSearchHeuristic(
    source.heuristic,
    `${label}.heuristic`
  );
  const courtValueScale = optionalNonnegativeNumber(
    source.courtValueScale,
    `${label}.courtValueScale`
  );
  return optionalObjectProperties({
    ...config,
    ...(heuristic ? { heuristic } : {}),
    ...(courtValueScale !== undefined ? { courtValueScale } : {}),
  });
}

function requiredRecord(
  value: unknown,
  label: string
): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  return requiredString(value, label);
}

function requiredPositiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value as number;
}

function requiredProbability(value: unknown, label: string): number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > 1
  ) {
    throw new Error(`${label} must be a finite number in [0, 1].`);
  }
  return value;
}

function optionalNonnegativeNumber(
  value: unknown,
  label: string
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a finite number >= 0.`);
  }
  return value;
}

function optionalSearchHeuristic(
  value: unknown,
  label: string
): SearchHeuristicVersion | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (
    value === SearchHeuristicVersion.V1 ||
    value === SearchHeuristicVersion.V2
  ) {
    return value;
  }
  throw new Error(`${label} must be v1 or v2.`);
}

function optionalObjectProperties<T extends object>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter((_entry): boolean => _entry[1] !== undefined)
  ) as T;
}

export { BotKind } from './values';
