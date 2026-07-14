import { heuristicPolicy } from './heuristicPolicy';
import { randomPolicy } from './randomPolicy';
import {
  createSearchPolicy,
  type SearchHeuristicVersion,
  type SearchPolicyConfig,
} from './searchPolicy';
import {
  createTdRootSearchPolicy,
  type TdRootSearchPolicyOptions,
} from './tdRootSearchPolicy';
import type {
  TdRootGuidanceSource,
  TdRootSearchGuidanceConfig,
} from './tdRootGuidanceConfig';
import type { ActionPolicy } from './types';

export type BotKind =
  | 'random'
  | 'heuristic'
  | 'search'
  | 'td-root-search';

export interface RandomBotSpec {
  id: string;
  kind: 'random';
}

export interface HeuristicBotSpec {
  id: string;
  kind: 'heuristic';
}

export interface SearchBotSpec {
  id: string;
  kind: 'search';
  config: SearchPolicyConfig;
}

export interface TdRootSearchBotSpec {
  id: string;
  kind: 'td-root-search';
  config: SearchPolicyConfig;
  modelIndexPath?: string;
  guidance?: TdRootSearchGuidanceConfig;
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
    case 'random':
      return randomPolicy;
    case 'heuristic':
      return heuristicPolicy;
    case 'search':
      return createSearchPolicy(spec.config);
    case 'td-root-search':
      return createTdRootSearchPolicy({
        ...spec.config,
        modelIndexPath: spec.modelIndexPath,
        guidance: spec.guidance,
        loadModel: overrides.tdRootSearchLoadModel,
      });
  }
}

export function parseBotSpec(value: unknown, label = 'bot spec'): BotSpec {
  const source = requiredRecord(value, label);
  const id = requiredString(source.id, `${label}.id`);
  const kind = requiredString(source.kind, `${label}.kind`);

  switch (kind) {
    case 'random':
      return { id, kind };
    case 'heuristic':
      return { id, kind };
    case 'search':
      return {
        id,
        kind,
        config: parseSearchConfig(source.config, `${label}.config`),
      };
    case 'td-root-search':
      return optionalObjectProperties({
        id,
        kind,
        config: parseSearchConfig(source.config, `${label}.config`),
        modelIndexPath: optionalString(
          source.modelIndexPath,
          `${label}.modelIndexPath`
        ),
        guidance: parseOptionalTdRootGuidanceConfig(
          source.guidance,
          `${label}.guidance`
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
  return heuristic ? { ...config, heuristic } : config;
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

function optionalSearchHeuristic(
  value: unknown,
  label: string
): SearchHeuristicVersion | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === 'v1' || value === 'v2') {
    return value;
  }
  throw new Error(`${label} must be v1 or v2.`);
}

function parseOptionalTdRootGuidanceConfig(
  value: unknown,
  label: string
): TdRootSearchGuidanceConfig | undefined {
  if (value === undefined) {
    return undefined;
  }
  const source = requiredRecord(value, label);
  return optionalObjectProperties({
    root: optionalTdRootGuidanceSource(source.root, `${label}.root`),
    rollout: optionalTdRootGuidanceSource(source.rollout, `${label}.rollout`),
    leaf: optionalTdRootGuidanceSource(source.leaf, `${label}.leaf`),
  });
}

function optionalTdRootGuidanceSource(
  value: unknown,
  label: string
): TdRootGuidanceSource | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (value === 'td' || value === 'heuristic') {
    return value;
  }
  throw new Error(`${label} must be td or heuristic.`);
}

function optionalObjectProperties<T extends object>(value: T): T {
  return Object.fromEntries(
    Object.entries(value).filter((_entry): boolean => _entry[1] !== undefined)
  ) as T;
}
