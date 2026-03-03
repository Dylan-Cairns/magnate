import { heuristicPolicy } from './heuristicPolicy';
import { randomPolicy } from './randomPolicy';
import { createSearchPolicy, type SearchPolicyConfig } from './searchPolicy';
import {
  createTdSearchPolicy,
  type TdSearchPolicyConfig,
  type TdSearchPolicyOptions,
} from './tdSearchPolicy';
import type { ActionPolicy } from './types';

export type BotKind = 'random' | 'heuristic' | 'search' | 'td-search';

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

export interface TdSearchBotSpec {
  id: string;
  kind: 'td-search';
  config: TdSearchPolicyConfig;
  modelIndexPath?: string;
}

export type BotSpec =
  | RandomBotSpec
  | HeuristicBotSpec
  | SearchBotSpec
  | TdSearchBotSpec;

export interface BotPolicyRuntimeOverrides {
  tdSearchLoadModel?: TdSearchPolicyOptions['loadModel'];
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
    case 'td-search':
      return createTdSearchPolicy({
        ...spec.config,
        modelIndexPath: spec.modelIndexPath,
        loadModel: overrides.tdSearchLoadModel,
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
    case 'td-search':
      return {
        id,
        kind,
        config: parseTdSearchConfig(source.config, `${label}.config`),
        modelIndexPath: optionalString(
          source.modelIndexPath,
          `${label}.modelIndexPath`
        ),
      };
    default:
      throw new Error(
        `${label}.kind must be random, heuristic, search, or td-search.`
      );
  }
}

function parseSearchConfig(value: unknown, label: string): SearchPolicyConfig {
  const source = requiredRecord(value, label);
  return {
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
}

function parseTdSearchConfig(
  value: unknown,
  label: string
): TdSearchPolicyConfig {
  const source = requiredRecord(value, label);
  return {
    ...parseSearchConfig(source, label),
    opponentTemperature: requiredPositiveNumber(
      source.opponentTemperature,
      `${label}.opponentTemperature`
    ),
    sampleOpponentActions: requiredBoolean(
      source.sampleOpponentActions,
      `${label}.sampleOpponentActions`
    ),
  };
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

function requiredPositiveNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number.`);
  }
  return value;
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

function requiredBoolean(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean.`);
  }
  return value;
}
