import { getBotProfile } from '../policies/catalog';
import { parseBotSpec, type BotSpec } from '../policies/botSpec';
import {
  HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION,
  type HeadToHeadConfig,
} from './types';

export function parseHeadToHeadConfig(value: unknown): HeadToHeadConfig {
  const source = requiredRecord(value, 'head-to-head config');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'head-to-head config.schemaVersion'
  );
  if (schemaVersion !== HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported head-to-head config schemaVersion=${String(schemaVersion)}.`
    );
  }

  const config: HeadToHeadConfig = {
    schemaVersion,
    runLabel: requiredString(source.runLabel, 'head-to-head config.runLabel'),
    seedPrefix: requiredString(
      source.seedPrefix,
      'head-to-head config.seedPrefix'
    ),
    gamesPerSide: requiredPositiveInteger(
      source.gamesPerSide,
      'head-to-head config.gamesPerSide'
    ),
    candidate: parseBotReference(
      source.candidate,
      'head-to-head config.candidate'
    ),
    opponent: parseBotReference(
      source.opponent,
      'head-to-head config.opponent'
    ),
    maxDecisionsPerGame: optionalPositiveInteger(
      source.maxDecisionsPerGame,
      'head-to-head config.maxDecisionsPerGame'
    ),
  };

  if (config.candidate.id === config.opponent.id) {
    throw new Error('candidate and opponent bot ids must be distinct.');
  }
  return config;
}

function parseBotReference(value: unknown, label: string): BotSpec {
  const source = requiredRecord(value, label);
  if ('profileId' in source) {
    const profileId = requiredString(source.profileId, `${label}.profileId`);
    return structuredClone(getBotProfile(profileId).spec);
  }
  return parseBotSpec(source, label);
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

function requiredInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value)) {
    throw new Error(`${label} must be an integer.`);
  }
  return value as number;
}

function requiredPositiveInteger(value: unknown, label: string): number {
  const integer = requiredInteger(value, label);
  if (integer <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return integer;
}

function optionalPositiveInteger(
  value: unknown,
  label: string
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  return requiredPositiveInteger(value, label);
}
