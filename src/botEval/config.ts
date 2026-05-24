import { getBotProfile } from '../policies/catalog';
import { parseBotSpec, type BotSpec } from '../policies/botSpec';
import {
  HEAD_TO_HEAD_CONFIG_SCHEMA_VERSION,
  ROLLOUT_SEARCH_SWEEP_CONFIG_SCHEMA_VERSION,
  TD_REPLAY_CONFIG_SCHEMA_VERSION,
  type HeadToHeadConfig,
  type RolloutSearchSweepConfig,
  type TdReplayConfig,
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

export function parseRolloutSearchSweepConfig(
  value: unknown
): RolloutSearchSweepConfig {
  const source = requiredRecord(value, 'rollout-search sweep config');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'rollout-search sweep config.schemaVersion'
  );
  if (schemaVersion !== ROLLOUT_SEARCH_SWEEP_CONFIG_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported rollout-search sweep config schemaVersion=${String(schemaVersion)}.`
    );
  }

  const opponent = parseBotReference(
    source.opponent,
    'rollout-search sweep config.opponent'
  );
  const candidates = requiredArray(
    source.candidates,
    'rollout-search sweep config.candidates'
  ).map((candidate, index) => {
    const parsed = parseBotSpec(
      candidate,
      `rollout-search sweep config.candidates[${String(index)}]`
    );
    if (parsed.kind !== 'search') {
      throw new Error(
        `rollout-search sweep config.candidates[${String(index)}] must have kind search.`
      );
    }
    return parsed;
  });
  if (candidates.length === 0) {
    throw new Error(
      'rollout-search sweep config.candidates must include at least one candidate.'
    );
  }

  const ids = new Set<string>([opponent.id]);
  for (const candidate of candidates) {
    if (ids.has(candidate.id)) {
      throw new Error(
        `rollout-search sweep bot ids must be distinct; duplicate=${candidate.id}.`
      );
    }
    ids.add(candidate.id);
  }

  return {
    schemaVersion,
    runLabel: requiredString(
      source.runLabel,
      'rollout-search sweep config.runLabel'
    ),
    seedPrefix: requiredString(
      source.seedPrefix,
      'rollout-search sweep config.seedPrefix'
    ),
    gamesPerSide: requiredPositiveInteger(
      source.gamesPerSide,
      'rollout-search sweep config.gamesPerSide'
    ),
    opponent,
    candidates,
    maxDecisionsPerGame: optionalPositiveInteger(
      source.maxDecisionsPerGame,
      'rollout-search sweep config.maxDecisionsPerGame'
    ),
  };
}

export function parseTdReplayConfig(value: unknown): TdReplayConfig {
  const source = requiredRecord(value, 'TD replay config');
  const schemaVersion = requiredInteger(
    source.schemaVersion,
    'TD replay config.schemaVersion'
  );
  if (schemaVersion !== TD_REPLAY_CONFIG_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported TD replay config schemaVersion=${String(schemaVersion)}.`
    );
  }

  return {
    schemaVersion,
    runLabel: requiredString(source.runLabel, 'TD replay config.runLabel'),
    seedPrefix: requiredString(
      source.seedPrefix,
      'TD replay config.seedPrefix'
    ),
    games: requiredPositiveInteger(source.games, 'TD replay config.games'),
    playerA: parseTdReplayBotReference(
      source.playerA,
      'TD replay config.playerA'
    ),
    playerB: parseTdReplayBotReference(
      source.playerB,
      'TD replay config.playerB'
    ),
    policyTargetAlpha: optionalPositiveNumber(
      source.policyTargetAlpha,
      'TD replay config.policyTargetAlpha'
    ),
    maxDecisionsPerGame: optionalPositiveInteger(
      source.maxDecisionsPerGame,
      'TD replay config.maxDecisionsPerGame'
    ),
  };
}

function parseBotReference(value: unknown, label: string): BotSpec {
  const source = requiredRecord(value, label);
  if ('profileId' in source) {
    const profileId = requiredString(source.profileId, `${label}.profileId`);
    return structuredClone(getBotProfile(profileId).spec);
  }
  return parseBotSpec(source, label);
}

function parseTdReplayBotReference(value: unknown, label: string): BotSpec {
  const spec = parseBotReference(value, label);
  if (spec.kind === 'td-search' || spec.kind === 'td-root-search') {
    throw new Error(
      `${label}.kind ${spec.kind} is not supported by collect-td-replay yet; use random, heuristic, or search.`
    );
  }
  return spec;
}

function requiredArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array.`);
  }
  return value;
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

function optionalPositiveNumber(
  value: unknown,
  label: string
): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be a positive finite number.`);
  }
  return value;
}
