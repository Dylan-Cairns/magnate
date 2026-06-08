import { createPolicyFromBotSpec, type BotKind, type BotSpec } from './botSpec';
import type { ActionPolicy } from './types';
import { createWorkerBackedPolicy } from './workerPolicy';

export type BotProfileId =
  | 'heuristic'
  | 'td-search-fast'
  | 'rollout-eval-search'
  | 'rollout-search-v2'
  | 'td-root-rollout-search'
  | 'td-search-browser'
  | 'random-legal';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: BotKind;
  available: boolean;
  turnDelayMs: number;
  spec: BotSpec;
  policy: ActionPolicy;
}

export interface ResolvedBotProfile {
  selected: BotProfile;
  policy: ActionPolicy;
  statusText: string;
}

export const BOT_PROFILES: readonly BotProfile[] = [
  createBotProfile({
    id: 'heuristic',
    label: 'Heuristic',
    description: 'Fast deterministic heuristic.',
    available: true,
    turnDelayMs: 250,
    spec: {
      id: 'heuristic',
      kind: 'heuristic',
    },
  }),
  createBotProfile({
    id: 'td-search-fast',
    label: 'TD Search Fast',
    description: '',
    available: true,
    turnDelayMs: 175,
    spec: {
      id: 'td-search-fast',
      kind: 'td-search',
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 6,
        maxRootActions: 4,
        rolloutEpsilon: 0.01,
        opponentTemperature: 1.0,
        sampleOpponentActions: false,
      },
    },
  }),
  createBotProfile({
    id: 'td-search-browser',
    label: 'TD Search',
    description: '',
    available: true,
    turnDelayMs: 450,
    spec: {
      id: 'td-search-browser',
      kind: 'td-search',
      config: {
        worlds: 6,
        rollouts: 1,
        depth: 14,
        maxRootActions: 6,
        rolloutEpsilon: 0.04,
        opponentTemperature: 1.0,
        sampleOpponentActions: false,
      },
    },
  }),
  createBotProfile({
    id: 'rollout-eval-search',
    label: 'Rollout Search',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-eval-search',
      kind: 'search',
      config: {
        worlds: 50,
        rollouts: 1,
        depth: 160,
        maxRootActions: 8,
        rolloutEpsilon: 0.0,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'rollout-search-v2',
    label: 'Rollout Search V2',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-search-v2',
      kind: 'search',
      config: {
        worlds: 50,
        rollouts: 1,
        depth: 270,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: 'v2',
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'td-root-rollout-search',
    label: 'TD Root Rollout Search',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'td-root-rollout-search',
      kind: 'td-root-search',
      config: {
        worlds: 50,
        rollouts: 1,
        depth: 160,
        maxRootActions: 8,
        rolloutEpsilon: 0.0,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'random-legal',
    label: 'Random legal',
    description: 'Random choice among legal actions.',
    available: true,
    turnDelayMs: 250,
    spec: {
      id: 'random-legal',
      kind: 'random',
    },
  }),
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'rollout-search-v2';

export function getBotProfile(id: string): BotProfile {
  const match = BOT_PROFILES.find((profile) => profile.id === id);
  if (match) {
    return match;
  }
  throw new Error(`Unknown bot profile: ${id}`);
}

export function resolveBotProfile(id: string): ResolvedBotProfile {
  const selected = getBotProfile(id);
  if (!selected.available) {
    throw new Error(`Bot profile is not available: ${id}`);
  }

  return {
    selected,
    policy: selected.policy,
    statusText: selected.description,
  };
}

function createBotProfile(
  profile: Omit<BotProfile, 'kind' | 'policy'> & {
    createPolicy?: (spec: BotSpec) => ActionPolicy;
  }
): BotProfile {
  const { createPolicy, ...profileConfig } = profile;
  return {
    ...profileConfig,
    kind: profileConfig.spec.kind,
    policy: (createPolicy ?? createPolicyFromBotSpec)(profileConfig.spec),
  };
}
