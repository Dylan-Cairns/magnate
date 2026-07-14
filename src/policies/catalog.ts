import { createPolicyFromBotSpec, type BotKind, type BotSpec } from './botSpec';
import type { ActionPolicy } from './types';
import { createWorkerBackedPolicy } from './workerPolicy';

export type BotProfileId =
  | 'rollout-search-v2-hard'
  | 'rollout-search-v2-medium-hard'
  | 'rollout-search-v2-medium'
  | 'td-root-search-v2-medium'
  | 'td-root-search-v2-medium-heuristic-leaf'
  | 'rollout-search-v2-easy';

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
    id: 'rollout-search-v2-hard',
    label: 'V2 Hard',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-search-v2-hard',
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
    id: 'rollout-search-v2-medium-hard',
    label: 'Heuristic V2 Medium Hard',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-search-v2-medium-hard',
      kind: 'search',
      config: {
        worlds: 40,
        rollouts: 1,
        depth: 180,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: 'v2',
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'rollout-search-v2-medium',
    label: 'V2 Medium',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-search-v2-medium',
      kind: 'search',
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: 'v2',
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'td-root-search-v2-medium',
    label: 'TD V2 Medium',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'td-root-search-v2-medium',
      kind: 'td-root-search',
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'td-root-search-v2-medium-heuristic-leaf',
    label: 'TD V2 Medium + Heuristic Leaf',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'td-root-search-v2-medium-heuristic-leaf',
      kind: 'td-root-search',
      guidance: {
        root: 'td',
        rollout: 'td',
        leaf: 'heuristic',
      },
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: 'v2',
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: 'rollout-search-v2-easy',
    label: 'V2 Easy',
    description: '',
    available: true,
    turnDelayMs: 0,
    spec: {
      id: 'rollout-search-v2-easy',
      kind: 'search',
      config: {
        worlds: 20,
        rollouts: 1,
        depth: 80,
        maxRootActions: 10,
        rolloutEpsilon: 0.0,
        heuristic: 'v2',
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'rollout-search-v2-hard';

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
