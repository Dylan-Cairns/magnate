import { BotProfileId, SearchHeuristicVersion } from './values';
import { Ruleset } from '../engine/types';
import { createPolicyFromBotSpec, BotKind, type BotSpec } from './botSpec';
import type { ActionPolicy } from './types';
import { createWorkerBackedPolicy } from './workerPolicy';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: BotKind;
  available: boolean;
  supportedRulesets: readonly Ruleset[];
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
    id: BotProfileId.RolloutSearchV2Easy,
    label: 'Easy',
    description: '',
    available: true,
    supportedRulesets: [Ruleset.Standard, Ruleset.Extended],
    turnDelayMs: 0,
    spec: {
      id: BotProfileId.RolloutSearchV2Easy,
      kind: BotKind.Search,
      config: {
        worlds: 20,
        rollouts: 1,
        depth: 80,
        maxRootActions: 10,
        rolloutEpsilon: 0.0,
        heuristic: SearchHeuristicVersion.V2,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: BotProfileId.RolloutSearchV2Medium,
    label: 'Medium',
    description: '',
    available: true,
    supportedRulesets: [Ruleset.Standard, Ruleset.Extended],
    turnDelayMs: 0,
    spec: {
      id: BotProfileId.RolloutSearchV2Medium,
      kind: BotKind.Search,
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 40,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: SearchHeuristicVersion.V2,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: BotProfileId.RolloutSearchV2Hard,
    label: 'Hard',
    description: '',
    available: true,
    supportedRulesets: [Ruleset.Standard, Ruleset.Extended],
    turnDelayMs: 0,
    spec: {
      id: BotProfileId.RolloutSearchV2Hard,
      kind: BotKind.Search,
      config: {
        worlds: 50,
        rollouts: 1,
        depth: 270,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
        heuristic: SearchHeuristicVersion.V2,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
  createBotProfile({
    id: BotProfileId.TdRootSearchV2Medium,
    label: 'Experimental',
    description: '',
    available: true,
    supportedRulesets: [Ruleset.Standard],
    turnDelayMs: 0,
    spec: {
      id: BotProfileId.TdRootSearchV2Medium,
      kind: BotKind.TdRootSearch,
      config: {
        worlds: 10,
        rollouts: 1,
        depth: 270,
        maxRootActions: 16,
        rolloutEpsilon: 0.0,
      },
    },
    createPolicy: createWorkerBackedPolicy,
  }),
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId =
  BotProfileId.RolloutSearchV2Easy;

export function getBotProfile(id: string): BotProfile {
  const match = BOT_PROFILES.find((profile) => profile.id === id);
  if (match) {
    return match;
  }
  throw new Error(`Unknown bot profile: ${id}`);
}

export function profilesForRuleset(ruleset: Ruleset): readonly BotProfile[] {
  return BOT_PROFILES.filter(
    (profile) =>
      profile.available && profile.supportedRulesets.includes(ruleset)
  );
}

export function botProfileSupportsRuleset(
  id: string,
  ruleset: Ruleset
): boolean {
  return getBotProfile(id).supportedRulesets.includes(ruleset);
}

export function defaultBotProfileIdForRuleset(ruleset: Ruleset): BotProfileId {
  const profiles = profilesForRuleset(ruleset);
  const defaultProfile = profiles.find(
    (profile) => profile.id === DEFAULT_BOT_PROFILE_ID
  );
  const fallback = profiles[0];
  if (!defaultProfile && !fallback) {
    throw new Error(`No bot profiles support the ${ruleset} ruleset.`);
  }
  return (defaultProfile ?? fallback).id;
}

export function resolveBotProfile(
  id: string,
  ruleset?: Ruleset
): ResolvedBotProfile {
  const selected = getBotProfile(id);
  if (!selected.available) {
    throw new Error(`Bot profile is not available: ${id}`);
  }
  if (ruleset && !selected.supportedRulesets.includes(ruleset)) {
    throw new Error(
      `Bot profile ${id} is not available for the ${ruleset} ruleset.`
    );
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

export { BotProfileId } from './values';
