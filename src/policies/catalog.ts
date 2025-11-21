import { randomPolicy } from './randomPolicy';
import { createSearchPolicy } from './searchPolicy';
import type { ActionPolicy } from './types';

export type BotProfileId = 'rollout-eval-search' | 'random-legal';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: 'random' | 'search';
  available: boolean;
  policy: ActionPolicy;
}

export interface ResolvedBotProfile {
  selected: BotProfile;
  policy: ActionPolicy;
  statusText: string;
}

const rolloutEvalSearchPolicy = createSearchPolicy({
  worlds: 6,
  rollouts: 1,
  depth: 14,
  maxRootActions: 6,
  rolloutEpsilon: 0.08,
});

export const BOT_PROFILES: readonly BotProfile[] = [
  {
    id: 'rollout-eval-search',
    label: 'Rollout Eval Search',
    description:
      'Default bot. Determinized search: worlds=6, depth=14, rootActions=6, rolloutEpsilon=0.08. No fallback on failure.',
    kind: 'search',
    available: true,
    policy: rolloutEvalSearchPolicy,
  },
  {
    id: 'random-legal',
    label: 'Random legal',
    description: 'Uniform random choice among legal actions.',
    kind: 'random',
    available: true,
    policy: randomPolicy,
  },
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'rollout-eval-search';

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
