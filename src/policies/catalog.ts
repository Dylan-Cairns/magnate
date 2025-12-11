import { randomPolicy } from './randomPolicy';
import { createSearchPolicy } from './searchPolicy';
import { createTdValuePolicy } from './tdValuePolicy';
import type { ActionPolicy } from './types';

export type BotProfileId =
  | 'rollout-eval-search'
  | 'td-value-browser'
  | 'random-legal';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: 'random' | 'search' | 'td-value';
  available: boolean;
  policy: ActionPolicy;
}

export interface ResolvedBotProfile {
  selected: BotProfile;
  policy: ActionPolicy;
  statusText: string;
}

const rolloutEvalSearchPolicy = createSearchPolicy({
  worlds: 64,
  rollouts: 8,
  depth: 28,
  maxRootActions: 12,
  rolloutEpsilon: 0.0,
});
const tdValueBrowserPolicy = createTdValuePolicy({
  worlds: 8,
});

export const BOT_PROFILES: readonly BotProfile[] = [
  {
    id: 'rollout-eval-search',
    label: 'Rollout Eval Search',
    description:
      'Default bot. Determinized search: worlds=96, depth=32, rootActions=14, rolloutEpsilon=0.0, rollouts=12. No fallback on failure.',
    kind: 'search',
    available: true,
    policy: rolloutEvalSearchPolicy,
  },
  {
    id: 'td-value-browser',
    label: 'TD Value (Browser)',
    description:
      'Loads exported TD value model pack from public/model-packs and scores legal actions across determinized worlds.',
    kind: 'td-value',
    available: true,
    policy: tdValueBrowserPolicy,
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
