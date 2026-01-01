import { randomPolicy } from './randomPolicy';
import { createSearchPolicy } from './searchPolicy';
import { createTdSearchPolicy } from './tdSearchPolicy';
import type { ActionPolicy } from './types';

export type BotProfileId =
  | 'td-search-fast'
  | 'rollout-eval-search'
  | 'td-search-browser'
  | 'random-legal';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: 'random' | 'search' | 'td-search';
  available: boolean;
  turnDelayMs: number;
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
const tdSearchFastPolicy = createTdSearchPolicy({
  worlds: 1,
  rollouts: 1,
  depth: 6,
  maxRootActions: 4,
  rolloutEpsilon: 0.01,
  opponentTemperature: 1.0,
  sampleOpponentActions: false,
});
const tdSearchBrowserPolicy = createTdSearchPolicy({
  worlds: 6,
  rollouts: 1,
  depth: 14,
  maxRootActions: 6,
  rolloutEpsilon: 0.04,
  opponentTemperature: 1.0,
  sampleOpponentActions: false,
});

export const BOT_PROFILES: readonly BotProfile[] = [
  {
    id: 'td-search-fast',
    label: 'TD Search Fast',
    description: '',
    kind: 'td-search',
    available: true,
    turnDelayMs: 175,
    policy: tdSearchFastPolicy,
  },
  {
    id: 'td-search-browser',
    label: 'TD Search',
    description: '',
    kind: 'td-search',
    available: true,
    turnDelayMs: 450,
    policy: tdSearchBrowserPolicy,
  },
  {
    id: 'rollout-eval-search',
    label: 'Rollout Search',
    description: '',
    kind: 'search',
    available: true,
    turnDelayMs: 450,
    policy: rolloutEvalSearchPolicy,
  },
  {
    id: 'random-legal',
    label: 'Random legal',
    description: 'Random choice among legal actions.',
    kind: 'random',
    available: true,
    turnDelayMs: 250,
    policy: randomPolicy,
  },
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'td-search-fast';

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
