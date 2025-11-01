import { randomPolicy } from './randomPolicy';
import { createPpoBrowserPolicy } from './ppoBrowserPolicy';
import { createSearchPolicy } from './searchPolicy';
import type { ActionPolicy } from './types';

export type BotProfileId =
  | 'ppo-champion-2026-02-23-seed7'
  | 'search-t2-teacher'
  | 'search-t3-teacher'
  | 'random-legal';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: 'random' | 'trained' | 'search';
  available: boolean;
  policy: ActionPolicy;
}

export interface ResolvedBotProfile {
  selected: BotProfile;
  policy: ActionPolicy;
  statusText: string;
}

const CHAMPION_MODEL_URL = `${import.meta.env.BASE_URL}models/ppo_champion_2026-02-23_seed7.browser.json`;
const championPolicy = createPpoBrowserPolicy({
  modelUrl: CHAMPION_MODEL_URL,
});
const searchT2Policy = createSearchPolicy({
  worlds: 4,
  rollouts: 1,
  depth: 12,
  maxRootActions: 6,
  rolloutEpsilon: 0.1,
});
const searchT3Policy = createSearchPolicy({
  worlds: 6,
  rollouts: 1,
  depth: 14,
  maxRootActions: 6,
  rolloutEpsilon: 0.08,
});

export const BOT_PROFILES: readonly BotProfile[] = [
  {
    id: 'ppo-champion-2026-02-23-seed7',
    label: 'Champion PPO (2026-02-23 seed7)',
    description: 'Current best trained PPO checkpoint. Default bot profile.',
    kind: 'trained',
    available: true,
    policy: championPolicy,
  },
  {
    id: 'search-t2-teacher',
    label: 'Search Teacher T2 (Stronger, Slower)',
    description:
      'Determinized search: worlds=4, depth=12, rootActions=6, rolloutEpsilon=0.10. No fallback on failure.',
    kind: 'search',
    available: true,
    policy: searchT2Policy,
  },
  {
    id: 'search-t3-teacher',
    label: 'Search Teacher T3 (Strongest, Slowest)',
    description:
      'Determinized search: worlds=6, depth=14, rootActions=6, rolloutEpsilon=0.08. No fallback on failure.',
    kind: 'search',
    available: true,
    policy: searchT3Policy,
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

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'ppo-champion-2026-02-23-seed7';

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
