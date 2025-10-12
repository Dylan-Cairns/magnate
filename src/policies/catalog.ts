import { randomPolicy } from './randomPolicy';
import type { ActionPolicy } from './types';

export type BotProfileId =
  | 'random-legal'
  | 'trained-baseline'
  | 'trained-conservative';

export interface BotProfile {
  id: BotProfileId;
  label: string;
  description: string;
  kind: 'random' | 'trained';
  available: boolean;
  policy: ActionPolicy;
}

export interface ResolvedBotProfile {
  selected: BotProfile;
  policy: ActionPolicy;
  statusText: string;
}

export const BOT_PROFILES: readonly BotProfile[] = [
  {
    id: 'random-legal',
    label: 'Random legal',
    description: 'Uniform random choice among legal actions.',
    kind: 'random',
    available: true,
    policy: randomPolicy,
  },
  {
    id: 'trained-baseline',
    label: 'Trained baseline (coming soon)',
    description: 'Checkpoint profile placeholder for the baseline trained bot.',
    kind: 'trained',
    available: false,
    policy: randomPolicy,
  },
  {
    id: 'trained-conservative',
    label: 'Trained conservative (coming soon)',
    description: 'Checkpoint profile placeholder for a conservative trained bot.',
    kind: 'trained',
    available: false,
    policy: randomPolicy,
  },
];

export const DEFAULT_BOT_PROFILE_ID: BotProfileId = 'random-legal';

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
