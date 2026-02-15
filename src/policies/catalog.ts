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
  usingFallback: boolean;
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

export function getBotProfile(id: string | undefined): BotProfile {
  const match = BOT_PROFILES.find((profile) => profile.id === id);
  if (match) {
    return match;
  }

  const fallback = BOT_PROFILES[0];
  if (!fallback) {
    throw new Error('Bot profile catalog is empty.');
  }
  return fallback;
}

export function resolveBotProfile(id: string | undefined): ResolvedBotProfile {
  const selected = getBotProfile(id);
  const usingFallback = !selected.available;
  const policy = usingFallback ? randomPolicy : selected.policy;
  const statusText = usingFallback
    ? `${selected.description} Using random legal fallback until the model is integrated.`
    : selected.description;

  return {
    selected,
    policy,
    usingFallback,
    statusText,
  };
}
