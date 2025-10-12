import { randomPolicy } from './randomPolicy';
import { createPpoBrowserPolicy } from './ppoBrowserPolicy';
import type { ActionPolicy } from './types';

export type BotProfileId = 'ppo-champion-2026-02-23-seed7' | 'random-legal';

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

const CHAMPION_MODEL_URL = `${import.meta.env.BASE_URL}models/ppo_champion_2026-02-23_seed7.browser.json`;
const championPolicy = createPpoBrowserPolicy({
  modelUrl: CHAMPION_MODEL_URL,
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
