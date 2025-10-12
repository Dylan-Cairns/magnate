import { describe, expect, it } from 'vitest';

import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  getBotProfile,
  resolveBotProfile,
} from './catalog';

describe('bot policy catalog', () => {
  it('throws when profile id is unknown', () => {
    expect(() => getBotProfile('unknown-profile')).toThrow('Unknown bot profile');
  });

  it('uses champion profile as default and resolves an available policy', () => {
    const resolved = resolveBotProfile(DEFAULT_BOT_PROFILE_ID);
    expect(resolved.selected.available).toBe(true);
    expect(resolved.selected.kind).toBe('trained');
  });

  it('keeps all configured profiles available', () => {
    expect(BOT_PROFILES.length).toBeGreaterThanOrEqual(2);
    expect(BOT_PROFILES.every((profile) => profile.available)).toBe(true);
  });
});
