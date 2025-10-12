import { describe, expect, it } from 'vitest';

import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  getBotProfile,
  resolveBotProfile,
} from './catalog';
import { randomPolicy } from './randomPolicy';

describe('bot policy catalog', () => {
  it('defaults to random profile when id is missing or unknown', () => {
    expect(getBotProfile(undefined).id).toBe(DEFAULT_BOT_PROFILE_ID);
    expect(getBotProfile('unknown-profile').id).toBe(DEFAULT_BOT_PROFILE_ID);
  });

  it('keeps random policy for available random profile', () => {
    const resolved = resolveBotProfile('random-legal');
    expect(resolved.selected.available).toBe(true);
    expect(resolved.usingFallback).toBe(false);
    expect(resolved.policy).toBe(randomPolicy);
  });

  it('falls back to random policy for unavailable trained profiles', () => {
    const trainedProfiles = BOT_PROFILES.filter((profile) => profile.kind === 'trained');
    expect(trainedProfiles.length).toBeGreaterThan(0);

    trainedProfiles.forEach((profile) => {
      const resolved = resolveBotProfile(profile.id);
      expect(resolved.selected.available).toBe(false);
      expect(resolved.usingFallback).toBe(true);
      expect(resolved.policy).toBe(randomPolicy);
      expect(resolved.statusText).toContain('random legal fallback');
    });
  });
});
