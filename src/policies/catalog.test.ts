import { describe, expect, it } from 'vitest';

import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  getBotProfile,
  resolveBotProfile,
} from './catalog';
import { randomPolicy } from './randomPolicy';

describe('bot policy catalog', () => {
  it('throws when profile id is unknown', () => {
    expect(() => getBotProfile('unknown-profile')).toThrow('Unknown bot profile');
  });

  it('keeps random policy for available random profile', () => {
    const resolved = resolveBotProfile(DEFAULT_BOT_PROFILE_ID);
    expect(resolved.selected.available).toBe(true);
    expect(resolved.policy).toBe(randomPolicy);
  });

  it('throws for unavailable trained profiles', () => {
    const trainedProfiles = BOT_PROFILES.filter((profile) => profile.kind === 'trained');
    expect(trainedProfiles.length).toBeGreaterThan(0);

    trainedProfiles.forEach((profile) => {
      expect(() => resolveBotProfile(profile.id)).toThrow('not available');
    });
  });
});
