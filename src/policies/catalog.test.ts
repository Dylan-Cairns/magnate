import { describe, expect, it } from 'vitest';

import { BOT_PROFILES, getBotProfile } from './catalog';

describe('bot policy catalog', () => {
  it('throws when profile id is unknown', () => {
    expect(() => getBotProfile('unknown-profile')).toThrow(
      'Unknown bot profile'
    );
  });

  it('keeps all configured profiles available', () => {
    expect(BOT_PROFILES.length).toBeGreaterThanOrEqual(2);
    expect(BOT_PROFILES.every((profile) => profile.available)).toBe(true);
  });

  it('includes only rollout-search profiles for the active browser catalog', () => {
    expect(BOT_PROFILES.length).toBe(3);
    expect(BOT_PROFILES.some((profile) => profile.kind === 'search')).toBe(
      true
    );
    expect(BOT_PROFILES.every((profile) => profile.kind === 'search')).toBe(
      true
    );
  });

  it('records serializable specs for every configured profile', () => {
    expect(
      BOT_PROFILES.every(
        (profile) =>
          profile.spec.id === profile.id && profile.spec.kind === profile.kind
      )
    ).toBe(true);
  });

  it('includes a rollout-search v2 profile using the v2 heuristic', () => {
    const profile = getBotProfile('rollout-search-v2-medium');

    expect(profile.kind).toBe('search');
    expect(profile.available).toBe(true);
    expect(profile.spec.kind).toBe('search');
    if (profile.spec.kind !== 'search') {
      throw new Error('Expected rollout-search-v2 to use a search spec.');
    }
    expect(profile.spec.config.heuristic).toBe('v2');
  });
});
