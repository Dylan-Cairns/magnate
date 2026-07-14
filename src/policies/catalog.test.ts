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
    expect(BOT_PROFILES.length).toBe(6);
    expect(BOT_PROFILES.some((profile) => profile.kind === 'search')).toBe(
      true
    );
    expect(
      BOT_PROFILES.every(
        (profile) =>
          profile.kind === 'search' || profile.kind === 'td-root-search'
      )
    ).toBe(true);
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

  it('includes the medium-hard heuristic v2 training profile', () => {
    const profile = getBotProfile('rollout-search-v2-medium-hard');

    expect(profile.label).toBe('Heuristic V2 Medium Hard');
    expect(profile.kind).toBe('search');
    expect(profile.available).toBe(true);
    expect(profile.spec.kind).toBe('search');
    if (profile.spec.kind !== 'search') {
      throw new Error('Expected medium-hard v2 profile to use a search spec.');
    }
    expect(profile.spec.config).toEqual({
      worlds: 40,
      rollouts: 1,
      depth: 180,
      maxRootActions: 16,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    });
  });

  it('includes a TD-root profile using heuristic v2 leaf evaluation', () => {
    const profile = getBotProfile('td-root-search-v2-medium-heuristic-leaf');

    expect(profile.kind).toBe('td-root-search');
    expect(profile.available).toBe(true);
    expect(profile.spec.kind).toBe('td-root-search');
    if (profile.spec.kind !== 'td-root-search') {
      throw new Error('Expected hybrid TD profile to use a TD-root spec.');
    }
    expect(profile.spec.config).toMatchObject({
      worlds: 10,
      rollouts: 1,
      depth: 40,
      maxRootActions: 16,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    });
    expect(profile.spec.guidance).toEqual({
      root: 'td',
      rollout: 'td',
      leaf: 'heuristic',
    });
  });
});
