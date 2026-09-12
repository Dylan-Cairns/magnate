import { describe, expect, it } from 'vitest';

import { BOT_PROFILES, getBotProfile } from './catalog';

describe('bot policy catalog', () => {
  it('throws when profile id is unknown', () => {
    expect(() => getBotProfile('unknown-profile')).toThrow(
      'Unknown bot profile'
    );
  });

  it('keeps all configured profiles available', () => {
    expect(BOT_PROFILES.length).toBe(4);
    expect(BOT_PROFILES.every((profile) => profile.available)).toBe(true);
  });

  it('exposes the expected labelled browser profiles', () => {
    expect(BOT_PROFILES.map((profile) => profile.label)).toEqual([
      'Easy',
      'Medium',
      'Hard',
      'Experimental',
    ]);
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

  it('includes the rollout-search v2 difficulty profiles', () => {
    const easy = getBotProfile('rollout-search-v2-easy');
    const medium = getBotProfile('rollout-search-v2-medium');
    const hard = getBotProfile('rollout-search-v2-hard');

    for (const profile of [easy, medium, hard]) {
      expect(profile.kind).toBe('search');
      expect(profile.spec.kind).toBe('search');
      if (profile.spec.kind !== 'search') {
        throw new Error('Expected rollout-search-v2 to use a search spec.');
      }
      expect(profile.spec.config.heuristic).toBe('v2');
    }

    expect(medium.spec.kind === 'search' && medium.spec.config).toEqual({
      worlds: 10,
      rollouts: 1,
      depth: 40,
      maxRootActions: 16,
      rolloutEpsilon: 0,
      heuristic: 'v2',
    });
  });

  it('includes an all-TD experimental profile', () => {
    const profile = getBotProfile('td-root-search-v2-medium');

    expect(profile.label).toBe('Experimental');
    expect(profile.kind).toBe('td-root-search');
    expect(profile.available).toBe(true);
    expect(profile.spec.kind).toBe('td-root-search');
    if (profile.spec.kind !== 'td-root-search') {
      throw new Error('Expected experimental profile to use a TD-root spec.');
    }
    expect(profile.spec.config).toEqual({
      worlds: 10,
      rollouts: 1,
      depth: 40,
      maxRootActions: 16,
      rolloutEpsilon: 0,
    });
    expect(profile.spec.modelIndexPath).toBeUndefined();
  });
});
