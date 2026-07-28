import { describe, expect, it } from 'vitest';

import type { BotSpec } from './botSpec';
import {
  resolveEffectiveSearchExecutionMode,
  searchWorkerPoolConfigurationMatches,
  validateSearchExecutionMode,
} from './searchExecutionMode';

const TD_SPEC: BotSpec = {
  id: 'execution-mode-td',
  kind: 'td-root-search',
  config: {
    worlds: 2,
    rollouts: 1,
    depth: 4,
    maxRootActions: 2,
    rolloutEpsilon: 0,
  },
};

describe('search execution mode validation', () => {
  it('keeps omitted and legacy execution available to ordinary policies', () => {
    expect(() =>
      validateSearchExecutionMode(TD_SPEC, undefined, 1)
    ).not.toThrow();
    expect(() =>
      validateSearchExecutionMode({ ...TD_SPEC, kind: 'search' }, 'legacy', 1)
    ).not.toThrow();
  });

  it('accepts paired execution only for parallel TD rollout search', () => {
    expect(() =>
      validateSearchExecutionMode(TD_SPEC, 'resumable-paired-td', 2)
    ).not.toThrow();
    expect(() =>
      validateSearchExecutionMode(
        { ...TD_SPEC, kind: 'search' },
        'resumable-paired-td',
        2
      )
    ).toThrow('requires a TD-root search policy');
    expect(() =>
      validateSearchExecutionMode(TD_SPEC, 'resumable-paired-td', 1)
    ).toThrow('requires parallel search workers');
    expect(() =>
      validateSearchExecutionMode(
        {
          ...TD_SPEC,
          guidance: { rollout: 'heuristic' },
        },
        'resumable-paired-td',
        2
      )
    ).toThrow('requires TD rollout guidance');
  });

  it('rejects unknown structured-clone payload values', () => {
    expect(() =>
      validateSearchExecutionMode(TD_SPEC, 'future-mode', 2)
    ).toThrow('Unsupported search execution mode');
  });

  it('defaults eligible parallel TD rollout search to paired execution', () => {
    expect(resolveEffectiveSearchExecutionMode(TD_SPEC, undefined, 2)).toBe(
      'resumable-paired-td'
    );
  });

  it('honors an explicit legacy rollback for eligible search', () => {
    expect(resolveEffectiveSearchExecutionMode(TD_SPEC, 'legacy', 2)).toBe(
      'legacy'
    );
  });

  it('leaves ineligible and synchronous search behavior unchanged', () => {
    expect(
      resolveEffectiveSearchExecutionMode(
        { ...TD_SPEC, guidance: { rollout: 'heuristic' } },
        undefined,
        2
      )
    ).toBeUndefined();
    expect(
      resolveEffectiveSearchExecutionMode(
        { ...TD_SPEC, kind: 'search' },
        undefined,
        2
      )
    ).toBeUndefined();
    expect(
      resolveEffectiveSearchExecutionMode(TD_SPEC, undefined, 1)
    ).toBeUndefined();
  });

  it('requires pool recreation when worker count or execution mode changes', () => {
    expect(searchWorkerPoolConfigurationMatches(8, 'legacy', 8, 'legacy')).toBe(
      true
    );
    expect(
      searchWorkerPoolConfigurationMatches(
        8,
        'legacy',
        8,
        'resumable-paired-td'
      )
    ).toBe(false);
    expect(
      searchWorkerPoolConfigurationMatches(
        8,
        'resumable-paired-td',
        4,
        'resumable-paired-td'
      )
    ).toBe(false);
  });
});
