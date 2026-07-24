import { describe, expect, it } from 'vitest';

import { searchExecutionModeOverrideFromSearch } from './searchExecutionRuntime';

describe('browser search execution mode override', () => {
  it('leaves the production default in effect when the parameter is absent', () => {
    expect(searchExecutionModeOverrideFromSearch('')).toBeUndefined();
    expect(
      searchExecutionModeOverrideFromSearch('?unrelated=value')
    ).toBeUndefined();
  });

  it('maps the supported rollback and explicit paired values', () => {
    expect(
      searchExecutionModeOverrideFromSearch('?tdSearchExecutor=legacy')
    ).toBe('legacy');
    expect(
      searchExecutionModeOverrideFromSearch('?tdSearchExecutor=paired')
    ).toBe('resumable-paired-td');
  });

  it('rejects invalid values instead of silently changing execution', () => {
    expect(() =>
      searchExecutionModeOverrideFromSearch('?tdSearchExecutor=future')
    ).toThrow('tdSearchExecutor must be legacy or paired; received future.');
  });
});
