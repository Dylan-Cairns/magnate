import { describe, expect, it } from 'vitest';

import { legacyAuthoritativeValue } from './shadowAuthority';

describe('outer-worker shadow authority', () => {
  it('always advances with the legacy result even when the candidate differs', () => {
    const legacy = { actionKey: 'legacy-action' };
    const candidate = { actionKey: 'candidate-action' };

    expect(legacyAuthoritativeValue(legacy, candidate)).toBe(legacy);
  });
});
