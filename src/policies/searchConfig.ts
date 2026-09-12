export interface SearchPolicyConfig {
  worlds: number;
  rollouts: number;
  depth: number;
  maxRootActions: number;
  rolloutEpsilon: number;
  heuristic?: SearchHeuristicVersion;
  /**
   * Heuristic v2 district-potential weight granted to a deed before any
   * progress tokens. 0 reproduces the pre-floor scoring behavior and is the
   * control arm for benchmark A/B runs.
   */
  deedPotentialBase?: number;
}

export type SearchHeuristicVersion = 'v1' | 'v2';
export type SearchPolicyOptions = Partial<SearchPolicyConfig>;

export const DEFAULT_DEED_POTENTIAL_BASE = 0.2;

export const DEFAULT_SEARCH_POLICY_CONFIG: SearchPolicyConfig = {
  worlds: 4,
  rollouts: 1,
  depth: 12,
  maxRootActions: 6,
  rolloutEpsilon: 0.04,
  heuristic: 'v1',
  deedPotentialBase: DEFAULT_DEED_POTENTIAL_BASE,
};

export function resolveSearchConfig(
  options: SearchPolicyOptions = {}
): SearchPolicyConfig {
  const worlds = integerWithFloor(
    options.worlds ?? DEFAULT_SEARCH_POLICY_CONFIG.worlds,
    1
  );
  const rollouts = integerWithFloor(
    options.rollouts ?? DEFAULT_SEARCH_POLICY_CONFIG.rollouts,
    1
  );
  const depth = integerWithFloor(
    options.depth ?? DEFAULT_SEARCH_POLICY_CONFIG.depth,
    1
  );
  const maxRootActions = integerWithFloor(
    options.maxRootActions ?? DEFAULT_SEARCH_POLICY_CONFIG.maxRootActions,
    1
  );
  const rolloutEpsilon =
    options.rolloutEpsilon ?? DEFAULT_SEARCH_POLICY_CONFIG.rolloutEpsilon;
  const heuristic = options.heuristic ?? DEFAULT_SEARCH_POLICY_CONFIG.heuristic;
  const deedPotentialBase =
    options.deedPotentialBase ??
    DEFAULT_SEARCH_POLICY_CONFIG.deedPotentialBase ??
    DEFAULT_DEED_POTENTIAL_BASE;
  if (
    !Number.isFinite(rolloutEpsilon) ||
    rolloutEpsilon < 0 ||
    rolloutEpsilon > 1
  ) {
    throw new Error(
      `Search policy rolloutEpsilon must be in [0, 1]; received ${String(rolloutEpsilon)}.`
    );
  }
  if (heuristic !== 'v1' && heuristic !== 'v2') {
    throw new Error(
      `Search policy heuristic must be v1 or v2; received ${String(heuristic)}.`
    );
  }
  if (
    !Number.isFinite(deedPotentialBase) ||
    deedPotentialBase < 0 ||
    deedPotentialBase > 1
  ) {
    throw new Error(
      `Search policy deedPotentialBase must be in [0, 1]; received ${String(deedPotentialBase)}.`
    );
  }
  return {
    worlds,
    rollouts,
    depth,
    maxRootActions,
    rolloutEpsilon,
    heuristic,
    deedPotentialBase,
  };
}

function integerWithFloor(value: number, floor: number): number {
  if (!Number.isFinite(value)) {
    throw new Error(
      `Search policy expected a finite number; received ${String(value)}.`
    );
  }
  const rounded = Math.trunc(value);
  if (rounded < floor) {
    throw new Error(
      `Search policy value must be >= ${String(floor)}; received ${String(value)}.`
    );
  }
  return rounded;
}
