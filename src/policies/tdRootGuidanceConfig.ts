import type { RolloutSearchGuidanceKind } from './rolloutSearchCore';

export type TdRootGuidanceSource = 'td' | 'heuristic';

export interface TdRootSearchGuidanceConfig {
  root?: TdRootGuidanceSource;
  rollout?: TdRootGuidanceSource;
  leaf?: TdRootGuidanceSource;
}

export interface ResolvedTdRootSearchGuidanceConfig {
  root: TdRootGuidanceSource;
  rollout: TdRootGuidanceSource;
  leaf: TdRootGuidanceSource;
}

export const DEFAULT_TD_ROOT_SEARCH_GUIDANCE: ResolvedTdRootSearchGuidanceConfig =
  {
    root: 'td',
    rollout: 'td',
    leaf: 'td',
  };

export function resolveTdRootSearchGuidanceConfig(
  config: TdRootSearchGuidanceConfig | undefined
): ResolvedTdRootSearchGuidanceConfig {
  return {
    root: resolveTdRootGuidanceSource(config?.root, 'root'),
    rollout: resolveTdRootGuidanceSource(config?.rollout, 'rollout'),
    leaf: resolveTdRootGuidanceSource(config?.leaf, 'leaf'),
  };
}

export function tdRootGuidanceUsesModel(
  config: ResolvedTdRootSearchGuidanceConfig
): boolean {
  return config.root === 'td' || config.rollout === 'td' || config.leaf === 'td';
}

export function tdRootGuidanceUsesWorkerModel(
  config: ResolvedTdRootSearchGuidanceConfig
): boolean {
  return config.rollout === 'td' || config.leaf === 'td';
}

export function tdRootGuidanceKind(
  config: ResolvedTdRootSearchGuidanceConfig
): RolloutSearchGuidanceKind {
  if (
    config.root === 'td' &&
    config.rollout === 'td' &&
    config.leaf === 'td'
  ) {
    return 'td-root';
  }
  if (
    config.root === 'heuristic' &&
    config.rollout === 'heuristic' &&
    config.leaf === 'heuristic'
  ) {
    return 'heuristic';
  }
  return 'custom';
}

function resolveTdRootGuidanceSource(
  value: TdRootGuidanceSource | undefined,
  label: keyof ResolvedTdRootSearchGuidanceConfig
): TdRootGuidanceSource {
  if (value === undefined) {
    return DEFAULT_TD_ROOT_SEARCH_GUIDANCE[label];
  }
  if (value === 'td' || value === 'heuristic') {
    return value;
  }
  throw new Error(
    `TD root search guidance ${label} must be td or heuristic; received ${String(value)}.`
  );
}
