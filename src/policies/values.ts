// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const BotKind = {
  Random: 'random',
  Heuristic: 'heuristic',
  Search: 'search',
  TdRootSearch: 'td-root-search',
} as const;
export type BotKind = (typeof BotKind)[keyof typeof BotKind];

export const BotProfileId = {
  RolloutSearchV2Easy: 'rollout-search-v2-easy',
  RolloutSearchV2Medium: 'rollout-search-v2-medium',
  RolloutSearchV2Hard: 'rollout-search-v2-hard',
  TdRootSearchV2Medium: 'td-root-search-v2-medium',
} as const;
export type BotProfileId = (typeof BotProfileId)[keyof typeof BotProfileId];

export const SearchHeuristicVersion = {
  V1: 'v1',
  V2: 'v2',
} as const;
export type SearchHeuristicVersion =
  (typeof SearchHeuristicVersion)[keyof typeof SearchHeuristicVersion];

export const RolloutSearchGuidanceKind = {
  Heuristic: 'heuristic',
  TdRoot: 'td-root',
} as const;
export type RolloutSearchGuidanceKind =
  (typeof RolloutSearchGuidanceKind)[keyof typeof RolloutSearchGuidanceKind];
