import {
  selectRolloutSearchActionSync,
  type RolloutSearchSelectionInput,
} from './rolloutSearchCore';
import {
  resolveSearchConfig,
  type SearchHeuristicVersion,
  type SearchPolicyConfig,
  type SearchPolicyOptions,
} from './searchConfig';
import type { ActionPolicy } from './types';

export type {
  SearchHeuristicVersion,
  SearchPolicyConfig,
  SearchPolicyOptions,
};

export function createSearchPolicy(
  options: SearchPolicyOptions = {}
): ActionPolicy {
  const config = resolveSearchConfig(options);
  return {
    selectAction({
      view,
      state,
      legalActions: candidateActions,
      random,
      randomSeed,
      onSearchDiagnostics,
      onProgress,
    }) {
      const input: RolloutSearchSelectionInput = {
        state,
        view,
        candidateActions,
        config,
        random,
        ...(randomSeed ? { randomSeed } : {}),
        onSearchDiagnostics,
        onProgress,
      };
      return selectRolloutSearchActionSync(input);
    },
  };
}
