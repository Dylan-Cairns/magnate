import type { ActionPolicy } from './types';
import { botCandidateActions } from './policyUtils';

export const randomPolicy: ActionPolicy = {
  selectAction({ legalActions, random }) {
    const candidates = botCandidateActions(legalActions);
    if (candidates.length === 0) {
      return undefined;
    }
    const index = Math.floor(random() * candidates.length);
    return candidates[index];
  },
};
