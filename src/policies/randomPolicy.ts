import type { ActionPolicy } from './types';

export const randomPolicy: ActionPolicy = {
  selectAction({ legalActions, random }) {
    if (legalActions.length === 0) {
      return undefined;
    }
    const index = Math.floor(random() * legalActions.length);
    return legalActions[index];
  },
};
