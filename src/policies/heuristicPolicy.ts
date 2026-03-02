import { selectHeuristicAction } from './heuristicScorer';
import type { ActionPolicy } from './types';

export const heuristicPolicy: ActionPolicy = {
  selectAction(context) {
    return selectHeuristicAction(context);
  },
};
