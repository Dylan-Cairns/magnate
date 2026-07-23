import { describe, expect, it } from 'vitest';

import { legalActions } from '../engine/actionBuilders';
import { actionStableKey, toKeyedActions } from '../engine/actionSurface';
import { isTerminal } from '../engine/scoring';
import {
  createSession,
  stepKnownLegalActionToDecisionForSimulation,
} from '../engine/session';
import type { GameAction } from '../engine/types';
import {
  runRolloutSearchTask,
  type RolloutSearchWorkerTask,
} from './rolloutSearchCore';
import { runRolloutSearchTaskBatchResumable } from './rolloutSearchPairedTd';
import { createTdRootSearchRolloutGuidance } from './tdRootSearchPolicy';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import { ACTION_FEATURE_DIM, OBSERVATION_DIM } from './trainingEncoding';

describe('paired TD rollout search executor', () => {
  it('matches the legacy task oracle exactly while using paired action inference', () => {
    const world = createSession('paired-td-rollout-world', 'PlayerB');
    const rootPlayer = 'PlayerB' as const;
    const rootAction = toKeyedActions(legalActions(world)).find((candidate) => {
      const next = stepKnownLegalActionToDecisionForSimulation(
        world,
        candidate.action
      );
      return !isTerminal(next) && legalActions(next).length > 0;
    });
    if (!rootAction) {
      throw new Error('Expected a root action with a follow-up decision.');
    }

    let pairCalls = 0;
    const logits = (
      _observation: readonly number[],
      actions: readonly GameAction[]
    ) => Float32Array.from(actions, (_action, index) => -index);
    const model: LoadedTdGuidanceModel = {
      valueScorer: {
        observationDim: OBSERVATION_DIM,
        predict() {
          return 0.125;
        },
      },
      opponentScorer: {
        observationDim: OBSERVATION_DIM,
        actionFeatureDim: ACTION_FEATURE_DIM,
        logits() {
          throw new Error('Test scorer expects direct actions.');
        },
        logitsForActions: logits,
        logitsForActionPair(observationA, actionsA, observationB, actionsB) {
          pairCalls += 1;
          expect(actionsA.map(actionStableKey)).toEqual(
            actionsA
              .map(actionStableKey)
              .sort((left, right) => left.localeCompare(right))
          );
          expect(actionsB.map(actionStableKey)).toEqual(
            actionsB
              .map(actionStableKey)
              .sort((left, right) => left.localeCompare(right))
          );
          return [
            logits(observationA, actionsA),
            logits(observationB, actionsB),
          ];
        },
      },
    };
    const guidance = createTdRootSearchRolloutGuidance({ model });
    const baseTask: RolloutSearchWorkerTask = {
      kind: 'rollout-search',
      contextId: 'paired-td-rollout-context',
      visitIndex: 0,
      actionVisitIndex: 0,
      scenarioIndex: 0,
      worldIndex: 0,
      engineSeed: 'paired-td-rollout-engine',
      rootPlayer,
      rootAction: rootAction.action,
      rootActionKey: rootAction.actionKey,
      config: {
        worlds: 1,
        rollouts: 1,
        depth: 8,
        maxRootActions: 1,
        rolloutEpsilon: 0,
        heuristic: 'v2',
      },
      randomSeed: 'paired-td-rollout-random',
    };
    const tasks = [baseTask, { ...baseTask, visitIndex: 1 }];

    const legacy = tasks.map((task) =>
      runRolloutSearchTask(task, [world], undefined, guidance)
    );
    const paired = runRolloutSearchTaskBatchResumable(
      tasks,
      [world],
      guidance,
      model,
      true
    );

    expect(paired.results).toEqual(legacy);
    expect(paired.pairedActionEvaluations).toBeGreaterThan(0);
    expect(paired.scalarActionEvaluations).toBe(0);
    expect(pairCalls).toBe(paired.pairedActionEvaluations);
  });
});
