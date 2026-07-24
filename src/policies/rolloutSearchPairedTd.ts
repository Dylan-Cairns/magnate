import { toDecisionPlayerView } from '../engine/decisionActor';
import { toKeyedActions } from '../engine/actionSurface';
import type { GameAction, GameState } from '../engine/types';
import {
  createRolloutSearchVisitRunner,
  type RolloutSearchRuntimeGuidance,
  type RolloutSearchVisitRequest,
  type RolloutSearchVisitResult,
  type RolloutSearchVisitRunner,
  type RolloutSearchWorkerTask,
} from './rolloutSearchCore';
import type { LoadedTdGuidanceModel } from './tdGuidanceModel';
import { tdRolloutActionFromLogits } from './tdRootSearchPolicy';
import { encodeObservation } from './trainingEncoding';

export interface RolloutSearchResumableBatchResult {
  readonly results: readonly RolloutSearchVisitResult[];
  readonly pairedActionEvaluations: number;
  readonly scalarActionEvaluations: number;
  readonly leafEvaluations: number;
}

interface RunnerState {
  readonly runner: RolloutSearchVisitRunner;
  step: IteratorResult<RolloutSearchVisitRequest, RolloutSearchVisitResult>;
}

export function assertPairedTdRolloutAvailable(
  model: LoadedTdGuidanceModel | undefined,
  pairTdActions: boolean
): void {
  if (!pairTdActions) {
    throw new Error(
      'Paired TD rollout execution requires TD rollout guidance.'
    );
  }
  if (!model?.opponentScorer.logitsForActionPair) {
    throw new Error(
      'Paired TD rollout execution requires a paired opponent scorer.'
    );
  }
}

/**
 * Lockstep executor that preserves task creation, visit scheduling, UCB
 * allocation, merge order, and RNG streams. It only pauses the two visits
 * already assigned to one worker and evaluates coincident TD action requests
 * through the paired kernel.
 */
export function runRolloutSearchTaskBatchResumable(
  tasks: readonly RolloutSearchWorkerTask[],
  worldStates: readonly GameState[],
  guidance: RolloutSearchRuntimeGuidance | undefined,
  model: LoadedTdGuidanceModel | undefined,
  pairTdActions: boolean
): RolloutSearchResumableBatchResult {
  const results: RolloutSearchVisitResult[] = [];
  let pairedActionEvaluations = 0;
  let scalarActionEvaluations = 0;
  let leafEvaluations = 0;

  for (let index = 0; index < tasks.length; index += 2) {
    const pairTasks = tasks.slice(index, index + 2);
    const runners = pairTasks.map((task) => {
      const runner = createRolloutSearchVisitRunner(task, worldStates, {
        requestGuidedActions: guidance?.chooseRolloutAction !== undefined,
        requestGuidedLeaf: guidance?.evaluateLeaf !== undefined,
      });
      return { runner, step: runner.next() } satisfies RunnerState;
    });

    while (runners.some(({ step }) => !step.done)) {
      const first = runners[0];
      const second = runners[1];
      if (
        first &&
        second &&
        !first.step.done &&
        !second.step.done &&
        first.step.value.kind === 'guided-action' &&
        second.step.value.kind === 'guided-action' &&
        pairTdActions &&
        model?.opponentScorer.logitsForActionPair
      ) {
        const firstRequest = first.step.value;
        const secondRequest = second.step.value;
        const firstActions = toKeyedActions(firstRequest.actions).map(
          (candidate) => candidate.action
        );
        const secondActions = toKeyedActions(secondRequest.actions).map(
          (candidate) => candidate.action
        );
        const [firstLogits, secondLogits] =
          model.opponentScorer.logitsForActionPair(
            encodeObservation(
              toDecisionPlayerView(
                firstRequest.state,
                firstRequest.decisionPlayer
              )
            ),
            firstActions,
            encodeObservation(
              toDecisionPlayerView(
                secondRequest.state,
                secondRequest.decisionPlayer
              )
            ),
            secondActions
          );
        first.step = first.runner.next(
          tdRolloutActionFromLogits(firstActions, firstLogits)
        );
        second.step = second.runner.next(
          tdRolloutActionFromLogits(secondActions, secondLogits)
        );
        pairedActionEvaluations += 1;
        continue;
      }

      for (const state of runners) {
        if (state.step.done) {
          continue;
        }
        const request = state.step.value;
        const response = serviceScalarRequest(request, guidance);
        if (request.kind === 'guided-action') {
          scalarActionEvaluations += 1;
        } else {
          leafEvaluations += 1;
        }
        state.step = state.runner.next(response);
      }
    }
    for (const { step } of runners) {
      if (!step.done) {
        throw new Error('Resumable rollout search visit did not finish.');
      }
      results.push(step.value);
    }
  }

  return {
    results,
    pairedActionEvaluations,
    scalarActionEvaluations,
    leafEvaluations,
  };
}

function serviceScalarRequest(
  request: RolloutSearchVisitRequest,
  guidance: RolloutSearchRuntimeGuidance | undefined
): GameAction | number | undefined {
  if (request.kind === 'guided-action') {
    return guidance?.chooseRolloutAction?.(request);
  }
  return guidance?.evaluateLeaf?.(request);
}
