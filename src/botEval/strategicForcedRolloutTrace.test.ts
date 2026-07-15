import { describe, expect, it } from 'vitest';

import type { SearchPolicyConfig } from '../policies/searchConfig';
import type { LoadedTdGuidanceModel } from '../policies/tdGuidanceModel';
import {
  ACTION_FEATURE_DIM,
  OBSERVATION_DIM,
} from '../policies/trainingEncoding';
import { createStrategicPositionCatalogV0 } from './strategicPositionCatalog';
import { runStrategicForcedRolloutTraceV0 } from './strategicForcedRolloutTrace';

const TEST_CONFIG: SearchPolicyConfig = {
  worlds: 2,
  rollouts: 1,
  depth: 40,
  maxRootActions: 16,
  rolloutEpsilon: 0,
  heuristic: 'v2',
};

describe('strategic forced rollout trace', () => {
  it('keeps all four cells matched and produces deterministic terminal traces', async () => {
    const positions = createStrategicPositionCatalogV0().filter((position) =>
      [
        'known-hand-optionality-original',
        'known-hand-optionality-mirror',
      ].includes(position.id)
    );
    const options = {
      positions,
      repetitionIds: [7],
      scenarioIndices: [0, 1, 2],
      config: TEST_CONFIG,
      modelIndexPath: 'test-model-index.json',
      loadModel: async () => tieBreakingModel(),
    } as const;

    const first = await runStrategicForcedRolloutTraceV0(options);
    const second = await runStrategicForcedRolloutTraceV0(options);

    expect(second).toEqual(first);
    expect(first.positions).toHaveLength(2);
    expect(first.config).toEqual(TEST_CONFIG);
    for (const position of first.positions) {
      expect(position.repetitions).toHaveLength(1);
      expect(position.repetitions[0].scenarios).toHaveLength(3);
      for (const scenario of position.repetitions[0].scenarios) {
        expect(scenario.traces).toHaveLength(4);
        expect(
          new Set(
            scenario.traces.map(
              (trace) => `${trace.rootFocusActionId}:${trace.guide}`
            )
          )
        ).toEqual(
          new Set([
            'preserve-option:td',
            'preserve-option:heuristic-v2',
            'overwrite-option:td',
            'overwrite-option:heuristic-v2',
          ])
        );
        for (const trace of scenario.traces) {
          expect(trace.terminatedBeforeDepthLimit).toBe(true);
          expect(trace.finalScore).toBeDefined();
          expect(trace.steps).toHaveLength(trace.simulatedActionSteps);
          expect(trace.steps[0].actionKey).toBe(
            position.focusActions.find(
              (focus) => focus.id === trace.rootFocusActionId
            )?.actionKey
          );
          for (const step of trace.steps.slice(1)) {
            expect(step.proposals).not.toBeNull();
            expect(step.actionKey).toBe(
              trace.guide === 'td'
                ? step.proposals!.td.actionKey
                : step.proposals!.heuristicV2.actionKey
            );
          }
        }
      }
    }

    for (const scenarioIndex of [0, 1, 2]) {
      const original = first.positions[0].repetitions[0].scenarios.find(
        (scenario) => scenario.scenarioIndex === scenarioIndex
      );
      const mirror = first.positions[1].repetitions[0].scenarios.find(
        (scenario) => scenario.scenarioIndex === scenarioIndex
      );
      expect(mirror?.engineSeed).toBe(original?.engineSeed);
      expect(mirror?.rolloutRandomSeed).toBe(original?.rolloutRandomSeed);
      expect(mirror?.hiddenAssignmentFingerprint).toBe(
        original?.hiddenAssignmentFingerprint
      );
    }
    for (const position of first.positions) {
      const scenarios = position.repetitions[0].scenarios;
      expect(scenarios[2].worldIndex).toBe(0);
      expect(scenarios[2].hiddenAssignmentFingerprint).toBe(
        scenarios[0].hiddenAssignmentFingerprint
      );
      expect(scenarios[2].engineSeed).not.toBe(scenarios[0].engineSeed);
      expect(scenarios[2].rolloutRandomSeed).not.toBe(
        scenarios[0].rolloutRandomSeed
      );
    }
  });

  it('rejects trace configs that do not describe one deterministic rollout', async () => {
    const position = createStrategicPositionCatalogV0().find(
      (entry) => entry.id === 'known-hand-optionality-original'
    );
    if (!position) {
      throw new Error('Missing known-hand optionality position.');
    }
    await expect(
      runStrategicForcedRolloutTraceV0({
        positions: [position],
        repetitionIds: [0],
        scenarioIndices: [0],
        config: { ...TEST_CONFIG, rolloutEpsilon: 0.1 },
        loadModel: async () => tieBreakingModel(),
      })
    ).rejects.toThrow('rolloutEpsilon=0');

    await expect(
      runStrategicForcedRolloutTraceV0({
        positions: [position],
        repetitionIds: [0],
        scenarioIndices: [0],
        config: { ...TEST_CONFIG, rollouts: 2 },
        loadModel: async () => tieBreakingModel(),
      })
    ).rejects.toThrow('rollouts=1');
  });

  it('reads holdout targets and semantic lanes from catalog metadata', async () => {
    const positions = createStrategicPositionCatalogV0().filter((position) =>
      [
        'known-hand-optionality-holdout-original',
        'known-hand-optionality-holdout-mirror',
        'unknown-pool-optionality-holdout-original',
        'unknown-pool-optionality-holdout-mirror',
      ].includes(position.id)
    );
    const run = await runStrategicForcedRolloutTraceV0({
      positions,
      repetitionIds: [0],
      scenarioIndices: [0],
      config: TEST_CONFIG,
      modelIndexPath: 'test-model-index.json',
      loadModel: async () => tieBreakingModel(),
    });

    expect(
      run.positions.map((position) => ({
        positionId: position.positionId,
        family: position.family,
        targetCardId: position.targetCardId,
        valuableDistrictId: position.valuableDistrictId,
        alternativeDistrictId: position.alternativeDistrictId,
      }))
    ).toEqual([
      {
        positionId: 'known-hand-optionality-holdout-original',
        family: 'known-hand',
        targetCardId: '8',
        valuableDistrictId: 'D1',
        alternativeDistrictId: 'D5',
      },
      {
        positionId: 'known-hand-optionality-holdout-mirror',
        family: 'known-hand',
        targetCardId: '8',
        valuableDistrictId: 'D5',
        alternativeDistrictId: 'D1',
      },
      {
        positionId: 'unknown-pool-optionality-holdout-original',
        family: 'unknown-pool',
        targetCardId: '19',
        valuableDistrictId: 'D4',
        alternativeDistrictId: 'D5',
      },
      {
        positionId: 'unknown-pool-optionality-holdout-mirror',
        family: 'unknown-pool',
        targetCardId: '19',
        valuableDistrictId: 'D5',
        alternativeDistrictId: 'D4',
      },
    ]);
    for (const [leftIndex, rightIndex] of [
      [0, 1],
      [2, 3],
    ] as const) {
      const left = run.positions[leftIndex].repetitions[0].scenarios[0];
      const right = run.positions[rightIndex].repetitions[0].scenarios[0];
      expect(right.engineSeed).toBe(left.engineSeed);
      expect(right.rolloutRandomSeed).toBe(left.rolloutRandomSeed);
      expect(right.hiddenAssignmentFingerprint).toBe(
        left.hiddenAssignmentFingerprint
      );
    }
  });
});

function tieBreakingModel(): LoadedTdGuidanceModel {
  return {
    valueScorer: {
      observationDim: OBSERVATION_DIM,
      predict() {
        return 0;
      },
    },
    opponentScorer: {
      observationDim: OBSERVATION_DIM,
      actionFeatureDim: ACTION_FEATURE_DIM,
      logits(_observation, actionFeatures) {
        return new Float32Array(actionFeatures.length);
      },
      logitsForActions(_observation, actions) {
        return new Float32Array(actions.length);
      },
    },
  };
}
