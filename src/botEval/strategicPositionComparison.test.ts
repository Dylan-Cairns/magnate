import { describe, expect, it } from 'vitest';

import { actionStableKey } from '../engine/actionSurface';
import { legalActionsForDecisionPlayer } from '../engine/decisionActor';
import type { GameAction } from '../engine/types';
import { getBotProfile } from '../policies/catalog';
import type { ActionPolicy } from '../policies/types';
import { createStrategicPositionCatalogV0 } from './strategicPositionCatalog';
import {
  runStrategicPositionComparisonV0,
  strategicComparisonSeed,
  type StrategicComparisonVariantV0,
} from './strategicPositionComparison';

describe('strategic position comparison', () => {
  it('uses one common deterministic random seed across variants', async () => {
    const observedSeeds = new Map<string, string[]>();
    const variants = [
      recordingVariant('first', 0, observedSeeds),
      recordingVariant('last', -1, observedSeeds),
    ];
    const position = requiredPosition('minimum-winning-coalition');

    const run = await runStrategicPositionComparisonV0({
      positions: [position],
      variants,
      repetitions: 2,
      now: () => 100,
    });

    expect(observedSeeds.get('first')).toEqual([
      strategicComparisonSeed(position.id, 0),
      strategicComparisonSeed(position.id, 1),
    ]);
    expect(observedSeeds.get('last')).toEqual(observedSeeds.get('first'));
    expect(run.positions[0].repetitions).toHaveLength(2);
    expect(
      run.positions[0].repetitions.flatMap((entry) =>
        entry.decisions.map((decision) => decision.latencyMs)
      )
    ).toEqual([0, 0, 0, 0]);
  });

  it('records legal selections, focus signals, and preference status', async () => {
    const position = requiredPosition('minimum-winning-coalition');
    const preferredKey = position.focusActions.find(
      (focus) => focus.id === 'pivotal'
    )?.actionKey;
    if (!preferredKey) {
      throw new Error('Missing pivotal focus action.');
    }
    const preferredPolicy: ActionPolicy = {
      selectAction(context) {
        return context.legalActions.find(
          (action) => actionStableKey(action) === preferredKey
        );
      },
    };
    const variant: StrategicComparisonVariantV0 = {
      descriptor: {
        kind: 'custom',
        id: 'preferred-test',
        label: 'Preferred test policy',
        implementationId: 'test:preferred-v1',
      },
      policy: preferredPolicy,
      inspectHeuristicV2: true,
    };

    const run = await runStrategicPositionComparisonV0({
      positions: [position],
      variants: [variant],
      now: () => 0,
    });
    const decision = run.positions[0].repetitions[0].decisions[0];

    expect(decision).toMatchObject({
      variantId: 'preferred-test',
      selectedActionKey: preferredKey,
      selectedFocusActionId: 'pivotal',
      matchesExpectedPreference: true,
      searchDiagnostics: null,
    });
    expect(decision.focusSignals).toHaveLength(position.focusActions.length);
    expect(run.positions[0].stateSummary.perspectivePlayerId).toBe('PlayerA');
    expect(run.positions[0].positionFingerprint).toMatch(
      /^sha256:[0-9a-f]{64}$/
    );
    expect(
      decision.focusSignals.every(
        (signal) =>
          signal.heuristicScore !== null && signal.heuristicRank !== null
      )
    ).toBe(true);
  });

  it('assesses only the declared pairwise preference', async () => {
    const position = requiredPosition('minimum-winning-coalition');
    const overKey = position.focusActions.find(
      (focus) => focus.id === 'fortress'
    )?.actionKey;
    const focusKeys = new Set(
      position.focusActions.map((focus) => focus.actionKey)
    );
    const thirdKey = legalActionKeys(position).find(
      (actionKey) => !focusKeys.has(actionKey)
    );
    if (!overKey || !thirdKey) {
      throw new Error('Missing declared or third comparison action.');
    }

    const run = await runStrategicPositionComparisonV0({
      positions: [position],
      variants: [
        selectingVariant('declared-other', overKey),
        selectingVariant('unassessed-third', thirdKey),
      ],
      now: () => 0,
    });

    expect(
      run.positions[0].repetitions[0].decisions.map(
        (decision) => decision.matchesExpectedPreference
      )
    ).toEqual([false, null]);
  });

  it('is repeatable after excluding intentionally measured latency', async () => {
    const positions = [requiredPosition('tie-denial-restores-match')];
    const makeVariants = () => [
      recordingVariant('first', 0, new Map()),
      recordingVariant('last', -1, new Map()),
    ];
    const first = await runStrategicPositionComparisonV0({
      positions,
      variants: makeVariants(),
      now: () => 0,
    });
    const second = await runStrategicPositionComparisonV0({
      positions,
      variants: makeVariants(),
      now: () => 0,
    });

    expect(second).toEqual(first);
  });

  it('uses matched random scenarios for declared position pairs', async () => {
    const positions = createStrategicPositionCatalogV0().filter(
      (position) => position.pairId === 'reshuffle-boundary'
    );
    const run = await runStrategicPositionComparisonV0({
      positions,
      variants: [recordingVariant('first', 0, new Map())],
      now: () => 0,
    });

    expect(run.positions).toHaveLength(2);
    expect(run.positions[0].randomGroupId).toBe('reshuffle-boundary');
    expect(run.positions[1].randomGroupId).toBe('reshuffle-boundary');
    expect(run.positions[0].repetitions[0].sharedRandomSeed).toBe(
      run.positions[1].repetitions[0].sharedRandomSeed
    );
  });

  it('constructs declared built-ins instead of accepting mislabeled policies', async () => {
    const position = requiredPosition('minimum-winning-coalition');
    await expect(
      runStrategicPositionComparisonV0({
        positions: [position],
        variants: [
          {
            descriptor: {
              kind: 'heuristic-v2-direct',
              id: 'mislabeled-direct',
              label: 'Mislabeled direct',
            },
            policy: firstLegalPolicy,
          },
        ],
      })
    ).rejects.toThrow('cannot override its declared policy');

    const spec = structuredClone(getBotProfile('rollout-search-v2-easy').spec);
    await expect(
      runStrategicPositionComparisonV0({
        positions: [position],
        variants: [
          {
            descriptor: {
              kind: 'bot-spec',
              id: 'wrong-spec-id',
              label: 'Wrong spec id',
              spec,
            },
          },
        ],
      })
    ).rejects.toThrow('does not match spec id');
  });

  it('isolates state and legal-action inputs between variants', async () => {
    const position = requiredPosition('minimum-winning-coalition');
    const originalMoons = position.state.players[0].resources.Moons;
    const originalLegalCount = legalActionKeys(position).length;
    let observedMoons: number | undefined;
    let observedLegalCount: number | undefined;

    const run = await runStrategicPositionComparisonV0({
      positions: [position],
      variants: [
        customVariant('mutator', {
          selectAction(context) {
            const selected = context.legalActions[0];
            context.state.players[0].resources.Moons = 999;
            (context.legalActions as GameAction[]).splice(0);
            return selected;
          },
        }),
        customVariant('observer', {
          selectAction(context) {
            observedMoons = context.state.players[0].resources.Moons;
            observedLegalCount = context.legalActions.length;
            return context.legalActions[0];
          },
        }),
      ],
      now: () => 0,
    });

    expect(run.positions[0].repetitions[0].decisions).toHaveLength(2);
    expect(observedMoons).toBe(originalMoons);
    expect(observedLegalCount).toBe(originalLegalCount);
    expect(position.state.players[0].resources.Moons).toBe(originalMoons);
  });
});

const firstLegalPolicy: ActionPolicy = {
  selectAction(context) {
    return context.legalActions[0];
  },
};

function recordingVariant(
  id: string,
  actionIndex: number,
  observedSeeds: Map<string, string[]>
): StrategicComparisonVariantV0 {
  return {
    descriptor: {
      kind: 'custom',
      id,
      label: id,
      implementationId: `test:recording:${id}:v1`,
    },
    policy: {
      selectAction(context) {
        observedSeeds.set(id, [
          ...(observedSeeds.get(id) ?? []),
          context.randomSeed ?? '',
        ]);
        const index =
          actionIndex < 0 ? context.legalActions.length - 1 : actionIndex;
        return context.legalActions[index];
      },
    },
  };
}

function selectingVariant(
  id: string,
  selectedActionKey: string
): StrategicComparisonVariantV0 {
  return {
    descriptor: {
      kind: 'custom',
      id,
      label: id,
      implementationId: `test:selecting:${id}:v1`,
    },
    policy: {
      selectAction(context) {
        return context.legalActions.find(
          (action) => actionStableKey(action) === selectedActionKey
        );
      },
    },
  };
}

function customVariant(
  id: string,
  policy: ActionPolicy
): StrategicComparisonVariantV0 {
  return {
    descriptor: {
      kind: 'custom',
      id,
      label: id,
      implementationId: `test:custom:${id}:v1`,
    },
    policy,
  };
}

function legalActionKeys(position: ReturnType<typeof requiredPosition>) {
  return legalActionsForDecisionPlayer(
    position.state,
    position.perspectivePlayerId
  ).map(actionStableKey);
}

function requiredPosition(id: string) {
  const position = createStrategicPositionCatalogV0().find(
    (candidate) => candidate.id === id
  );
  if (!position) {
    throw new Error(`Missing strategic position ${id}.`);
  }
  return position;
}
