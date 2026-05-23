import { describe, expect, it } from 'vitest';

import {
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from '../../engine/__tests__/fixtures';
import {
  applyTurnCycleResourcePreviewEvent,
  buildTurnCycleResourcePreviewPlan,
} from './turnCycleResourcePreview';
import type { TurnCycleAnimationPlan } from './turnCycleVisualPlan';

describe('buildTurnCycleResourcePreviewPlan', () => {
  it('schedules per-token tax losses before replacing the preview after income', () => {
    const previous = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Moons: 1 }) }),
      ] as const,
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 4 }) }),
        makePlayer(PLAYER_B, { resources: makeResources({ Moons: 2 }) }),
      ] as const,
    });
    const plan = buildTurnCycleResourcePreviewPlan(
      previous,
      next,
      makeTurnCyclePlan()
    );

    expect(plan.initialPreview.PlayerA?.Moons).toBe(3);
    expect(plan.events).toMatchObject([
      {
        kind: 'tax-loss',
        atMs: 1630,
        token: { playerId: PLAYER_A, suit: 'Moons' },
      },
      {
        kind: 'tax-loss',
        atMs: 2130,
        token: { playerId: PLAYER_A, suit: 'Moons' },
      },
      {
        kind: 'replace',
        atMs: 3200,
      },
    ]);

    const afterFirst = applyTurnCycleResourcePreviewEvent(
      plan.initialPreview,
      plan.events[0]
    );
    const afterSecond = applyTurnCycleResourcePreviewEvent(
      afterFirst,
      plan.events[1]
    );
    const afterIncome = applyTurnCycleResourcePreviewEvent(
      afterSecond,
      plan.events[2]
    );
    expect(afterFirst?.PlayerA?.Moons).toBe(2);
    expect(afterSecond?.PlayerA?.Moons).toBe(1);
    expect(afterIncome?.PlayerA?.Moons).toBe(4);
  });

  it('supports the summary-tax fallback event', () => {
    const preview = {
      [PLAYER_A]: makeResources({ Moons: 3 }),
      [PLAYER_B]: makeResources({ Moons: 2 }),
    };
    const next = applyTurnCycleResourcePreviewEvent(preview, {
      kind: 'tax-loss-summary',
      atMs: 1000,
      suit: 'Moons',
      lossesByPlayer: [
        { playerId: PLAYER_A, count: 2 },
        { playerId: PLAYER_B, count: 1 },
      ],
    });

    expect(next?.PlayerA?.Moons).toBe(1);
    expect(next?.PlayerB?.Moons).toBe(1);
  });
});

function makeTurnCyclePlan(): TurnCycleAnimationPlan {
  return {
    incomeAnimationEndMs: 3200,
    totalDurationMs: 3440,
    visualPlan: {
      taxPulseStartAtMs: 0,
      taxPulseEndAtMs: 1000,
      taxFlightLaunchAtMs: 1630,
      taxResourcesApplyAtMs: 3030,
      taxPulseTargets: [{ playerId: PLAYER_A, suit: 'Moons' }],
      taxFlightTokens: [
        { playerId: PLAYER_A, suit: 'Moons' },
        { playerId: PLAYER_A, suit: 'Moons' },
      ],
      taxLossesByPlayer: [{ playerId: PLAYER_A, count: 2 }],
      incomeFlightLaunchAtMs: 2200,
      incomeFlightTokens: [],
      incomeHighlightStartAtMs: 1600,
      incomeHighlightEndAtMs: 3200,
      hideAllAtMs: 3420,
      taxSuit: 'Moons',
      highlightCardIds: [],
      highlightCrowns: [],
    },
  };
}
