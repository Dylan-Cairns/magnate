import { describe, expect, it } from 'vitest';

import { makeGameState } from '../../engine/__tests__/fixtures';
import type { TurnCycleEvents } from '../turnCycleEvents';
import {
  buildTurnCycleVisualPlan,
  collectTurnCycleAnimationPlan,
} from './turnCycleVisualPlan';

describe('buildTurnCycleVisualPlan', () => {
  it('schedules income flights after d10 dice settle (no tax)', () => {
    const plan = buildTurnCycleVisualPlan(makeCycle());

    expect(plan.visualPlan).toMatchObject({
      taxFlightLaunchAtMs: null,
      // cursor starts at DICE_D10_SETTLE_MS (1000); income flights launch 400ms later
      incomeFlightLaunchAtMs: 1400,
      incomeHighlightStartAtMs: 1400,
      incomeHighlightEndAtMs: 1400,
      hideAllAtMs: 1620,
      taxSuit: null,
    });
    expect(plan.incomeAnimationEndMs).toBe(1400);
    expect(plan.totalDurationMs).toBe(1640);
  });

  it('keeps the tax stage when tax triggers without resource losses', () => {
    const plan = buildTurnCycleVisualPlan(
      makeCycle({
        tax: {
          suit: 'Moons',
          lossesByPlayer: [
            { playerId: 'PlayerA', count: 0 },
            { playerId: 'PlayerB', count: 0 },
          ],
        },
      }),
    );

    expect(plan.visualPlan).toMatchObject({
      taxPulseStartAtMs: null,
      taxPulseEndAtMs: null,
      taxFlightLaunchAtMs: null,
      taxResourcesApplyAtMs: null,
      taxPulseTargets: [],
      taxFlightTokens: [],
      // cursor starts at DICE_TAX_SETTLE_MS (2000); income flights at 2570 + 400 = 2970
      hideAllAtMs: 3190,
    });
    expect(plan.totalDurationMs).toBe(3210);
  });

  it('orders tax losses before income flights and preserves visual targets', () => {
    const cycle = makeCycle({
      tax: {
        suit: 'Moons',
        lossesByPlayer: [
          { playerId: 'PlayerA', count: 2 },
          { playerId: 'PlayerB', count: 1 },
        ],
      },
      incomeTokens: [
        {
          playerId: 'PlayerA',
          suit: 'Suns',
          source: {
            kind: 'district-card',
            districtId: 'D1',
            cardId: '29',
          },
        },
        {
          playerId: 'PlayerB',
          suit: 'Waves',
          source: {
            kind: 'district-card',
            districtId: 'D3',
            cardId: '27',
          },
        },
      ],
      incomeHighlights: [
        { playerId: 'PlayerA', districtId: 'D1', cardId: '29' },
        { playerId: 'PlayerB', districtId: 'D3', cardId: '27' },
      ],
    });
    const plan = buildTurnCycleVisualPlan(cycle);

    expect(plan.visualPlan).toMatchObject({
      // cursor = DICE_TAX_SETTLE_MS (2000); pulse starts there
      taxPulseStartAtMs: 2000,
      // flights start at 2000 + 350 = 2350
      taxFlightLaunchAtMs: 2350,
      // 3 tokens, stagger 500, duration 900: 2350 + 2*500 + 900 = 4250
      taxResourcesApplyAtMs: 4250,
      taxPulseEndAtMs: 4250,
      taxPulseTargets: [
        { playerId: 'PlayerA', suit: 'Moons' },
        { playerId: 'PlayerB', suit: 'Moons' },
      ],
      taxFlightTokens: [
        { playerId: 'PlayerA', suit: 'Moons' },
        { playerId: 'PlayerA', suit: 'Moons' },
        { playerId: 'PlayerB', suit: 'Moons' },
      ],
      taxLossesByPlayer: [
        { playerId: 'PlayerA', count: 2 },
        { playerId: 'PlayerB', count: 1 },
      ],
      // income cursor = 4250 + 220 = 4470; flights at 4470 + 400 = 4870
      incomeFlightLaunchAtMs: 4870,
      incomeHighlightStartAtMs: 4870,
      // 2 tokens, stagger 95, duration 560: 4870 + 95 + 560 = 5525;
      // highlighted income sources stay visible for at least 400 + 560ms
      incomeHighlightEndAtMs: 5830,
      hideAllAtMs: 6050,
      highlightCardIds: ['29', '27'],
    });
    expect(plan.incomeAnimationEndMs).toBe(5525);
    expect(plan.totalDurationMs).toBe(6070);
  });

  it('deduplicates crown highlights by player and suit', () => {
    const plan = buildTurnCycleVisualPlan(
      makeCycle({
        incomeRank: 10,
        incomeTokens: [
          {
            playerId: 'PlayerA',
            suit: 'Moons',
            source: { kind: 'crown', cardId: '32' },
          },
          {
            playerId: 'PlayerA',
            suit: 'Moons',
            source: { kind: 'crown', cardId: '32' },
          },
          {
            playerId: 'PlayerB',
            suit: 'Suns',
            source: { kind: 'crown', cardId: '33' },
          },
        ],
      }),
    );

    expect(plan.visualPlan.highlightCrowns).toEqual([
      { playerId: 'PlayerA', suit: 'Moons' },
      { playerId: 'PlayerB', suit: 'Suns' },
    ]);
    expect(plan.visualPlan.highlightCardIds).toEqual([]);
    expect(plan.incomeAnimationEndMs).toBe(2150);
    expect(plan.visualPlan.incomeHighlightStartAtMs).toBe(1400);
    expect(plan.visualPlan.incomeHighlightEndAtMs).toBe(2360);
  });
});

describe('collectTurnCycleAnimationPlan', () => {
  it('derives a turn-cycle plan only for an end-turn income roll', () => {
    const previous = makeGameState();
    const next = makeGameState({ lastIncomeRoll: { die1: 7, die2: 4 } });

    expect(
      collectTurnCycleAnimationPlan(
        previous,
        next,
        { type: 'trade', give: 'Moons', receive: 'Suns' },
      )
    ).toBeNull();
    expect(
      collectTurnCycleAnimationPlan(previous, next, { type: 'end-turn' })
        ?.visualPlan
    ).toMatchObject({
      // no tax: cursor starts at DICE_D10_SETTLE_MS (1000), flights at 1400
      incomeFlightLaunchAtMs: 1400,
    });
  });
});

function makeCycle(overrides: Partial<TurnCycleEvents> = {}): TurnCycleEvents {
  return {
    cycleOwner: 'PlayerB',
    roll: { die1: 7, die2: 4 },
    incomeRank: 7,
    tax: null,
    incomeTokens: [],
    incomeHighlights: [],
    pendingChoices: [],
    ...overrides,
  };
}
