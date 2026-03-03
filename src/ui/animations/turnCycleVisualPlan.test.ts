import { describe, expect, it } from 'vitest';

import { makeGameState } from '../../engine/__tests__/fixtures';
import type { TurnCycleEvents } from '../turnCycleEvents';
import {
  buildTurnCycleVisualPlan,
  collectTurnCycleAnimationPlan,
} from './turnCycleVisualPlan';

describe('buildTurnCycleVisualPlan', () => {
  it('schedules a text-only income stage after an existing card-flight delay', () => {
    const plan = buildTurnCycleVisualPlan(makeCycle(), 280);

    expect(plan.visualPlan).toMatchObject({
      taxLabelAtMs: null,
      taxFlightLaunchAtMs: null,
      incomeLabelAtMs: 280,
      incomeLabelHideAtMs: 1780,
      incomeFlightLaunchAtMs: 1780,
      incomeHighlightStartAtMs: 280,
      incomeHighlightEndAtMs: 1780,
      hideAllAtMs: 2000,
      taxSuit: null,
      incomeRank: 7,
    });
    expect(plan.incomeAnimationEndMs).toBe(1780);
    expect(plan.totalDurationMs).toBe(2020);
  });

  it('keeps the tax label stage when tax triggers without resource losses', () => {
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
      0
    );

    expect(plan.visualPlan).toMatchObject({
      taxLabelAtMs: 0,
      taxLabelHideAtMs: 1000,
      taxPulseStartAtMs: null,
      taxPulseEndAtMs: null,
      taxFlightLaunchAtMs: null,
      taxResourcesApplyAtMs: null,
      taxPulseTargets: [],
      taxFlightTokens: [],
      incomeLabelAtMs: 1340,
      hideAllAtMs: 3060,
    });
    expect(plan.totalDurationMs).toBe(3080);
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
    const plan = buildTurnCycleVisualPlan(cycle, 280);

    expect(plan.visualPlan).toMatchObject({
      taxLabelAtMs: 280,
      taxLabelHideAtMs: 1280,
      taxPulseStartAtMs: 280,
      taxPulseEndAtMs: 1280,
      taxFlightLaunchAtMs: 1400,
      taxResourcesApplyAtMs: 3300,
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
      incomeLabelAtMs: 3520,
      incomeFlightLaunchAtMs: 5020,
      incomeHighlightStartAtMs: 3520,
      incomeHighlightEndAtMs: 5675,
      hideAllAtMs: 5895,
      highlightCardIds: ['29', '27'],
    });
    expect(plan.incomeAnimationEndMs).toBe(5675);
    expect(plan.totalDurationMs).toBe(5915);
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
      0
    );

    expect(plan.visualPlan.highlightCrowns).toEqual([
      { playerId: 'PlayerA', suit: 'Moons' },
      { playerId: 'PlayerB', suit: 'Suns' },
    ]);
    expect(plan.visualPlan.highlightCardIds).toEqual([]);
    expect(plan.visualPlan.incomeHighlightEndAtMs).toBe(
      plan.incomeAnimationEndMs
    );
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
        280
      )
    ).toBeNull();
    expect(
      collectTurnCycleAnimationPlan(previous, next, { type: 'end-turn' }, 280)
        ?.visualPlan
    ).toMatchObject({
      incomeLabelAtMs: 280,
      incomeRank: 7,
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
