import { afterEach, describe, expect, it } from 'vitest';

import {
  PLAYER_A,
  makeGameState,
  makePlayer,
  withDeed,
} from '../../engine/__tests__/fixtures';
import { clearAllDeedTokenLayouts } from '../components/deedTokenLayout';
import { elementCenter, type AnimationDomTargets } from './domTargets';
import {
  buildIncomeFlightsFromDom,
  buildTaxLossFlightsFromDom,
  collectCardPlayFlights,
  collectDeedResourceFlights,
  collectIncomeChoiceResourceFlights,
  collectTerminalCleanupFlights,
} from './flightPlans';
import {
  RESOURCE_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from './timing';

describe('flightPlans', () => {
  afterEach(() => {
    clearAllDeedTokenLayouts();
  });

  it('builds staggered tax and income flights from resolved DOM targets', () => {
    const resource = makeElement({ left: 10, top: 20, width: 20, height: 20 });
    const districtCard = makeElement({
      left: 100,
      top: 200,
      width: 80,
      height: 120,
    });
    const crown = makeElement({ left: 300, top: 40, width: 20, height: 20 });
    const targets = makeTargets({
      resourceToken: () => resource,
      districtCard: () => districtCard,
      crownToken: () => crown,
    });

    expect(
      buildTaxLossFlightsFromDom(
        [
          { playerId: 'PlayerA', suit: 'Moons' },
          { playerId: 'PlayerA', suit: 'Suns' },
        ],
        makeIds('tax'),
        targets
      )
    ).toMatchObject([
      {
        id: 'tax-1',
        suit: 'Moons',
        startX: 20,
        startY: 30,
        endX: 20,
        endY: 500,
        delayMs: 0,
        variant: 'tax-loss',
      },
      {
        id: 'tax-2',
        suit: 'Suns',
        delayMs: TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
      },
    ]);

    expect(
      buildIncomeFlightsFromDom(
        [
          {
            playerId: 'PlayerA',
            suit: 'Moons',
            source: {
              kind: 'district-card',
              cardId: '6',
              districtId: 'D1',
            },
          },
          {
            playerId: 'PlayerA',
            suit: 'Suns',
            source: { kind: 'crown', cardId: '30' },
          },
          {
            playerId: 'PlayerA',
            suit: 'Knots',
            source: {
              kind: 'income-choice',
              cardId: '8',
              districtId: 'D2',
            },
          },
        ],
        makeIds('income'),
        targets
      )
    ).toMatchObject([
      {
        id: 'income-1',
        startX: 140,
        startY: 260,
        endX: 20,
        endY: 30,
        delayMs: 0,
      },
      {
        id: 'income-2',
        startX: 310,
        startY: 50,
        delayMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
      },
      {
        id: 'income-3',
        startX: 140,
        startY: 260,
        delayMs: 2 * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
      },
    ]);
  });

  it('plans deed transfers against the remembered token-side rail', () => {
    const state = withDeed(makeGameState(), 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });
    const source = makeElement({ width: 20, height: 20 });
    const rail = makeElement({ left: 100, top: 100, width: 20, height: 80 });
    const targets = makeTargets({
      developingCard: () => makeElement(),
      resourceTokenForDeedTransfer: () => source,
      deedTokenRail: (_card, side) => (side === 'left' ? rail : null),
    });

    expect(
      collectDeedResourceFlights(
        state,
        {
          type: 'develop-deed',
          districtId: 'D1',
          cardId: '6',
          tokens: { Moons: 1 },
        },
        PLAYER_A,
        makeIds('deed'),
        targets
      )
    ).toEqual([
      {
        id: 'deed-1',
        suit: 'Moons',
        startX: 10,
        startY: 10,
        endX: 110,
        endY: 140,
        delayMs: 0 * RESOURCE_FLIGHT_STAGGER_MS,
        variant: 'transfer',
      },
    ]);
  });

  it('does not plan terminal cleanup flights for retained deeds', () => {
    const previous = withDeed(makeGameState(), 'D1', PLAYER_A, {
      cardId: '6',
      progress: 2,
      tokens: { Moons: 2 },
    });
    const next = { ...previous, phase: 'GameOver' as const };

    expect(previous.phase).not.toBe('GameOver');
    expect(next.phase).toBe('GameOver');
    expect(collectTerminalCleanupFlights()).toBeNull();
  });

  it('plans income choice reveal, sold-card, lane-play, and draw flights', () => {
    const hand = makeElement({
      left: 10,
      top: 20,
      width: 80,
      height: 120,
      attributes: { 'data-hand-slot-kind': 'occupied' },
    });
    const target = makeElement({
      left: 200,
      top: 300,
      width: 90,
      height: 130,
    });
    const lane = makeElement({ classNames: ['is-bot'] });
    const targets = makeTargets({
      districtCard: () => hand,
      resourceToken: () => target,
      handSource: () => hand,
      handDrawTarget: () => target,
      discardTarget: () => target,
      deckSource: () => hand,
      lane: () => lane,
      districtColumn: () => target,
      laneTargetCenter: () => ({ x: 400, y: 500 }),
      laneCardSize: () => ({ width: 96, height: 140 }),
    });

    expect(
      collectIncomeChoiceResourceFlights(
        makeGameState({
          phase: 'CollectIncome',
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D1',
              cardId: '6',
              suits: ['Moons', 'Knots'],
            },
          ],
        }),
        makeGameState({ phase: 'ActionWindow' }),
        {
          type: 'choose-income-suit',
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
        },
        makeIds('choice'),
        targets
      )
    ).toMatchObject([{ id: 'choice-1', startX: 50, endX: 245 }]);

    expect(
      collectIncomeChoiceResourceFlights(
        makeGameState({
          phase: 'CollectIncome',
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D1',
              cardId: '6',
              suits: ['Moons', 'Knots'],
            },
            {
              playerId: 'PlayerB',
              districtId: 'D2',
              cardId: '8',
              suits: ['Waves', 'Leaves'],
            },
          ],
        }),
        makeGameState({
          phase: 'CollectIncome',
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D1',
              cardId: '6',
              suits: ['Moons', 'Knots'],
            },
            {
              playerId: 'PlayerB',
              districtId: 'D2',
              cardId: '8',
              suits: ['Waves', 'Leaves'],
            },
          ],
        }),
        {
          type: 'choose-income-suit',
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
        },
        makeIds('choice-pending'),
        targets
      )
    ).toEqual([]);

    expect(
      collectIncomeChoiceResourceFlights(
        makeGameState({
          phase: 'CollectIncome',
          pendingIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D1',
              cardId: '6',
              suits: ['Moons', 'Knots'],
            },
            {
              playerId: 'PlayerB',
              districtId: 'D2',
              cardId: '8',
              suits: ['Waves', 'Leaves'],
            },
          ],
          submittedIncomeChoices: [
            {
              playerId: PLAYER_A,
              districtId: 'D1',
              cardId: '6',
              suit: 'Knots',
            },
          ],
        }),
        makeGameState({ phase: 'ActionWindow' }),
        {
          type: 'choose-income-suit',
          playerId: 'PlayerB',
          districtId: 'D2',
          cardId: '8',
          suit: 'Leaves',
        },
        makeIds('choice-reveal'),
        targets
      )
    ).toMatchObject([
      {
        id: 'choice-reveal-1',
        suit: 'Knots',
        delayMs: 0,
      },
      {
        id: 'choice-reveal-2',
        suit: 'Leaves',
        delayMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
      },
    ]);

    expect(
      collectCardPlayFlights(
        makeGameState(),
        makeGameState(),
        { type: 'sell-card', cardId: '6' },
        PLAYER_A,
        makeIds('sell'),
        targets
      )
    ).toMatchObject([{ id: 'sell-1', visual: 'face', cardId: '6' }]);

    expect(
      collectCardPlayFlights(
        makeGameState(),
        makeGameState(),
        { type: 'buy-deed', cardId: '6', districtId: 'D1' },
        PLAYER_A,
        makeIds('lane'),
        targets
      )
    ).toMatchObject([
      {
        id: 'lane-1',
        isDeed: true,
        perspective: 'bot',
        endX: 400,
        endY: 500,
        endWidth: 96,
        endHeight: 140,
      },
    ]);

    const previous = makeGameState({
      players: [makePlayer(PLAYER_A, { hand: ['6'] }), makePlayer('PlayerB')],
    });
    const next = makeGameState({
      players: [
        makePlayer(PLAYER_A, { hand: ['6', '7'] }),
        makePlayer('PlayerB'),
      ],
    });
    expect(
      collectCardPlayFlights(
        previous,
        next,
        { type: 'end-turn' },
        PLAYER_A,
        makeIds('draw'),
        targets
      )
    ).toMatchObject([{ id: 'draw-1', variant: 'draw', visual: 'back' }]);
  });
});

function makeTargets(
  overrides: Partial<AnimationDomTargets> = {}
): AnimationDomTargets {
  return {
    isAvailable: () => true,
    viewportCenterY: () => 500,
    elementCenter,
    tokenVisualCenter: elementCenter,
    resourceToken: () => null,
    resourceTokenForDeedTransfer: () => null,
    crownToken: () => null,
    districtCard: () => null,
    terminalDevelopingCard: () => null,
    developingCard: () => null,
    deedToken: () => null,
    deedTokenOnSide: () => null,
    deedTokenRail: () => null,
    computedStyle: () => ({ rowGap: '', gap: '' }) as CSSStyleDeclaration,
    lane: () => null,
    districtColumn: () => null,
    laneFrame: () => null,
    laneTargetCenter: () => null,
    laneCardSize: () => null,
    deckSource: () => null,
    discardTarget: () => null,
    handSource: () => null,
    handDrawTarget: () => null,
    ...overrides,
  };
}

function makeIds(prefix: string): () => string {
  let next = 0;
  return () => {
    next += 1;
    return `${prefix}-${next}`;
  };
}

function makeElement({
  left = 0,
  top = 0,
  width = 10,
  height = 10,
  classNames = [],
  attributes = {},
}: {
  left?: number;
  top?: number;
  width?: number;
  height?: number;
  classNames?: readonly string[];
  attributes?: Readonly<Record<string, string>>;
} = {}): HTMLElement {
  return {
    classList: {
      contains: (className: string) => classNames.includes(className),
    },
    getAttribute: (name: string) => attributes[name] ?? null,
    getBoundingClientRect: () => ({
      bottom: top + height,
      height,
      left,
      right: left + width,
      top,
      width,
      x: left,
      y: top,
      toJSON: () => ({}),
    }),
  } as unknown as HTMLElement;
}
