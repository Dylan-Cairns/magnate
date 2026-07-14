import { afterEach, describe, expect, it } from 'vitest';

import { PLAYER_A } from '../../engine/__tests__/fixtures';
import { clearAllDeedTokenLayouts } from '../components/deedTokenLayout';
import { elementCenter, type AnimationDomTargets } from './domTargets';
import {
  buildCardToDistrictFlightFromDom,
  buildDeedResourceFlightsFromDom,
  buildDrawCardFlightFromDom,
  buildIncomeFlightsFromDom,
  buildPaymentFlightsFromDom,
  buildSoldCardFlightFromDom,
  buildTaxLossFlightsFromDom,
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
    const source = makeElement({ width: 20, height: 20 });
    const rail = makeElement({ left: 100, top: 100, width: 20, height: 80 });
    const targets = makeTargets({
      developingCard: () => makeElement(),
      resourceTokenForDeedTransfer: () => source,
      deedTokenRail: (_card, side) => (side === 'left' ? rail : null),
    });

    expect(
      buildDeedResourceFlightsFromDom(
        [
          {
            type: 'deed-token-paid',
            playerId: PLAYER_A,
            districtId: 'D1',
            cardId: '6',
            suit: 'Moons',
            tokenIndex: 0,
            previousTokens: {},
            nextTokens: { Moons: 1 },
          },
        ],
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

  it('plans payment flights as fading removals from the resource area', () => {
    const moons = makeElement({ left: 10, top: 20, width: 20, height: 20 });
    const knots = makeElement({ left: 50, top: 70, width: 20, height: 20 });
    const targets = makeTargets({
      resourceToken: (_playerId, suit) =>
        suit === 'Moons' ? moons : suit === 'Knots' ? knots : null,
    });

    expect(
      buildPaymentFlightsFromDom(
        {
          type: 'resource-payment-started',
          reason: 'buy-deed',
          playerId: PLAYER_A,
          cardId: '6',
          districtId: 'D1',
          payment: { Moons: 1, Knots: 2 },
        },
        makeIds('payment'),
        targets
      )
    ).toMatchObject([
      {
        id: 'payment-1',
        suit: 'Moons',
        startX: 20,
        startY: 30,
        endX: 20,
        endY: 500,
        delayMs: 0,
        variant: 'payment',
      },
      {
        id: 'payment-2',
        suit: 'Knots',
        delayMs: RESOURCE_FLIGHT_STAGGER_MS,
        variant: 'payment',
      },
      {
        id: 'payment-3',
        suit: 'Knots',
        startX: 60,
        startY: 80,
        delayMs: 2 * RESOURCE_FLIGHT_STAGGER_MS,
        variant: 'payment',
      },
    ]);
  });

  it('plans sold-card, lane-play, and draw flights', () => {
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
      buildSoldCardFlightFromDom(
        PLAYER_A,
        '6',
        makeIds('sell'),
        targets
      )
    ).toMatchObject([{ id: 'sell-1', visual: 'face', cardId: '6' }]);

    expect(
      buildCardToDistrictFlightFromDom(
        {
          type: 'card-played-to-district',
          playerId: PLAYER_A,
          cardId: '6',
          districtId: 'D1',
          placement: 'deed',
        },
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

    expect(
      buildDrawCardFlightFromDom(
        PLAYER_A,
        '7',
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
