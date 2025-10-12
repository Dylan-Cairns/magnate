import { describe, expect, it } from 'vitest';

import { legalActions } from './actionBuilders';
import {
  PLAYER_A,
  PLAYER_B,
  makeDefaultDistricts,
  makeDistrict,
  makeGameState,
  makePlayer,
  makeResources,
  withDeed,
} from './__tests__/fixtures';

describe('legalActions', () => {
  it('returns no actions in non-action phases', () => {
    const state = makeGameState({ phase: 'StartTurn' });
    expect(legalActions(state)).toEqual([]);
  });

  it('CollectIncome exposes choose-income-suit actions when pending choices exist', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
      ],
    });

    const actions = legalActions(state);
    expect(actions).toEqual([
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '7',
        suit: 'Suns',
      },
      {
        type: 'choose-income-suit',
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '7',
        suit: 'Wyrms',
      },
    ]);
  });

  it('CollectIncome emits no choice actions for non-owning active player', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      activePlayerIndex: 1,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
      ],
    });

    expect(legalActions(state)).toEqual([]);
  });

  it('ActionWindow only includes trade give suits with at least 3 tokens', () => {
    const players = [
      makePlayer(PLAYER_A, {
        resources: makeResources({ Moons: 3, Suns: 2, Knots: 6 }),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'ActionWindow', players });
    const trades = legalActions(state).filter((action) => action.type === 'trade');

    expect(trades.length).toBeGreaterThan(0);
    expect(trades.every((trade) => trade.give === 'Moons' || trade.give === 'Knots')).toBe(
      true
    );
  });

  it('ActionWindow never includes trades with same give/receive suit', () => {
    const players = [
      makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'ActionWindow', players });
    const trades = legalActions(state).filter((action) => action.type === 'trade');
    expect(trades.every((trade) => trade.give !== trade.receive)).toBe(true);
  });

  it('ActionWindow includes trade actions when at least one trade is available', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'trade')).toBe(true);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(false);
  });

  it('ActionWindow exposes card-play actions before card play even without trades', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: false,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(true);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(false);
  });

  it('ActionWindow includes end-turn once a card has been played this turn', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(true);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(false);
    expect(actions.some((action) => action.type === 'buy-deed')).toBe(false);
    expect(actions.some((action) => action.type === 'develop-outright')).toBe(false);
  });

  it('ActionWindow emits one-token deed actions only for affordable deed suits', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 1, Knots: 0, Suns: 5 }),
        }),
        makePlayer(PLAYER_B),
      ],
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const developActions = legalActions(state).filter(
      (action): action is Extract<(typeof action), { type: 'develop-deed' }> =>
        action.type === 'develop-deed'
    );

    expect(developActions).toHaveLength(1);
    expect(developActions[0].tokens).toEqual({ Moons: 1 });
  });

  it('ActionWindow includes develop actions when a deed is developable', () => {
    let state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 1 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'develop-deed')).toBe(true);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(true);
  });

  it('ActionWindow includes both trade and develop actions in the same state', () => {
    let state = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    state = withDeed(state, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const actions = legalActions({ ...state, phase: 'ActionWindow' });
    expect(actions.some((action) => action.type === 'trade')).toBe(true);
    expect(actions.some((action) => action.type === 'develop-deed')).toBe(true);
  });

  it('ActionWindow includes a sell action for each property card in hand', () => {
    const players = [
      makePlayer(PLAYER_A, {
        hand: ['6', '8', '13'],
        resources: makeResources(),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'ActionWindow', players });
    const sells = legalActions(state).filter((action) => action.type === 'sell-card');
    expect(sells.map((action) => action.cardId).sort()).toEqual(['13', '6', '8']);
  });

  it('ActionWindow excludes non-property cards present in hand', () => {
    const players = [
      makePlayer(PLAYER_A, {
        hand: ['6', '30', '36', '37'],
        resources: makeResources(),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'ActionWindow', players });
    const actions = legalActions(state);

    const nonPropertyIds = new Set(['30', '36', '37']);
    const actionsForNonProperty = actions.filter((action) => {
      switch (action.type) {
        case 'sell-card':
        case 'buy-deed':
        case 'develop-outright':
        case 'develop-deed':
          return nonPropertyIds.has(action.cardId);
        case 'trade':
        case 'choose-income-suit':
        case 'end-turn':
          return false;
      }
    });
    expect(actionsForNonProperty).toHaveLength(0);
  });

  it('ActionWindow excludes card-play actions after a card has already been played this turn', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(false);
    expect(actions.some((action) => action.type === 'buy-deed')).toBe(false);
    expect(actions.some((action) => action.type === 'develop-outright')).toBe(false);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(true);
  });

  it('ActionWindow buy-deed actions only appear when placement is legal and cost is affordable', () => {
    const blockedDistricts = makeDefaultDistricts().map((district) => ({
      ...district,
      stacks: {
        ...district.stacks,
        [PLAYER_A]: {
          ...district.stacks[PLAYER_A],
          deed: {
            cardId: '7',
            progress: 0,
            tokens: {},
          },
        },
      },
    }));

    const players = [
      makePlayer(PLAYER_A, {
        hand: ['6'],
        resources: makeResources({ Moons: 0, Knots: 0 }),
      }),
      makePlayer(PLAYER_B),
    ] as const;

    const blockedState = makeGameState({
      phase: 'ActionWindow',
      players,
      districts: blockedDistricts,
    });
    expect(
      legalActions(blockedState).some((action) => action.type === 'buy-deed')
    ).toBe(false);

    const affordableState = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
      districts: makeDefaultDistricts(),
    });
    expect(
      legalActions(affordableState).some((action) => action.type === 'buy-deed')
    ).toBe(true);
  });

  it('ActionWindow blocks first-placement buy/develop actions in districts without pawn-suit overlap', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['7'],
          resources: makeResources({ Suns: 1, Wyrms: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
      districts: [
        makeDistrict('D1', ['Moons']),
        makeDistrict('D2', ['Suns']),
        makeDistrict('D3', ['Waves']),
        makeDistrict('D4', ['Leaves']),
        makeDistrict('D5', []),
      ],
    });

    const actions = legalActions(state);
    const districtActions = actions.filter(
      (
        action
      ): action is
        | Extract<typeof action, { type: 'buy-deed' }>
        | Extract<typeof action, { type: 'develop-outright' }> =>
        (action.type === 'buy-deed' || action.type === 'develop-outright') &&
        action.cardId === '7'
    );

    const districtIds = new Set(districtActions.map((action) => action.districtId));
    expect(districtIds.has('D1')).toBe(false);
    expect(districtIds.has('D2')).toBe(true);
    expect(districtIds.has('D3')).toBe(false);
    expect(districtIds.has('D4')).toBe(false);
    expect(districtIds.has('D5')).toBe(true);
  });
});
