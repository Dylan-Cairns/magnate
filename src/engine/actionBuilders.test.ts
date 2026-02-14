import { describe, expect, it } from 'vitest';

import { legalActions } from './actionBuilders';
import {
  PLAYER_A,
  PLAYER_B,
  makeDefaultDistricts,
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

  it('OptionalTrade only includes suits with at least 3 tokens', () => {
    const players = [
      makePlayer(PLAYER_A, {
        resources: makeResources({ Moons: 3, Suns: 2, Knots: 6 }),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'OptionalTrade', players });
    const trades = legalActions(state).filter((action) => action.type === 'trade');

    expect(trades.length).toBeGreaterThan(0);
    expect(trades.every((trade) => trade.give === 'Moons' || trade.give === 'Knots')).toBe(
      true
    );
  });

  it('OptionalTrade never includes trades with same give/receive suit', () => {
    const players = [
      makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'OptionalTrade', players });
    const trades = legalActions(state).filter((action) => action.type === 'trade');
    expect(trades.every((trade) => trade.give !== trade.receive)).toBe(true);
  });

  it('OptionalTrade includes trade actions when at least one trade is available', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'trade')).toBe(true);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(false);
  });

  it('OptionalTrade still exposes card-play actions before card play even without trades', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      cardPlayedThisTurn: false,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(true);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(false);
  });

  it('OptionalTrade includes end-turn once a card has been played this turn', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(true);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(false);
    expect(actions.some((action) => action.type === 'buy-deed')).toBe(false);
    expect(actions.some((action) => action.type === 'develop-outright')).toBe(false);
  });

  it('OptionalDevelop emits one-token deed actions only for affordable deed suits', () => {
    let state = makeGameState({
      phase: 'OptionalDevelop',
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

  it('OptionalDevelop includes develop actions when a deed is developable', () => {
    let state = makeGameState({
      phase: 'OptionalDevelop',
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

  it('OptionalDevelop still exposes card-play actions before card play even without deeds', () => {
    const state = makeGameState({
      phase: 'OptionalDevelop',
      cardPlayedThisTurn: false,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(true);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(false);
  });

  it('OptionalDevelop includes end-turn once a card has been played this turn', () => {
    const state = makeGameState({
      phase: 'OptionalDevelop',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'end-turn')).toBe(true);
    expect(actions.some((action) => action.type === 'sell-card')).toBe(false);
    expect(actions.some((action) => action.type === 'buy-deed')).toBe(false);
    expect(actions.some((action) => action.type === 'develop-outright')).toBe(false);
  });

  it('OptionalTrade and OptionalDevelop expose the same action kinds for the same state', () => {
    let base = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          resources: makeResources({ Moons: 3, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    base = withDeed(base, 'D1', PLAYER_A, {
      cardId: '6',
      progress: 0,
      tokens: {},
    });

    const tradeKinds = new Set(
      legalActions({ ...base, phase: 'OptionalTrade' }).map((action) => action.type)
    );
    const developKinds = new Set(
      legalActions({ ...base, phase: 'OptionalDevelop' }).map((action) => action.type)
    );

    expect([...tradeKinds].sort()).toEqual([...developKinds].sort());
  });

  it('PlayCard includes a sell action for each property card in hand', () => {
    const players = [
      makePlayer(PLAYER_A, {
        hand: ['6', '8', '13'],
        resources: makeResources(),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'PlayCard', players });
    const sells = legalActions(state).filter((action) => action.type === 'sell-card');
    expect(sells.map((action) => action.cardId).sort()).toEqual(['13', '6', '8']);
  });

  it('PlayCard excludes non-property cards present in hand', () => {
    const players = [
      makePlayer(PLAYER_A, {
        hand: ['6', '30', '36', '37'],
        resources: makeResources(),
      }),
      makePlayer(PLAYER_B),
    ] as const;
    const state = makeGameState({ phase: 'PlayCard', players });
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

  it('PlayCard emits no actions after a card has already been played this turn', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    expect(legalActions(state)).toEqual([]);
  });

  it('PlayCard buy-deed actions only appear when placement is legal and cost is affordable', () => {
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
      phase: 'PlayCard',
      players,
      districts: blockedDistricts,
    });
    expect(
      legalActions(blockedState).some((action) => action.type === 'buy-deed')
    ).toBe(false);

    const affordableState = makeGameState({
      phase: 'PlayCard',
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
});
