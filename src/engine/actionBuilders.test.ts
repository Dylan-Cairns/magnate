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

  it('OptionalTrade always includes an explicit end action', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'end-optional-trade')).toBe(true);
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

  it('OptionalDevelop always includes an explicit end action', () => {
    const state = makeGameState({
      phase: 'OptionalDevelop',
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const actions = legalActions(state);
    expect(actions.some((action) => action.type === 'end-optional-develop')).toBe(true);
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
        case 'end-optional-trade':
        case 'end-optional-develop':
          return false;
      }
    });
    expect(actionsForNonProperty).toHaveLength(0);
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
