import { describe, expect, it } from 'vitest';

import { legalActions } from './actionBuilders';
import { applyAction } from './reducer';
import {
  PLAYER_A,
  PLAYER_B,
  findLegalActionByType,
  makeDefaultDistricts,
  makeGameState,
  makePlayer,
  makeResources,
  withDeed,
} from './__tests__/fixtures';
import type { GameAction } from './types';

function getPlayerAState(state: ReturnType<typeof makeGameState>) {
  return state.players[state.activePlayerIndex];
}

function getDistrict(state: ReturnType<typeof makeGameState>, districtId: string) {
  const district = state.districts.find((item) => item.id === districtId);
  if (!district) {
    throw new Error(`Missing district ${districtId}`);
  }
  return district;
}

describe('applyAction legality gate', () => {
  it('rejects actions that are illegal for the current phase', () => {
    const state = makeGameState({ phase: 'StartTurn' });
    const action: GameAction = { type: 'trade', give: 'Moons', receive: 'Suns' };
    expect(() => applyAction(state, action)).toThrow('Illegal action');
  });

  it('rejects modified payloads that do not exactly match legal actions', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const trade = findLegalActionByType(state, 'trade');
    const modified: GameAction = { ...trade, receive: trade.give };
    expect(() => applyAction(state, modified)).toThrow('Illegal action');
  });

  it('accepts exact legal action objects', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const trade = findLegalActionByType(state, 'trade');
    const next = applyAction(state, trade);
    expect(next.log).toHaveLength(1);
  });

  it('accepts cloned legal action payloads with same values', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Moons: 3 }) }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const trade = findLegalActionByType(state, 'trade');
    const cloned = JSON.parse(JSON.stringify(trade)) as GameAction;
    const next = applyAction(state, cloned);
    expect(next.log).toHaveLength(1);
  });
});

describe('optional-phase progression actions', () => {
  it('end-optional-trade transitions to OptionalDevelop', () => {
    const state = makeGameState({
      phase: 'OptionalTrade',
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const action = findLegalActionByType(state, 'end-optional-trade');
    const next = applyAction(state, action);
    expect(next.phase).toBe('OptionalDevelop');
  });

  it('end-optional-develop transitions to PlayCard', () => {
    const state = makeGameState({
      phase: 'OptionalDevelop',
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const action = findLegalActionByType(state, 'end-optional-develop');
    const next = applyAction(state, action);
    expect(next.phase).toBe('PlayCard');
  });
});

describe('buy-deed reducer semantics', () => {
  it('transitions to OptionalDevelop after buying a deed', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const buy = findLegalActionByType(state, 'buy-deed');
    const next = applyAction(state, buy);
    expect(next.phase).toBe('OptionalDevelop');
  });

  it('removes card from hand and creates deed with zero progress', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const buy = findLegalActionByType(state, 'buy-deed');
    const next = applyAction(state, buy);
    const player = getPlayerAState(next);
    const district = getDistrict(next, buy.districtId);

    expect(player.hand).not.toContain('6');
    expect(district.stacks[PLAYER_A].deed).toEqual({
      cardId: '6',
      progress: 0,
      tokens: {},
    });
  });

  it('debits deed cost from resources', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Knots: 1, Suns: 4 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const buy = findLegalActionByType(state, 'buy-deed');
    const next = applyAction(state, buy);
    const player = getPlayerAState(next);

    expect(player.resources.Moons).toBe(0);
    expect(player.resources.Knots).toBe(0);
    expect(player.resources.Suns).toBe(4);
  });
});

describe('develop-deed reducer semantics', () => {
  it('partial development increments progress and merges deed tokens', () => {
    const state = withDeed(
      makeGameState({
        phase: 'OptionalDevelop',
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 1, Knots: 1 }),
          }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '6', progress: 0, tokens: {} }
    );

    const action: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '6',
      tokens: { Moons: 1 },
    };
    const next = applyAction(state, action);
    const deed = getDistrict(next, 'D1').stacks[PLAYER_A].deed;

    expect(deed).toBeDefined();
    expect(deed?.progress).toBe(1);
    expect(deed?.tokens).toEqual({ Moons: 1 });
    expect(getPlayerAState(next).resources.Moons).toBe(0);
  });

  it('exact completion converts deed to developed property and clears deed', () => {
    const state = withDeed(
      makeGameState({
        phase: 'OptionalDevelop',
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Knots: 1 }),
          }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '6', progress: 1, tokens: { Moons: 1 } }
    );

    const action: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '6',
      tokens: { Knots: 1 },
    };
    const next = applyAction(state, action);
    const stack = getDistrict(next, 'D1').stacks[PLAYER_A];

    expect(stack.deed).toBeUndefined();
    expect(stack.developed).toContain('6');
  });

  it('rejects overspend payloads', () => {
    const state = withDeed(
      makeGameState({
        phase: 'OptionalDevelop',
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 5, Knots: 5 }),
          }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '6', progress: 1, tokens: { Moons: 1 } }
    );

    const overspend: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '6',
      tokens: { Moons: 2 },
    };
    expect(() => applyAction(state, overspend)).toThrow('Illegal action');
  });

  it('rejects wrong deed card references', () => {
    const state = withDeed(
      makeGameState({
        phase: 'OptionalDevelop',
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Moons: 1, Knots: 1, Suns: 2 }),
          }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '6', progress: 0, tokens: {} }
    );

    const wrongCard: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '7',
      tokens: { Suns: 1 },
    };
    expect(() => applyAction(state, wrongCard)).toThrow('Illegal action');
  });
});

describe('develop-outright reducer semantics', () => {
  it('requires exact payment total matching development cost', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 2, Knots: 2 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const action: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1 },
    };
    expect(() => applyAction(state, action)).toThrow('Illegal action');
  });

  it('requires at least one token from each card suit', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 2, Knots: 2 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const action: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 2 },
    };
    expect(() => applyAction(state, action)).toThrow('Illegal action');
  });

  it('rejects off-suit payment payloads', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Suns: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const action: GameAction = {
      type: 'develop-outright',
      cardId: '6',
      districtId: 'D1',
      payment: { Moons: 1, Suns: 1 },
    };
    expect(() => applyAction(state, action)).toThrow('Illegal action');
  });

  it('on success removes card from hand, adds developed card, and moves to DrawCard', () => {
    const state = makeGameState({
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

    const action = legalActions(state).find(
      (item): item is Extract<GameAction, { type: 'develop-outright' }> =>
        item.type === 'develop-outright' &&
        item.cardId === '6' &&
        item.districtId === 'D1' &&
        item.payment.Moons === 1 &&
        item.payment.Knots === 1
    );
    if (!action) {
      throw new Error('Missing expected legal develop-outright action.');
    }

    const next = applyAction(state, action);
    expect(next.phase).toBe('DrawCard');
    expect(getPlayerAState(next).hand).not.toContain('6');
    expect(getDistrict(next, 'D1').stacks[PLAYER_A].developed).toContain('6');
  });
});

describe('sell-card reducer semantics', () => {
  it('single-suit property sale grants +2 of that suit', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['0'],
          resources: makeResources({ Knots: 0 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const sell: GameAction = { type: 'sell-card', cardId: '0' };
    const next = applyAction(state, sell);
    expect(getPlayerAState(next).resources.Knots).toBe(2);
  });

  it('multi-suit property sale grants +1 per suit', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 0, Knots: 0 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const sell: GameAction = { type: 'sell-card', cardId: '6' };
    const next = applyAction(state, sell);
    expect(getPlayerAState(next).resources.Moons).toBe(1);
    expect(getPlayerAState(next).resources.Knots).toBe(1);
  });

  it('removes sold card from hand and puts it on top of discard', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources(),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });

    const next = applyAction(state, { type: 'sell-card', cardId: '6' });
    expect(getPlayerAState(next).hand).not.toContain('6');
    expect(next.deck.discard[0]).toBe('6');
  });

  it('transitions to DrawCard after selling', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [makePlayer(PLAYER_A, { hand: ['6'] }), makePlayer(PLAYER_B)] as const,
    });
    const next = applyAction(state, { type: 'sell-card', cardId: '6' });
    expect(next.phase).toBe('DrawCard');
  });
});

describe('issue regressions', () => {
  it('issue 2: invalid payloads are blocked by reducer legality checks', () => {
    const state = makeGameState({ phase: 'PlayCard' });
    const illegal: GameAction = {
      type: 'buy-deed',
      cardId: '6',
      districtId: 'NotADistrict',
    };
    expect(() => applyAction(state, illegal)).toThrow('Illegal action');
  });

  it('issue 3: buying deed preserves same-turn develop window', () => {
    const state = makeGameState({
      phase: 'PlayCard',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
    });
    const buy = findLegalActionByType(state, 'buy-deed');
    const next = applyAction(state, buy);
    expect(next.phase).toBe('OptionalDevelop');
  });

  it('issue 5: exact deed completion works while overspend remains blocked', () => {
    const state = withDeed(
      makeGameState({
        phase: 'OptionalDevelop',
        players: [
          makePlayer(PLAYER_A, { resources: makeResources({ Knots: 1, Moons: 2 }) }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '6', progress: 1, tokens: { Moons: 1 } }
    );

    const complete: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '6',
      tokens: { Knots: 1 },
    };
    const completed = applyAction(state, complete);
    expect(getDistrict(completed, 'D1').stacks[PLAYER_A].deed).toBeUndefined();

    const overspend: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '6',
      tokens: { Moons: 2 },
    };
    expect(() => applyAction(state, overspend)).toThrow('Illegal action');
  });
});
