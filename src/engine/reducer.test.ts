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
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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

describe('post-card progression actions', () => {
  it('end-turn transitions to DrawCard from ActionWindow after a card play', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const action = findLegalActionByType(state, 'end-turn');
    const next = applyAction(state, action);
    expect(next.phase).toBe('DrawCard');
  });

  it('end-turn is illegal before a card has been played', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      cardPlayedThisTurn: false,
      players: [makePlayer(PLAYER_A), makePlayer(PLAYER_B)] as const,
    });
    const endTurn: GameAction = { type: 'end-turn' };
    expect(() => applyAction(state, endTurn)).toThrow('Illegal action');
  });
});

describe('income choice reducer semantics', () => {
  it('choose-income-suit applies selected suit gain and advances when last pending choice resolves', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      players: [
        makePlayer(PLAYER_A, { resources: makeResources({ Suns: 0, Wyrms: 0 }) }),
        makePlayer(PLAYER_B),
      ] as const,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
      ],
      incomeChoiceReturnPlayerId: PLAYER_A,
    });

    const action: GameAction = {
      type: 'choose-income-suit',
      playerId: PLAYER_A,
      districtId: 'D1',
      cardId: '7',
      suit: 'Wyrms',
    };
    const next = applyAction(state, action);

    const playerA = next.players.find((player) => player.id === PLAYER_A);
    if (!playerA) {
      throw new Error('Missing PlayerA.');
    }
    expect(playerA.resources.Wyrms).toBe(1);
    expect(next.phase).toBe('ActionWindow');
    expect(next.activePlayerIndex).toBe(0);
    expect(next.pendingIncomeChoices).toBeUndefined();
    expect(next.incomeChoiceReturnPlayerId).toBeUndefined();
  });

  it('choose-income-suit keeps CollectIncome phase when more choices remain', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
        {
          playerId: PLAYER_B,
          districtId: 'D2',
          cardId: '8',
          suits: ['Waves', 'Leaves'],
        },
      ],
    });
    const action = findLegalActionByType(state, 'choose-income-suit');
    const next = applyAction(state, action);

    expect(next.phase).toBe('CollectIncome');
    expect(next.activePlayerIndex).toBe(1);
    expect(next.pendingIncomeChoices).toHaveLength(1);
    expect(next.pendingIncomeChoices?.[0].playerId).toBe(PLAYER_B);
  });
});

describe('buy-deed reducer semantics', () => {
  it('transitions to ActionWindow after buying a deed', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
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
    expect(next.phase).toBe('ActionWindow');
    expect(next.cardPlayedThisTurn).toBe(true);
  });

  it('removes card from hand and creates deed with zero progress', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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
        phase: 'ActionWindow',
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
        phase: 'ActionWindow',
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

  it('ace deeds require total progress of 3 to complete', () => {
    const state = withDeed(
      makeGameState({
        phase: 'ActionWindow',
        players: [
          makePlayer(PLAYER_A, {
            resources: makeResources({ Knots: 1 }),
          }),
          makePlayer(PLAYER_B),
        ] as const,
      }),
      'D1',
      PLAYER_A,
      { cardId: '0', progress: 2, tokens: { Knots: 2 } }
    );

    const action: GameAction = {
      type: 'develop-deed',
      districtId: 'D1',
      cardId: '0',
      tokens: { Knots: 1 },
    };
    const next = applyAction(state, action);
    const stack = getDistrict(next, 'D1').stacks[PLAYER_A];

    expect(stack.deed).toBeUndefined();
    expect(stack.developed).toContain('0');
  });

  it('rejects overspend payloads', () => {
    const state = withDeed(
      makeGameState({
        phase: 'ActionWindow',
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
        phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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

  it('on success removes card from hand, adds developed card, and moves to post-play optionals', () => {
    const state = makeGameState({
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
    expect(next.phase).toBe('ActionWindow');
    expect(next.cardPlayedThisTurn).toBe(true);
    expect(getPlayerAState(next).hand).not.toContain('6');
    expect(getDistrict(next, 'D1').stacks[PLAYER_A].developed).toContain('6');
  });
});

describe('sell-card reducer semantics', () => {
  it('single-suit property sale grants +2 of that suit', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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
      phase: 'ActionWindow',
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

  it('transitions to post-play optionals after selling', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [makePlayer(PLAYER_A, { hand: ['6'] }), makePlayer(PLAYER_B)] as const,
    });
    const next = applyAction(state, { type: 'sell-card', cardId: '6' });
    expect(next.phase).toBe('ActionWindow');
    expect(next.cardPlayedThisTurn).toBe(true);
  });
});

describe('one card per turn flow', () => {
  it('cannot perform a second card play after card play; end-turn remains available', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6', '7'],
          resources: makeResources({ Moons: 1, Knots: 1 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
      districts: makeDefaultDistricts(),
    });

    const play = legalActions(state).find(
      (item): item is Extract<GameAction, { type: 'develop-outright' }> =>
        item.type === 'develop-outright' &&
        item.cardId === '6' &&
        item.districtId === 'D1' &&
        item.payment.Moons === 1 &&
        item.payment.Knots === 1
    );
    if (!play) {
      throw new Error('Missing expected legal develop-outright action.');
    }

    const afterPlay = applyAction(state, play);
    expect(afterPlay.phase).toBe('ActionWindow');

    expect(legalActions(afterPlay).some((action) => action.type === 'buy-deed')).toBe(false);
    expect(legalActions(afterPlay).some((action) => action.type === 'sell-card')).toBe(false);
    expect(legalActions(afterPlay).some((action) => action.type === 'develop-outright')).toBe(
      false
    );
    expect(legalActions(afterPlay).some((action) => action.type === 'trade')).toBe(false);
    expect(legalActions(afterPlay).some((action) => action.type === 'develop-deed')).toBe(
      false
    );
    expect(legalActions(afterPlay).some((action) => action.type === 'end-turn')).toBe(true);

    const draw = applyAction(afterPlay, findLegalActionByType(afterPlay, 'end-turn'));
    expect(draw.phase).toBe('DrawCard');
  });
});

describe('issue regressions', () => {
  it('issue 2: invalid payloads are blocked by reducer legality checks', () => {
    const state = makeGameState({ phase: 'ActionWindow' });
    const illegal: GameAction = {
      type: 'buy-deed',
      cardId: '6',
      districtId: 'NotADistrict',
    };
    expect(() => applyAction(state, illegal)).toThrow('Illegal action');
  });

  it('blocks first-placement actions in districts without marker-suit overlap', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
      players: [
        makePlayer(PLAYER_A, {
          hand: ['7'],
          resources: makeResources({ Suns: 2, Wyrms: 2 }),
        }),
        makePlayer(PLAYER_B),
      ] as const,
      districts: makeDefaultDistricts(),
    });

    const illegalBuy: GameAction = {
      type: 'buy-deed',
      cardId: '7',
      districtId: 'D1',
    };
    expect(() => applyAction(state, illegalBuy)).toThrow('Illegal action');

    const illegalOutright: GameAction = {
      type: 'develop-outright',
      cardId: '7',
      districtId: 'D1',
      payment: { Suns: 1, Wyrms: 1 },
    };
    expect(() => applyAction(state, illegalOutright)).toThrow('Illegal action');
  });

  it('issue 3: buying deed preserves same-turn develop window', () => {
    const state = makeGameState({
      phase: 'ActionWindow',
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
    expect(next.phase).toBe('ActionWindow');
  });

  it('issue 5: exact deed completion works while overspend remains blocked', () => {
    const state = withDeed(
      makeGameState({
        phase: 'ActionWindow',
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
