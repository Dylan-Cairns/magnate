import { describe, expect, it } from 'vitest';

import { toActivePlayerView, toPlayerView } from './view';
import { makeGameState, makePlayer, makeResources, PLAYER_A, PLAYER_B } from './__tests__/fixtures';

describe('toPlayerView', () => {
  it('shows own hand and hides opponent hand contents', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6', '7', '8'],
          resources: makeResources({ Moons: 2 }),
        }),
        makePlayer(PLAYER_B, {
          hand: ['9', '10'],
          resources: makeResources({ Suns: 3 }),
        }),
      ] as const,
    });

    const asPlayerA = toPlayerView(state, PLAYER_A);
    const viewAPlayerA = asPlayerA.players.find((player) => player.id === PLAYER_A);
    const viewAPlayerB = asPlayerA.players.find((player) => player.id === PLAYER_B);
    if (!viewAPlayerA || !viewAPlayerB) {
      throw new Error('Missing expected players in view.');
    }

    expect(viewAPlayerA.hand).toEqual(['6', '7', '8']);
    expect(viewAPlayerA.handCount).toBe(3);
    expect(viewAPlayerA.handHidden).toBe(false);

    expect(viewAPlayerB.hand).toEqual([]);
    expect(viewAPlayerB.handCount).toBe(2);
    expect(viewAPlayerB.handHidden).toBe(true);
  });

  it('hides draw pile ordering but exposes public deck info', () => {
    const state = makeGameState({
      deck: {
        draw: ['6', '7', '8', '9'],
        discard: ['10', '11'],
        reshuffles: 1,
      },
    });

    const view = toPlayerView(state, PLAYER_A);
    expect(view.deck.drawCount).toBe(4);
    expect(view.deck.discard).toEqual(['10', '11']);
    expect(view.deck.reshuffles).toBe(1);
    expect((view.deck as unknown as { draw?: string[] }).draw).toBeUndefined();
  });

  it('preserves public board/resource/crown and pending choice information', () => {
    const state = makeGameState({
      phase: 'CollectIncome',
      activePlayerIndex: 1,
      players: [
        makePlayer(PLAYER_A, {
          crowns: ['30', '31', '32'],
          resources: makeResources({ Moons: 2, Knots: 1 }),
        }),
        makePlayer(PLAYER_B, {
          crowns: ['33', '34', '35'],
          resources: makeResources({ Suns: 3 }),
        }),
      ] as const,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_B,
          districtId: 'D2',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
      ],
      incomeChoiceReturnPlayerId: PLAYER_A,
    });

    const view = toPlayerView(state, PLAYER_A);
    expect(view.activePlayerId).toBe(PLAYER_B);
    expect(view.players.find((player) => player.id === PLAYER_A)?.resources.Moons).toBe(2);
    expect(view.players.find((player) => player.id === PLAYER_B)?.crowns).toEqual(['33', '34', '35']);
    expect(view.pendingIncomeChoices).toEqual([
      {
        playerId: PLAYER_B,
        districtId: 'D2',
        cardId: '7',
        suits: ['Suns', 'Wyrms'],
      },
    ]);
    expect(view.incomeChoiceReturnPlayerId).toBe(PLAYER_A);
  });

  it('returns cloned data so mutating view does not mutate state', () => {
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, { hand: ['6'], resources: makeResources({ Moons: 1 }) }),
        makePlayer(PLAYER_B, { hand: ['7'], resources: makeResources({ Suns: 1 }) }),
      ] as const,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Knots'],
        },
      ],
    });

    const view = toPlayerView(state, PLAYER_A);

    view.players[0].resources.Moons = 99;
    view.players[0].hand.push('8');
    view.deck.discard.push('9');
    view.districts[0].stacks.PlayerA.developed.push('6');
    if (view.pendingIncomeChoices) {
      const choice = view.pendingIncomeChoices[0];
      const mutableChoice = choice as {
        playerId: string;
        districtId: string;
        cardId: string;
        suits: string[];
      };
      mutableChoice.suits.push('Suns');
    }

    expect(state.players[0].resources.Moons).toBe(1);
    expect(state.players[0].hand).toEqual(['6']);
    expect(state.deck.discard).toEqual([]);
    expect(state.districts[0].stacks.PlayerA.developed).toEqual([]);
    expect(state.pendingIncomeChoices?.[0].suits).toEqual(['Moons', 'Knots']);
  });

  it('throws for unknown viewer ids', () => {
    const state = makeGameState();
    expect(() => toPlayerView(state, 'Ghost' as never)).toThrow('Unknown player');
  });
});

describe('toActivePlayerView', () => {
  it('uses active player as the viewer', () => {
    const state = makeGameState({
      activePlayerIndex: 1,
      players: [
        makePlayer(PLAYER_A, { hand: ['6', '7'] }),
        makePlayer(PLAYER_B, { hand: ['8', '9'] }),
      ] as const,
    });

    const view = toActivePlayerView(state);
    const playerA = view.players.find((player) => player.id === PLAYER_A);
    const playerB = view.players.find((player) => player.id === PLAYER_B);
    if (!playerA || !playerB) {
      throw new Error('Missing expected players in view.');
    }

    expect(view.viewerId).toBe(PLAYER_B);
    expect(playerB.hand).toEqual(['8', '9']);
    expect(playerA.hand).toEqual([]);
    expect(playerA.handCount).toBe(2);
  });
});
