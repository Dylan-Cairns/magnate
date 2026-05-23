import { describe, expect, it } from 'vitest';

import { toActivePlayerView, toPlayerView } from './view';
import {
  makeGameState,
  makePlayer,
  makeResources,
  PLAYER_A,
  PLAYER_B,
} from './__tests__/fixtures';

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
    const viewAPlayerA = asPlayerA.players.find(
      (player) => player.id === PLAYER_A
    );
    const viewAPlayerB = asPlayerA.players.find(
      (player) => player.id === PLAYER_B
    );
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
      submittedIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
        },
      ],
      incomeChoiceReturnPlayerId: PLAYER_A,
    });

    const view = toPlayerView(state, PLAYER_A);
    expect(view.activePlayerId).toBe(PLAYER_B);
    expect(
      view.players.find((player) => player.id === PLAYER_A)?.resources.Moons
    ).toBe(2);
    expect(
      view.players.find((player) => player.id === PLAYER_B)?.crowns
    ).toEqual(['33', '34', '35']);
    expect(view.pendingIncomeChoices).toEqual([
      {
        playerId: PLAYER_B,
        districtId: 'D2',
        cardId: '7',
        suits: ['Suns', 'Wyrms'],
      },
    ]);
    expect(view.submittedIncomeChoices).toEqual([
      {
        playerId: PLAYER_A,
        districtId: 'D1',
        cardId: '6',
        suit: 'Moons',
      },
    ]);
    expect(view.incomeChoiceReturnPlayerId).toBe(PLAYER_A);
  });

  it('returns cloned data so mutating view does not mutate state', () => {
    // Both players have pending choices; PlayerA has already submitted.
    // PlayerA can see their own submission, so we can verify mutation isolation.
    const state = makeGameState({
      players: [
        makePlayer(PLAYER_A, {
          hand: ['6'],
          resources: makeResources({ Moons: 1 }),
        }),
        makePlayer(PLAYER_B, {
          hand: ['7'],
          resources: makeResources({ Suns: 1 }),
        }),
      ] as const,
      pendingIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suits: ['Moons', 'Knots'],
        },
        {
          playerId: PLAYER_B,
          districtId: 'D2',
          cardId: '7',
          suits: ['Suns', 'Wyrms'],
        },
      ],
      submittedIncomeChoices: [
        {
          playerId: PLAYER_A,
          districtId: 'D1',
          cardId: '6',
          suit: 'Moons',
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
    if (view.submittedIncomeChoices) {
      const choice = view.submittedIncomeChoices[0];
      const mutableChoice = choice as {
        suit: string;
      };
      mutableChoice.suit = 'Wyrms';
    }

    expect(state.players[0].resources.Moons).toBe(1);
    expect(state.players[0].hand).toEqual(['6']);
    expect(state.deck.discard).toEqual([]);
    expect(state.districts[0].stacks.PlayerA.developed).toEqual([]);
    expect(state.pendingIncomeChoices?.[0].suits).toEqual(['Moons', 'Knots']);
    expect(state.submittedIncomeChoices?.[0].suit).toBe('Moons');
  });

  it('hides opponent submitted income choice and log entry until all choices are submitted', () => {
    // PlayerB submitted first; PlayerA has not yet submitted.
    // PlayerA's view must not reveal PlayerB's suit choice.
    const state = {
      ...makeGameState({
        phase: 'CollectIncome',
        players: [
          makePlayer(PLAYER_A, { hand: ['6'] }),
          makePlayer(PLAYER_B, { hand: ['7'] }),
        ] as const,
        pendingIncomeChoices: [
          {
            playerId: PLAYER_A,
            districtId: 'D1',
            cardId: '6',
            suits: ['Moons', 'Knots'],
          },
          {
            playerId: PLAYER_B,
            districtId: 'D2',
            cardId: '7',
            suits: ['Suns', 'Wyrms'],
          },
        ],
        submittedIncomeChoices: [
          {
            playerId: PLAYER_B,
            districtId: 'D2',
            cardId: '7',
            suit: 'Suns',
          },
        ],
        incomeChoiceReturnPlayerId: PLAYER_A,
      }),
      log: [
        {
          turn: 1,
          player: PLAYER_B,
          phase: 'CollectIncome' as const,
          summary: 'income choice 7:Suns',
        },
      ],
    };

    const viewAsA = toPlayerView(state, PLAYER_A);
    // PlayerA's submitted choices should be absent (they haven't submitted yet)
    expect(viewAsA.submittedIncomeChoices).toBeUndefined();
    // The bot's income choice log entry must not appear in PlayerA's log
    expect(
      viewAsA.log.some((e) => e.summary.startsWith('income choice '))
    ).toBe(false);

    // PlayerB's own view should still see their own submission
    const viewAsB = toPlayerView(state, PLAYER_B);
    expect(viewAsB.submittedIncomeChoices).toEqual([
      {
        playerId: PLAYER_B,
        districtId: 'D2',
        cardId: '7',
        suit: 'Suns',
      },
    ]);
    // And PlayerB's own log entry is visible to them
    expect(viewAsB.log.some((e) => e.summary === 'income choice 7:Suns')).toBe(
      true
    );
  });

  it('reveals all submitted income choices in the log once all players have submitted', () => {
    // Both players have now submitted — selection is complete.
    const state = {
      ...makeGameState({
        phase: 'CollectIncome',
        players: [
          makePlayer(PLAYER_A, { hand: ['6'] }),
          makePlayer(PLAYER_B, { hand: ['7'] }),
        ] as const,
        pendingIncomeChoices: [
          {
            playerId: PLAYER_A,
            districtId: 'D1',
            cardId: '6',
            suits: ['Moons', 'Knots'],
          },
          {
            playerId: PLAYER_B,
            districtId: 'D2',
            cardId: '7',
            suits: ['Suns', 'Wyrms'],
          },
        ],
        submittedIncomeChoices: [
          {
            playerId: PLAYER_B,
            districtId: 'D2',
            cardId: '7',
            suit: 'Suns',
          },
          {
            playerId: PLAYER_A,
            districtId: 'D1',
            cardId: '6',
            suit: 'Moons',
          },
        ],
        incomeChoiceReturnPlayerId: PLAYER_A,
      }),
      log: [
        {
          turn: 1,
          player: PLAYER_B,
          phase: 'CollectIncome' as const,
          summary: 'income choice 7:Suns',
        },
        {
          turn: 1,
          player: PLAYER_A,
          phase: 'CollectIncome' as const,
          summary: 'income choice 6:Moons',
        },
      ],
    };

    const viewAsA = toPlayerView(state, PLAYER_A);
    expect(viewAsA.submittedIncomeChoices).toHaveLength(2);
    expect(viewAsA.log.filter((e) => e.summary.startsWith('income choice '))).toHaveLength(2);
  });

  it('throws for unknown viewer ids', () => {
    const state = makeGameState();
    expect(() => toPlayerView(state, 'Ghost' as never)).toThrow(
      'Unknown player'
    );
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
