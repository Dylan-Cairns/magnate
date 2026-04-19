import type { CardId } from '../engine/cards';
import { newGame } from '../engine/game';
import { advanceToDecision } from '../engine/turnFlow';
import type { GameState, PlayerId } from '../engine/types';

export type DevFixtureId = 'multi-income';

const DEV_FIXTURE_PARAM = 'fixture';
const HUMAN_MULTI_INCOME_DEED_CARDS: readonly CardId[] = ['6', '7'];
const BOT_MULTI_INCOME_DEED_CARDS: readonly CardId[] = ['8'];
const MULTI_INCOME_DEED_CARDS: readonly CardId[] = [
  ...HUMAN_MULTI_INCOME_DEED_CARDS,
  ...BOT_MULTI_INCOME_DEED_CARDS,
];

export function devFixtureIdFromBrowserLocation(): DevFixtureId | null {
  if (!import.meta.env.DEV || typeof window === 'undefined') {
    return null;
  }
  return devFixtureIdFromSearch(window.location.search);
}

export function devFixtureIdFromSearch(search: string): DevFixtureId | null {
  if (!import.meta.env.DEV) {
    return null;
  }

  const fixtureId = new URLSearchParams(search).get(DEV_FIXTURE_PARAM);
  return fixtureId === 'multi-income' ? fixtureId : null;
}

export function createDevFixtureSession(
  fixtureId: DevFixtureId,
  humanPlayerId: PlayerId
): GameState {
  switch (fixtureId) {
    case 'multi-income':
      return createMultiIncomeFixture(humanPlayerId);
  }
}

function createMultiIncomeFixture(humanPlayerId: PlayerId): GameState {
  const fixtureCardSet = new Set<CardId>(MULTI_INCOME_DEED_CARDS);
  const botPlayerId = otherPlayerId(humanPlayerId);
  const state = newGame('dev-fixture-multi-income', {
    firstPlayer: humanPlayerId,
  });
  const activePlayerIndex = state.players.findIndex(
    (player) => player.id === humanPlayerId
  );
  if (activePlayerIndex < 0) {
    throw new Error(`Unknown human player for dev fixture: ${humanPlayerId}`);
  }

  return advanceToDecision({
    ...state,
    deck: {
      ...state.deck,
      draw: state.deck.draw.filter((cardId) => !fixtureCardSet.has(cardId)),
      discard: state.deck.discard.filter(
        (cardId) => !fixtureCardSet.has(cardId)
      ),
    },
    players: state.players.map((player) => ({
      ...player,
      hand: player.hand.filter((cardId) => !fixtureCardSet.has(cardId)),
      crowns: player.crowns.filter((cardId) => !fixtureCardSet.has(cardId)),
    })),
    activePlayerIndex,
    turn: 3,
    phase: 'CollectIncome',
    districts: state.districts.map((district, index) => {
      const humanDeedCardId = HUMAN_MULTI_INCOME_DEED_CARDS[index];
      const botDeedCardId =
        BOT_MULTI_INCOME_DEED_CARDS[
          index - HUMAN_MULTI_INCOME_DEED_CARDS.length
        ];

      return {
        ...district,
        stacks: {
          ...district.stacks,
          ...(humanDeedCardId
            ? {
                [humanPlayerId]: {
                  ...district.stacks[humanPlayerId],
                  deed: {
                    cardId: humanDeedCardId,
                    progress: 0,
                    tokens: {},
                  },
                },
              }
            : {}),
          ...(botDeedCardId
            ? {
                [botPlayerId]: {
                  ...district.stacks[botPlayerId],
                  deed: {
                    cardId: botDeedCardId,
                    progress: 0,
                    tokens: {},
                  },
                },
              }
            : {}),
        },
      };
    }),
    cardPlayedThisTurn: false,
    lastIncomeRoll: { die1: 2, die2: 2 },
    pendingIncomeChoices: undefined,
    submittedIncomeChoices: undefined,
    incomeChoiceReturnPlayerId: undefined,
    log: [
      ...state.log,
      {
        turn: 3,
        player: humanPlayerId,
        phase: 'CollectIncome',
        summary: 'Dev fixture: multiple partial-income choices',
      },
    ],
  });
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}
