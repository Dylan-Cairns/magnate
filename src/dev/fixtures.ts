import type { CardId } from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { newGame } from '../engine/game';
import { isTerminal } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import { advanceToDecision } from '../engine/turnFlow';
import type { GameState, PlayerId, Suit } from '../engine/types';
import { selectHeuristicAction } from '../policies/heuristicScorer';

export type DevFixtureId =
  | 'multi-income'
  | 'late-game'
  | 'd6-moons'
  | 'd6-wyrms'
  | 'd6-knots'
  | 'd6-suns';

const DEV_FIXTURE_PARAM = 'fixture';
const LATE_GAME_FIXTURE_SEED = 'dev-late-6';
const LATE_GAME_MAX_DECISIONS = 500;
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
  if (
    fixtureId === 'multi-income' ||
    fixtureId === 'late-game' ||
    fixtureId === 'd6-moons' ||
    fixtureId === 'd6-wyrms' ||
    fixtureId === 'd6-knots' ||
    fixtureId === 'd6-suns'
  ) {
    return fixtureId;
  }
  return null;
}

export function createDevFixtureSession(
  fixtureId: DevFixtureId,
  humanPlayerId: PlayerId
): GameState {
  switch (fixtureId) {
    case 'multi-income':
      return createMultiIncomeFixture(humanPlayerId);
    case 'late-game':
      return createLateGameFixture(humanPlayerId);
    case 'd6-moons':
      return createD6TaxFixture(humanPlayerId, 'Moons');
    case 'd6-wyrms':
      return createD6TaxFixture(humanPlayerId, 'Wyrms');
    case 'd6-knots':
      return createD6TaxFixture(humanPlayerId, 'Knots');
    case 'd6-suns':
      return createD6TaxFixture(humanPlayerId, 'Suns');
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

function createD6TaxFixture(humanPlayerId: PlayerId, taxSuit: Suit): GameState {
  const state = newGame('dev-fixture-d6', { firstPlayer: humanPlayerId });
  return advanceToDecision({
    ...state,
    phase: 'CollectIncome',
    lastIncomeRoll: { die1: 1, die2: 5, rollId: 1 },
    lastTaxSuit: taxSuit,
  });
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === 'PlayerA' ? 'PlayerB' : 'PlayerA';
}

function createLateGameFixture(humanPlayerId: PlayerId): GameState {
  let state = createSession(LATE_GAME_FIXTURE_SEED, humanPlayerId);

  for (
    let decisionCount = 0;
    decisionCount < LATE_GAME_MAX_DECISIONS && !isTerminal(state);
    decisionCount += 1
  ) {
    if (
      (state.finalTurnsRemaining ?? 0) === 2 &&
      state.phase === 'ActionWindow'
    ) {
      return appendFixtureLog(
        state,
        humanPlayerId,
        'Dev fixture: late game rollout'
      );
    }

    const decisionPlayerId = decisionPlayerIdForState(state);
    if (decisionPlayerId !== 'PlayerA' && decisionPlayerId !== 'PlayerB') {
      throw new Error(
        'Late-game dev fixture could not resolve decision player.'
      );
    }

    const actions = legalActionsForDecisionPlayer(state, decisionPlayerId);
    const action = selectHeuristicAction({
      state,
      view: toDecisionPlayerView(state, decisionPlayerId),
      legalActions: actions,
    });
    if (!action) {
      throw new Error('Late-game dev fixture rollout had no selected action.');
    }

    state = stepToDecision(state, action);
  }

  throw new Error(
    `Late-game dev fixture did not reach final turns within ${String(
      LATE_GAME_MAX_DECISIONS
    )} decisions.`
  );
}

function appendFixtureLog(
  state: GameState,
  humanPlayerId: PlayerId,
  summary: string
): GameState {
  return {
    ...state,
    log: [
      ...state.log,
      {
        turn: state.turn,
        player: humanPlayerId,
        phase: state.phase,
        summary,
      },
    ],
  };
}
