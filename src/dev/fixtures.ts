import { GamePhase } from '../engine/values';
import type { CardId } from '../engine/cards';
import {
  decisionPlayerIdForState,
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { newGame } from '../engine/game';
import { isTerminal, scoreLive } from '../engine/scoring';
import { createSession, stepToDecision } from '../engine/session';
import { advanceToDecision } from '../engine/turnFlow';
import { type GameState, PlayerId, Suit } from '../engine/types';
import { selectHeuristicAction } from '../policies/heuristicScorer';

export type DevFixtureId =
  | 'multi-income'
  | 'late-game'
  | 'end-game-win'
  | 'deep-lanes'
  | 'd6-moons'
  | 'd6-wyrms'
  | 'd6-knots'
  | 'd6-suns';

const DEV_FIXTURE_PARAM = 'fixture';
const LATE_GAME_FIXTURE_SEED = 'dev-late-6';
const LATE_GAME_MAX_DECISIONS = 500;
const END_GAME_WIN_FIXTURE_SEED = 'dev-end-win-2';
const END_GAME_WIN_MAX_DECISIONS = 700;
const DEEP_LANES_FIXTURE_SEED = 'dev-fixture-deep-lanes';
const DEEP_LANE_COUNTS: readonly (readonly [number, number])[] = [
  [8, 7],
  [2, 2],
  [1, 1],
  [1, 1],
  [0, 0],
];
const HUMAN_MULTI_INCOME_DEED_CARDS: readonly CardId[] = ['6', '7'];
const BOT_MULTI_INCOME_DEED_CARDS: readonly CardId[] = ['8'];
const MULTI_INCOME_DEED_CARDS: readonly CardId[] = [
  ...HUMAN_MULTI_INCOME_DEED_CARDS,
  ...BOT_MULTI_INCOME_DEED_CARDS,
];

/*
  Fixtures are dev-server only by default. A production build can opt in at
  build time with `VITE_ENABLE_DEV_FIXTURES=true`, which is how the
  profiling/verification harness loads a deterministic board from a real
  bundle. Vite statically replaces the flag, so the default deployed build
  (unset) still tree-shakes the fixture code out.
*/
export const DEV_FIXTURES_ENABLED =
  import.meta.env.DEV || import.meta.env.VITE_ENABLE_DEV_FIXTURES === 'true';

export function devFixtureIdFromBrowserLocation(): DevFixtureId | null {
  if (!DEV_FIXTURES_ENABLED || typeof window === 'undefined') {
    return null;
  }
  return devFixtureIdFromSearch(window.location.search);
}

export function devFixtureIdFromSearch(search: string): DevFixtureId | null {
  if (!DEV_FIXTURES_ENABLED) {
    return null;
  }

  const fixtureId = new URLSearchParams(search).get(DEV_FIXTURE_PARAM);
  if (
    fixtureId === 'multi-income' ||
    fixtureId === 'late-game' ||
    fixtureId === 'end-game-win' ||
    fixtureId === 'deep-lanes' ||
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
    case 'end-game-win':
      return createEndGameWinFixture(humanPlayerId);
    case 'deep-lanes':
      return createDeepLanesFixture(humanPlayerId);
    case 'd6-moons':
      return createD6TaxFixture(humanPlayerId, Suit.Moons);
    case 'd6-wyrms':
      return createD6TaxFixture(humanPlayerId, Suit.Wyrms);
    case 'd6-knots':
      return createD6TaxFixture(humanPlayerId, Suit.Knots);
    case 'd6-suns':
      return createD6TaxFixture(humanPlayerId, Suit.Suns);
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

  // Moving fixture cards onto the board must not leave either opening hand short.
  const draw = state.deck.draw.filter((cardId) => !fixtureCardSet.has(cardId));
  const players = state.players.map((player) => {
    const hand = player.hand.filter((cardId) => !fixtureCardSet.has(cardId));
    while (hand.length < 3) {
      const replacement = draw.shift();
      if (!replacement)
        throw new Error('Multi-income fixture ran out of replacement cards.');
      hand.push(replacement);
    }
    return { ...player, hand };
  });

  return advanceToDecision({
    ...state,
    deck: {
      ...state.deck,
      draw,
      discard: state.deck.discard.filter(
        (cardId) => !fixtureCardSet.has(cardId)
      ),
    },
    players,
    activePlayerIndex,
    turn: 3,
    phase: GamePhase.CollectIncome,
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
        phase: GamePhase.CollectIncome,
        summary: 'Dev fixture: multiple partial-income choices',
      },
    ],
  });
}

function createD6TaxFixture(humanPlayerId: PlayerId, taxSuit: Suit): GameState {
  const state = newGame('dev-fixture-d6', { firstPlayer: humanPlayerId });
  return advanceToDecision({
    ...state,
    phase: GamePhase.CollectIncome,
    lastIncomeRoll: { die1: 1, die2: 5, rollId: 1 },
    lastTaxSuit: taxSuit,
  });
}

function createDeepLanesFixture(humanPlayerId: PlayerId): GameState {
  const botPlayerId = otherPlayerId(humanPlayerId);
  const state = newGame(DEEP_LANES_FIXTURE_SEED, {
    firstPlayer: humanPlayerId,
  });
  const activePlayerIndex = state.players.findIndex(
    (player) => player.id === humanPlayerId
  );
  if (activePlayerIndex < 0) {
    throw new Error(`Unknown human player for dev fixture: ${humanPlayerId}`);
  }

  const available = [...state.deck.draw];
  const takeCards = (count: number): CardId[] => {
    if (available.length < count) {
      throw new Error('Deep-lanes fixture ran out of cards.');
    }
    return available.splice(0, count);
  };

  const districts = state.districts.map((district, index) => {
    const counts = DEEP_LANE_COUNTS[index];
    if (!counts) {
      throw new Error(`Missing deep-lane counts for ${district.id}.`);
    }
    const [botCount, humanCount] = counts;
    return {
      ...district,
      stacks: {
        ...district.stacks,
        [botPlayerId]: {
          ...district.stacks[botPlayerId],
          developed: takeCards(botCount),
        },
        [humanPlayerId]: {
          ...district.stacks[humanPlayerId],
          developed: takeCards(humanCount),
        },
      },
    };
  });

  return appendFixtureLog(
    {
      ...state,
      deck: { ...state.deck, draw: available },
      districts,
      activePlayerIndex,
      turn: 12,
      phase: GamePhase.ActionWindow,
      lastIncomeRoll: { die1: 4, die2: 6 },
      cardPlayedThisTurn: false,
    },
    humanPlayerId,
    'Dev fixture: deep district lanes'
  );
}

function otherPlayerId(playerId: PlayerId): PlayerId {
  return playerId === PlayerId.PlayerA ? PlayerId.PlayerB : PlayerId.PlayerA;
}

function createLateGameFixture(humanPlayerId: PlayerId): GameState {
  const state = rolloutHeuristicTo({
    seed: LATE_GAME_FIXTURE_SEED,
    humanPlayerId,
    maxDecisions: LATE_GAME_MAX_DECISIONS,
    label: 'Late-game dev fixture',
    isTarget: (candidate) =>
      (candidate.finalTurnsRemaining ?? 0) === 2 &&
      candidate.phase === GamePhase.ActionWindow,
  });

  return appendFixtureLog(
    state,
    humanPlayerId,
    'Dev fixture: late game rollout'
  );
}

/*
  A near-terminal board the human wins: the rollout stops on the human's own
  final turn (finalTurnsRemaining === 1) after their card play, so End Turn is
  available immediately. Ending that turn finalizes the game before the bot can
  respond, and the human leads at that point, so the win is locked in — which is
  what makes this fixture useful for exercising the end-game win celebration.
*/
function createEndGameWinFixture(humanPlayerId: PlayerId): GameState {
  const state = rolloutHeuristicTo({
    seed: END_GAME_WIN_FIXTURE_SEED,
    humanPlayerId,
    maxDecisions: END_GAME_WIN_MAX_DECISIONS,
    label: 'End-game win dev fixture',
    isTarget: (candidate) =>
      candidate.finalTurnsRemaining === 1 &&
      candidate.phase === GamePhase.ActionWindow &&
      candidate.cardPlayedThisTurn &&
      decisionPlayerIdForState(candidate) === humanPlayerId,
  });

  if (scoreLive(state).winner !== humanPlayerId) {
    throw new Error(
      'End-game win dev fixture reached the human final turn without the human ahead.'
    );
  }

  return appendFixtureLog(
    state,
    humanPlayerId,
    'Dev fixture: final human turn, human ahead'
  );
}

function rolloutHeuristicTo({
  seed,
  humanPlayerId,
  maxDecisions,
  label,
  isTarget,
}: {
  seed: string;
  humanPlayerId: PlayerId;
  maxDecisions: number;
  label: string;
  isTarget: (state: GameState) => boolean;
}): GameState {
  let state = createSession(seed, humanPlayerId);

  for (
    let decisionCount = 0;
    decisionCount < maxDecisions && !isTerminal(state);
    decisionCount += 1
  ) {
    if (isTarget(state)) {
      return state;
    }

    const decisionPlayerId = decisionPlayerIdForState(state);
    if (
      decisionPlayerId !== PlayerId.PlayerA &&
      decisionPlayerId !== PlayerId.PlayerB
    ) {
      throw new Error(`${label} could not resolve decision player.`);
    }

    const actions = legalActionsForDecisionPlayer(state, decisionPlayerId);
    const action = selectHeuristicAction({
      state,
      view: toDecisionPlayerView(state, decisionPlayerId),
      legalActions: actions,
    });
    if (!action) {
      throw new Error(`${label} rollout had no selected action.`);
    }

    state = stepToDecision(state, action);
  }

  throw new Error(
    `${label} did not reach its target within ${String(maxDecisions)} decisions.`
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
