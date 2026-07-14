import type { CardId } from './cards';
import { legalActions } from './actionBuilders';
import { drawOne } from './deck';
import { incomeForResult } from './income';
import { rngFromSeed } from './rng';
import { scoreGame } from './scoring';
import { applyDelta } from './stateHelpers';
import type {
  GamePhase,
  GameState,
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  Rank,
  Suit,
} from './types';

const BASE_DECISION_PHASES: ReadonlySet<GamePhase> = new Set(['GameOver']);

const MAX_ADVANCE_STEPS = 32;
const TAX_SUIT_BY_D6: readonly [Suit, Suit, Suit, Suit, Suit, Suit] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

export interface AdvanceToDecisionOptions {
  assumeActionWindowDecision?: boolean;
}

export function advanceToDecision(
  state: GameState,
  options: AdvanceToDecisionOptions = {}
): GameState {
  let current = state;

  for (let i = 0; i < MAX_ADVANCE_STEPS; i += 1) {
    if (isDecisionPhase(current, options)) {
      return current;
    }

    current = advanceOnePhase(current);
  }

  throw new Error(
    `advanceToDecision exceeded ${MAX_ADVANCE_STEPS} phase transitions.`
  );
}

function isDecisionPhase(
  state: GameState,
  options: AdvanceToDecisionOptions
): boolean {
  if (state.phase === 'ActionWindow') {
    if (options.assumeActionWindowDecision) {
      return true;
    }
    return legalActions(state).length > 0;
  }
  if (BASE_DECISION_PHASES.has(state.phase)) {
    return true;
  }
  if (state.phase === 'CollectIncome') {
    return hasUnsubmittedIncomeChoices(state);
  }
  return false;
}

function advanceOnePhase(state: GameState): GameState {
  switch (state.phase) {
    case 'StartTurn':
      return {
        ...state,
        phase: 'TaxCheck',
      };
    case 'TaxCheck':
      return resolveTaxCheck(state);
    case 'CollectIncome':
      return resolveCollectIncome(state);
    case 'DrawCard':
      return resolveDrawPhase(state);
    case 'ActionWindow':
    case 'GameOver':
      return state;
  }
}

function resolveTaxCheck(state: GameState): GameState {
  const first = rollDie(state.seed, state.rngCursor, 10);
  const second = rollDie(state.seed, first.rngCursor, 10);
  const incomeRoll: IncomeRollResult = {
    die1: first.value,
    die2: second.value,
    rollId: second.rngCursor,
  };

  let nextState: GameState = {
    ...state,
    rngCursor: second.rngCursor,
    lastIncomeRoll: incomeRoll,
    lastTaxSuit: undefined,
  };

  if (incomeRoll.die1 === 1 || incomeRoll.die2 === 1) {
    const taxSuitRoll = rollDie(state.seed, second.rngCursor, 6);
    const taxSuit = TAX_SUIT_BY_D6[taxSuitRoll.value - 1];
    nextState = {
      ...nextState,
      rngCursor: taxSuitRoll.rngCursor,
      lastTaxSuit: taxSuit,
      players: applyTaxation(nextState, taxSuit),
    };
  }

  return {
    ...nextState,
    phase: 'CollectIncome',
  };
}

function resolveCollectIncome(state: GameState): GameState {
  if (!state.lastIncomeRoll) {
    throw new Error(
      'CollectIncome phase requires lastIncomeRoll to be present.'
    );
  }
  if ((state.pendingIncomeChoices?.length ?? 0) > 0) {
    if (hasUnsubmittedIncomeChoices(state)) {
      return state;
    }

    const returnPlayerId = state.incomeChoiceReturnPlayerId;
    if (!returnPlayerId) {
      throw new Error('Missing return player while resolving income choices.');
    }

    return {
      ...resolveSubmittedIncomeChoices(state),
      activePlayerIndex: findPlayerIndexById(state, returnPlayerId),
      phase: 'ActionWindow',
      cardPlayedThisTurn: false,
      pendingIncomeChoices: undefined,
      submittedIncomeChoices: undefined,
      incomeChoiceReturnPlayerId: undefined,
    };
  }
  const incomeRoll = state.lastIncomeRoll;

  const pendingChoices: IncomeChoice[] = [];
  const players = state.players.map((player) => {
    const income = incomeDeltaForPlayer(state, player.id, incomeRoll);
    pendingChoices.push(...income.pendingChoices);
    return {
      ...player,
      resources: applyDelta(player.resources, income.delta),
    };
  });

  if (pendingChoices.length > 0) {
    return {
      ...state,
      players,
      phase: 'CollectIncome',
      pendingIncomeChoices: pendingChoices,
      submittedIncomeChoices: undefined,
      incomeChoiceReturnPlayerId: state.players[state.activePlayerIndex]?.id,
    };
  }

  return {
    ...state,
    players,
    phase: 'ActionWindow',
    cardPlayedThisTurn: false,
    pendingIncomeChoices: undefined,
    submittedIncomeChoices: undefined,
    incomeChoiceReturnPlayerId: undefined,
  };
}

function hasUnsubmittedIncomeChoices(state: GameState): boolean {
  const submitted = state.submittedIncomeChoices ?? [];
  return (state.pendingIncomeChoices ?? []).some(
    (choice) => !submitted.some((entry) => incomeChoiceMatches(choice, entry))
  );
}

function resolveSubmittedIncomeChoices(state: GameState): GameState {
  const pendingChoices = state.pendingIncomeChoices ?? [];
  const submissions = state.submittedIncomeChoices ?? [];
  let next = state;

  pendingChoices.forEach((choice) => {
    const submission = submissions.find((entry) =>
      incomeChoiceMatches(choice, entry)
    );
    if (!submission) {
      throw new Error(
        'Cannot resolve income choices before all are submitted.'
      );
    }
    if (!choice.suits.includes(submission.suit)) {
      throw new Error(
        `Suit ${submission.suit} is not valid for submitted income choice.`
      );
    }

    const players = next.players.map((player) =>
      player.id === choice.playerId
        ? {
            ...player,
            resources: applyDelta(player.resources, {
              [submission.suit]: 1,
            }),
          }
        : player
    );
    next = { ...next, players };
  });

  return next;
}

function incomeChoiceMatches(
  choice: {
    playerId: PlayerId;
    districtId: string;
    cardId: CardId;
  },
  submission: {
    playerId: PlayerId;
    districtId: string;
    cardId: CardId;
  }
): boolean {
  return (
    choice.playerId === submission.playerId &&
    choice.districtId === submission.districtId &&
    choice.cardId === submission.cardId
  );
}

function applyTaxation(state: GameState, taxSuit: Suit): GameState['players'] {
  return state.players.map((player) => ({
    ...player,
    resources: {
      ...player.resources,
      [taxSuit]: Math.min(player.resources[taxSuit], 1),
    },
  }));
}

function incomeDeltaForPlayer(
  state: GameState,
  playerId: PlayerId,
  roll: IncomeRollResult
): {
  delta: Partial<Record<Suit, number>>;
  pendingChoices: IncomeChoice[];
} {
  const result = Math.max(roll.die1, roll.die2) as Rank;
  const income = incomeForResult(state, playerId, result);
  return {
    delta: income.fixedDelta,
    pendingChoices: [...income.pendingChoices],
  };
}

interface RollResult {
  value: number;
  rngCursor: number;
}

function rollDie(seed: string, rngCursor: number, sides: number): RollResult {
  const rand = rngFromSeed(`${seed}:roll:${rngCursor}`);
  return {
    value: Math.floor(rand() * sides) + 1,
    rngCursor: rngCursor + 1,
  };
}

function resolveDrawPhase(state: GameState): GameState {
  const previousReshuffles = state.deck.reshuffles;
  const draw = drawOne({
    deck: state.deck,
    seed: state.seed,
    rngCursor: state.rngCursor,
    finalTurnsRemaining: state.finalTurnsRemaining,
  });

  const players = state.players.map((player, index) => {
    if (index !== state.activePlayerIndex || !draw.cardId) {
      return player;
    }
    return {
      ...player,
      hand: [...player.hand, draw.cardId],
    };
  });

  const withDraw = {
    ...state,
    deck: draw.deck,
    players,
    rngCursor: draw.rngCursor,
    finalTurnsRemaining: draw.finalTurnsRemaining,
  };

  const justEnteredFinalTurns =
    previousReshuffles !== 2 && withDraw.deck.reshuffles === 2;

  return endTurn(withDraw, justEnteredFinalTurns);
}

function endTurn(state: GameState, justEnteredFinalTurns = false): GameState {
  const nextPlayerIndex = (state.activePlayerIndex + 1) % state.players.length;
  const nextTurn = state.turn + 1;
  const handoff = handoffTurnState(state, nextPlayerIndex, nextTurn);

  if (state.deck.reshuffles === 2) {
    if (justEnteredFinalTurns) {
      return {
        ...handoff,
        phase: 'StartTurn',
        finalTurnsRemaining: state.finalTurnsRemaining ?? 2,
      };
    }

    const turnsLeft = state.finalTurnsRemaining ?? 0;
    const remaining = Math.max(turnsLeft - 1, 0);

    if (remaining === 0) {
      return finalizeGame({
        ...handoff,
        phase: 'GameOver',
        finalTurnsRemaining: 0,
      });
    }

    return {
      ...handoff,
      phase: 'StartTurn',
      finalTurnsRemaining: remaining,
    };
  }

  return {
    ...handoff,
    phase: 'StartTurn',
    finalTurnsRemaining: undefined,
  };
}

function finalizeGame(state: GameState): GameState {
  const discardedCards: CardId[] = [];

  const players = state.players.map((player) => {
    discardedCards.push(...player.hand);
    return {
      ...player,
      hand: [],
    };
  });

  const terminalState: GameState = {
    ...state,
    players,
    deck: {
      ...state.deck,
      discard: [...discardedCards, ...state.deck.discard],
    },
  };

  return {
    ...terminalState,
    finalScore: scoreGame(terminalState),
  };
}

function findPlayerIndexById(state: GameState, playerId: PlayerId): number {
  const index = state.players.findIndex((player) => player.id === playerId);
  if (index < 0) {
    throw new Error(`Unknown player: ${playerId}`);
  }
  return index;
}

function handoffTurnState(
  state: GameState,
  activePlayerIndex: number,
  turn: number
): GameState {
  return {
    ...state,
    activePlayerIndex,
    turn,
    cardPlayedThisTurn: false,
    lastIncomeRoll: undefined,
    lastTaxSuit: undefined,
    pendingIncomeChoices: undefined,
    submittedIncomeChoices: undefined,
    incomeChoiceReturnPlayerId: undefined,
  };
}
