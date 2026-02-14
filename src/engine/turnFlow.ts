import { CARD_BY_ID } from './cards';
import type { CardId } from './cards';
import { legalActions } from './actionBuilders';
import { drawOne } from './deck';
import { rngFromSeed } from './rng';
import { scoreGame } from './scoring';
import { applyDelta, findProperty } from './stateHelpers';
import type {
  DistrictStack,
  GamePhase,
  GameState,
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  Suit,
} from './types';

const BASE_DECISION_PHASES: ReadonlySet<GamePhase> = new Set(['PlayCard', 'GameOver']);

const MAX_ADVANCE_STEPS = 32;
const TAX_SUIT_BY_D6: readonly [Suit, Suit, Suit, Suit, Suit, Suit] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

export function advanceToDecision(state: GameState): GameState {
  let current = state;

  for (let i = 0; i < MAX_ADVANCE_STEPS; i += 1) {
    if (isDecisionPhase(current)) {
      return current;
    }

    current = advanceOnePhase(current);
  }

  throw new Error(
    `advanceToDecision exceeded ${MAX_ADVANCE_STEPS} phase transitions.`
  );
}

function isDecisionPhase(state: GameState): boolean {
  if (state.phase === 'OptionalTrade' || state.phase === 'OptionalDevelop') {
    return legalActions(state).length > 0;
  }
  if (BASE_DECISION_PHASES.has(state.phase)) {
    return true;
  }
  if (state.phase === 'CollectIncome') {
    return (state.pendingIncomeChoices?.length ?? 0) > 0;
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
    case 'IncomeRoll':
      return resolveIncomeRoll(state);
    case 'CollectIncome':
      return resolveCollectIncome(state);
    case 'DrawCard':
      return resolveDrawPhase(state);
    case 'OptionalTrade':
      return {
        ...state,
        phase: 'OptionalDevelop',
      };
    case 'OptionalDevelop':
      return {
        ...state,
        phase: state.cardPlayedThisTurn ? 'OptionalTrade' : 'PlayCard',
      };
    case 'PlayCard':
    case 'GameOver':
      return state;
  }
}

function resolveTaxCheck(state: GameState): GameState {
  const first = rollDie(state.seed, state.rngCursor, 10);
  const second = rollDie(state.seed, first.rngCursor, 10);
  const incomeRoll: IncomeRollResult = { die1: first.value, die2: second.value };

  let nextState: GameState = {
    ...state,
    rngCursor: second.rngCursor,
    lastIncomeRoll: incomeRoll,
  };

  if (incomeRoll.die1 === 1 || incomeRoll.die2 === 1) {
    const taxSuitRoll = rollDie(state.seed, second.rngCursor, 6);
    nextState = {
      ...nextState,
      rngCursor: taxSuitRoll.rngCursor,
      players: applyTaxation(nextState, taxSuitRoll.value),
    };
  }

  return {
    ...nextState,
    phase: 'IncomeRoll',
  };
}

function resolveIncomeRoll(state: GameState): GameState {
  if (!state.lastIncomeRoll) {
    throw new Error('IncomeRoll phase requires lastIncomeRoll to be present.');
  }

  return {
    ...state,
    phase: 'CollectIncome',
  };
}

function resolveCollectIncome(state: GameState): GameState {
  if (!state.lastIncomeRoll) {
    throw new Error('CollectIncome phase requires lastIncomeRoll to be present.');
  }
  if ((state.pendingIncomeChoices?.length ?? 0) > 0) {
    const [nextChoice] = state.pendingIncomeChoices ?? [];
    if (!nextChoice) {
      return state;
    }
    const nextActivePlayerIndex = findPlayerIndexById(state, nextChoice.playerId);
    if (nextActivePlayerIndex === state.activePlayerIndex) {
      return state;
    }
    return {
      ...state,
      activePlayerIndex: nextActivePlayerIndex,
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
    const [nextChoice] = pendingChoices;
    const nextActivePlayerIndex = findPlayerIndexById(state, nextChoice.playerId);
    return {
      ...state,
      players,
      activePlayerIndex: nextActivePlayerIndex,
      phase: 'CollectIncome',
      pendingIncomeChoices: pendingChoices,
      incomeChoiceReturnPlayerIndex: state.activePlayerIndex,
    };
  }

  return {
    ...state,
    players,
    phase: 'OptionalTrade',
    cardPlayedThisTurn: false,
    pendingIncomeChoices: undefined,
    incomeChoiceReturnPlayerIndex: undefined,
  };
}

function applyTaxation(state: GameState, d6: number): GameState['players'] {
  const taxSuit = TAX_SUIT_BY_D6[d6 - 1];

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
  const result = Math.max(roll.die1, roll.die2);
  const delta: Partial<Record<Suit, number>> = {};
  const pendingChoices: IncomeChoice[] = [];

  if (result === 10) {
    awardCrownIncome(state, playerId, delta);
    return { delta, pendingChoices };
  }

  if (result === 1) {
    awardAceIncome(state, playerId, delta);
    return { delta, pendingChoices };
  }

  awardRankIncome(state, playerId, result, delta, pendingChoices);
  return { delta, pendingChoices };
}

function awardCrownIncome(
  state: GameState,
  playerId: PlayerId,
  delta: Partial<Record<Suit, number>>
): void {
  const player = state.players.find((item) => item.id === playerId);
  if (!player) {
    return;
  }

  player.crowns.forEach((cardId) => {
    const card = CARD_BY_ID[cardId];
    if (card.kind !== 'Crown') {
      return;
    }
    addSuit(delta, card.suits[0], 1);
  });
}

function awardAceIncome(
  state: GameState,
  playerId: PlayerId,
  delta: Partial<Record<Suit, number>>
): void {
  state.districts.forEach((district) => {
    const stack = district.stacks[playerId];
    stack.developed.forEach((cardId) => {
      const property = findProperty(cardId);
      if (!property || property.rank !== 1) {
        return;
      }
      property.suits.forEach((suit) => addSuit(delta, suit, 1));
    });

    const deed = stack.deed ? findProperty(stack.deed.cardId) : undefined;
    if (deed && deed.rank === 1) {
      addSuit(delta, deed.suits[0], 1);
    }
  });
}

function awardRankIncome(
  state: GameState,
  playerId: PlayerId,
  rank: number,
  delta: Partial<Record<Suit, number>>,
  pendingChoices: IncomeChoice[]
): void {
  state.districts.forEach((district) => {
    const stack = district.stacks[playerId];
    stack.developed.forEach((cardId) => {
      const property = findProperty(cardId);
      if (!property || property.rank !== rank) {
        return;
      }
      property.suits.forEach((suit) => addSuit(delta, suit, 1));
    });

    const deed = stack.deed ? findProperty(stack.deed.cardId) : undefined;
    if (deed && deed.rank === rank) {
      if (deed.suits.length === 1) {
        addSuit(delta, deed.suits[0], 1);
      } else {
        pendingChoices.push({
          playerId,
          districtId: district.id,
          cardId: deed.id,
          suits: deed.suits,
        });
      }
    }
  });
}

function addSuit(
  delta: Partial<Record<Suit, number>>,
  suit: Suit,
  amount: number
): void {
  delta[suit] = (delta[suit] ?? 0) + amount;
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
  const previousExhaustionStage = state.exhaustionStage;
  const draw = drawOne({
    deck: state.deck,
    seed: state.seed,
    rngCursor: state.rngCursor,
    exhaustionStage: state.exhaustionStage,
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
    exhaustionStage: draw.exhaustionStage,
    finalTurnsRemaining: draw.finalTurnsRemaining,
  };

  const justEnteredFinalTurns =
    previousExhaustionStage !== 2 && withDraw.exhaustionStage === 2;

  return endTurn(withDraw, justEnteredFinalTurns);
}

function endTurn(state: GameState, justEnteredFinalTurns = false): GameState {
  const nextPlayerIndex = (state.activePlayerIndex + 1) % state.players.length;
  const nextTurn = state.turn + 1;

  if (state.exhaustionStage === 2) {
    if (justEnteredFinalTurns) {
      return {
        ...state,
        activePlayerIndex: nextPlayerIndex,
        turn: nextTurn,
        phase: 'StartTurn',
        cardPlayedThisTurn: false,
        finalTurnsRemaining: state.finalTurnsRemaining ?? 2,
        lastIncomeRoll: undefined,
        pendingIncomeChoices: undefined,
        incomeChoiceReturnPlayerIndex: undefined,
      };
    }

    const turnsLeft = state.finalTurnsRemaining ?? 0;
    const remaining = Math.max(turnsLeft - 1, 0);

    if (remaining === 0) {
      return finalizeGame({
        ...state,
        activePlayerIndex: nextPlayerIndex,
        turn: nextTurn,
        phase: 'GameOver',
        cardPlayedThisTurn: false,
        finalTurnsRemaining: 0,
        lastIncomeRoll: undefined,
        pendingIncomeChoices: undefined,
        incomeChoiceReturnPlayerIndex: undefined,
      });
    }

    return {
      ...state,
      activePlayerIndex: nextPlayerIndex,
      turn: nextTurn,
      phase: 'StartTurn',
      cardPlayedThisTurn: false,
      finalTurnsRemaining: remaining,
      lastIncomeRoll: undefined,
      pendingIncomeChoices: undefined,
      incomeChoiceReturnPlayerIndex: undefined,
    };
  }

  return {
    ...state,
    activePlayerIndex: nextPlayerIndex,
    turn: nextTurn,
    phase: 'StartTurn',
    cardPlayedThisTurn: false,
    lastIncomeRoll: undefined,
    pendingIncomeChoices: undefined,
    incomeChoiceReturnPlayerIndex: undefined,
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

  const districts = state.districts.map((district) => ({
    ...district,
    stacks: {
      ...district.stacks,
      PlayerA: clearIncompleteDeed(district.stacks.PlayerA, discardedCards),
      PlayerB: clearIncompleteDeed(district.stacks.PlayerB, discardedCards),
    },
  }));

  const terminalState: GameState = {
    ...state,
    players,
    districts,
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

function clearIncompleteDeed(
  stack: DistrictStack,
  discardedCards: CardId[]
): DistrictStack {
  if (stack.deed) {
    discardedCards.push(stack.deed.cardId);
  }
  return {
    developed: [...stack.developed],
  };
}

function findPlayerIndexById(state: GameState, playerId: PlayerId): number {
  const index = state.players.findIndex((player) => player.id === playerId);
  if (index < 0) {
    throw new Error(`Unknown player: ${playerId}`);
  }
  return index;
}
