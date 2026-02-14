import { CARD_BY_ID } from './cards';
import { drawOne } from './deck';
import { rngFromSeed } from './rng';
import { applyDelta, findProperty } from './stateHelpers';
import type { GamePhase, GameState, IncomeRollResult, PlayerId, Suit } from './types';

const DECISION_PHASES: ReadonlySet<GamePhase> = new Set([
  'OptionalTrade',
  'OptionalDevelop',
  'PlayCard',
  'GameOver',
]);

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
    if (DECISION_PHASES.has(current.phase)) {
      return current;
    }

    current = advanceOnePhase(current);
  }

  throw new Error(
    `advanceToDecision exceeded ${MAX_ADVANCE_STEPS} phase transitions.`
  );
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
    case 'OptionalDevelop':
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
  const incomeRoll = state.lastIncomeRoll;

  const players = state.players.map((player) => {
    const delta = incomeDeltaForPlayer(state, player.id, incomeRoll);
    return {
      ...player,
      resources: applyDelta(player.resources, delta),
    };
  });

  return {
    ...state,
    players,
    phase: 'OptionalTrade',
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
): Partial<Record<Suit, number>> {
  const result = Math.max(roll.die1, roll.die2);
  const delta: Partial<Record<Suit, number>> = {};

  if (result === 10) {
    awardCrownIncome(state, playerId, delta);
    return delta;
  }

  if (result === 1) {
    awardAceIncome(state, playerId, delta);
    return delta;
  }

  awardRankIncome(state, playerId, result, delta);
  return delta;
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
  delta: Partial<Record<Suit, number>>
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
      // Rules allow choosing deed suit at income time; until that choice is explicit,
      // use first suit deterministically to keep engine behavior reproducible.
      addSuit(delta, deed.suits[0], 1);
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

  return endTurn(withDraw);
}

function endTurn(state: GameState): GameState {
  const nextPlayerIndex = (state.activePlayerIndex + 1) % state.players.length;
  const nextTurn = state.turn + 1;

  if (state.exhaustionStage === 2) {
    const turnsLeft = state.finalTurnsRemaining ?? 0;
    const remaining = Math.max(turnsLeft - 1, 0);

    if (remaining === 0) {
      return {
        ...state,
        activePlayerIndex: nextPlayerIndex,
        turn: nextTurn,
        phase: 'GameOver',
        finalTurnsRemaining: 0,
      };
    }

    return {
      ...state,
      activePlayerIndex: nextPlayerIndex,
      turn: nextTurn,
      phase: 'StartTurn',
      finalTurnsRemaining: remaining,
    };
  }

  return {
    ...state,
    activePlayerIndex: nextPlayerIndex,
    turn: nextTurn,
    phase: 'StartTurn',
  };
}
