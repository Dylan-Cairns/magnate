import { drawOne } from './deck';
import type { GamePhase, GameState } from './types';

const DECISION_PHASES: ReadonlySet<GamePhase> = new Set([
  'OptionalTrade',
  'OptionalDevelop',
  'PlayCard',
  'GameOver',
]);

const MAX_ADVANCE_STEPS = 32;

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
      return {
        ...state,
        phase: 'IncomeRoll',
      };
    case 'IncomeRoll':
      return {
        ...state,
        phase: 'CollectIncome',
      };
    case 'CollectIncome':
      return {
        ...state,
        phase: 'OptionalTrade',
      };
    case 'DrawCard':
      return resolveDrawPhase(state);
    case 'OptionalTrade':
    case 'OptionalDevelop':
    case 'PlayCard':
    case 'GameOver':
      return state;
  }
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
