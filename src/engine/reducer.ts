import { CARD_BY_ID } from './cards';
import { discard } from './deck';
import {
  SUITS,
  applyDelta,
  deedCost,
  developmentCost,
  findProperty,
  mergeTokens,
  sumTokens,
} from './stateHelpers';
import type {
  GameAction,
  GameState,
  PlayerState,
  DistrictStack,
  Suit,
  PlayerId,
} from './types';

export function applyAction(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'trade':
      return trade(state, action);
    case 'develop-deed':
      return developDeed(state, action);
    case 'develop-outright':
      return developOutright(state, action);
    case 'buy-deed':
      return buyDeed(state, action);
    case 'sell-card':
      return sellCard(state, action);
  }
  return state;
}

function trade(state: GameState, action: Extract<GameAction, { type: 'trade' }>): GameState {
  const player = state.players[state.activePlayerIndex];
  const delta = { [action.give]: -3, [action.receive]: 1 };
  const updated = replaceActivePlayer(state, { ...player, resources: applyDelta(player.resources, delta) });
  return log(updated, 'trade ' + action.give + ' for ' + action.receive);
}

function developDeed(
  state: GameState,
  action: Extract<GameAction, { type: 'develop-deed' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = findProperty(action.cardId)!;
  const payment = applyDelta(player.resources, negate(action.tokens));
  const districts = updateDistricts(state, action.districtId, player.id, (stack) => {
    const progress = (stack.deed?.progress ?? 0) + sumTokens(action.tokens);
    if (progress >= developmentCost(card)) {
      return {
        developed: [...stack.developed, action.cardId],
      };
    }
    return {
      developed: stack.developed,
      deed: {
        cardId: action.cardId,
        progress,
        tokens: mergeTokens(stack.deed?.tokens, action.tokens),
      },
    };
  });
  const updated = replaceActivePlayer(state, { ...player, resources: payment });
  return log({ ...updated, districts }, 'advance ' + action.cardId);
}

function developOutright(
  state: GameState,
  action: Extract<GameAction, { type: 'develop-outright' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  const resources = applyDelta(player.resources, negate(action.payment));
  const hand = player.hand.filter((id) => id !== action.cardId);
  const districts = updateDistricts(state, action.districtId, player.id, (stack) => ({
    developed: [...stack.developed, action.cardId],
  }));
  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  return log({ ...updated, districts, phase: 'DrawCard' }, 'develop ' + action.cardId);
}

function buyDeed(state: GameState, action: Extract<GameAction, { type: 'buy-deed' }>): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = findProperty(action.cardId)!;
  const cost = deedCost(card);
  const resources = applyDelta(player.resources, negate(cost));
  const hand = player.hand.filter((id) => id !== action.cardId);
  const districts = updateDistricts(state, action.districtId, player.id, (stack) => ({
    developed: stack.developed,
    deed: { cardId: action.cardId, progress: 0, tokens: {} },
  }));
  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  return log({ ...updated, districts, phase: 'DrawCard' }, 'buy deed ' + action.cardId);
}

function sellCard(state: GameState, action: Extract<GameAction, { type: 'sell-card' }>): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = CARD_BY_ID[action.cardId]!;
  const gain = gainFor(card.kind === 'Property' ? card.suits : []);
  const resources = applyDelta(player.resources, gain);
  const hand = player.hand.filter((id) => id !== action.cardId);
  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  const deck = discard(state.deck, action.cardId);
  return log({ ...updated, deck, phase: 'DrawCard' }, 'sell ' + action.cardId);
}

function gainFor(suits: readonly Suit[]): Partial<Record<Suit, number>> {
  if (suits.length === 0) {
    return {};
  }
  if (suits.length === 1) {
    return { [suits[0]]: 2 };
  }
  return suits.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
    acc[suit] = (acc[suit] ?? 0) + 1;
    return acc;
  }, {});
}

function replaceActivePlayer(state: GameState, updated: PlayerState): GameState {
  const players = state.players.map((player, index) =>
    index === state.activePlayerIndex ? updated : player
  );
  return { ...state, players };
}

function updateDistricts(
  state: GameState,
  districtId: string,
  playerId: PlayerId,
  create: (stack: DistrictStack) => DistrictStack
): GameState['districts'] {
  return state.districts.map((district) => {
    if (district.id !== districtId) {
      return district;
    }
    const current = district.stacks[playerId] ?? { developed: [] };
    return {
      ...district,
      stacks: {
        ...district.stacks,
        [playerId]: create(current),
      },
    };
  });
}

function negate(tokens: Partial<Record<Suit, number>>): Partial<Record<Suit, number>> {
  const result: Partial<Record<Suit, number>> = {};
  SUITS.forEach((suit) => {
    if (tokens[suit]) {
      result[suit] = -(tokens[suit] ?? 0);
    }
  });
  return result;
}

function log(state: GameState, summary: string): GameState {
  const entry = {
    turn: state.turn,
    player: state.players[state.activePlayerIndex].id,
    phase: state.phase,
    summary,
  };
  return { ...state, log: [...state.log, entry] };
}
