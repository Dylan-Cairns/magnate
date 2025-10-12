import { CARD_BY_ID } from './cards';
import type { CardId } from './cards';
import { legalActions } from './actionBuilders';
import { discard } from './deck';
import {
  SUITS,
  applyDelta,
  canAfford,
  deedCost,
  developmentCost,
  findProperty,
  mergeTokens,
  placementAllowed,
  sumTokens,
} from './stateHelpers';
import type {
  GameAction,
  GameState,
  PlayerState,
  DistrictStack,
  DistrictState,
  Suit,
  PlayerId,
  PropertyCard,
} from './types';

export function applyAction(state: GameState, action: GameAction): GameState {
  assertActionIsLegal(state, action);

  switch (action.type) {
    case 'choose-income-suit':
      return chooseIncomeSuit(state, action);
    case 'end-turn':
      return endTurn(state);
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

function assertActionIsLegal(state: GameState, action: GameAction): void {
  const legal = legalActions(state);
  const isLegal = legal.some((candidate) => actionsEqual(candidate, action));
  if (!isLegal) {
    throw new Error(`Illegal action for phase ${state.phase}: ${JSON.stringify(action)}`);
  }
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type) {
    return false;
  }

  if (left.type === 'end-turn') {
    return true;
  }

  if (left.type === 'choose-income-suit' && right.type === 'choose-income-suit') {
    return (
      left.playerId === right.playerId &&
      left.districtId === right.districtId &&
      left.cardId === right.cardId &&
      left.suit === right.suit
    );
  }

  if (left.type === 'trade' && right.type === 'trade') {
    return left.give === right.give && left.receive === right.receive;
  }

  if (left.type === 'sell-card' && right.type === 'sell-card') {
    return left.cardId === right.cardId;
  }

  if (left.type === 'buy-deed' && right.type === 'buy-deed') {
    return left.cardId === right.cardId && left.districtId === right.districtId;
  }

  if (left.type === 'develop-deed' && right.type === 'develop-deed') {
    return (
      left.cardId === right.cardId &&
      left.districtId === right.districtId &&
      sameSuitCounts(left.tokens, right.tokens)
    );
  }

  if (left.type === 'develop-outright' && right.type === 'develop-outright') {
    return (
      left.cardId === right.cardId &&
      left.districtId === right.districtId &&
      sameSuitCounts(left.payment, right.payment)
    );
  }

  return false;
}

function sameSuitCounts(
  left: Partial<Record<Suit, number>>,
  right: Partial<Record<Suit, number>>
): boolean {
  return SUITS.every((suit) => (left[suit] ?? 0) === (right[suit] ?? 0));
}

function endTurn(state: GameState): GameState {
  if (!state.cardPlayedThisTurn) {
    throw new Error('Cannot end turn before a card has been played.');
  }
  return log({ ...state, phase: 'DrawCard' }, 'end turn');
}

function chooseIncomeSuit(
  state: GameState,
  action: Extract<GameAction, { type: 'choose-income-suit' }>
): GameState {
  const activePlayer = state.players[state.activePlayerIndex];
  if (!activePlayer || activePlayer.id !== action.playerId) {
    throw new Error(
      `Income choice must be made by active player ${activePlayer?.id ?? 'unknown'}.`
    );
  }

  const [nextChoice, ...restChoices] = state.pendingIncomeChoices ?? [];
  if (!nextChoice) {
    throw new Error('No pending income choices available.');
  }
  if (
    action.playerId !== nextChoice.playerId ||
    action.districtId !== nextChoice.districtId ||
    action.cardId !== nextChoice.cardId
  ) {
    throw new Error('Income choice does not match the next pending income choice.');
  }
  if (!nextChoice.suits.includes(action.suit)) {
    throw new Error(`Suit ${action.suit} is not valid for this income choice.`);
  }

  const player = findPlayerById(state, action.playerId);
  const updatedPlayer: PlayerState = {
    ...player,
    resources: applyDelta(player.resources, { [action.suit]: 1 }),
  };

  const pendingIncomeChoices = restChoices.length > 0 ? restChoices : undefined;
  const phase = pendingIncomeChoices ? 'CollectIncome' : 'ActionWindow';
  const returnPlayerId = state.incomeChoiceReturnPlayerId ?? state.players[state.activePlayerIndex]?.id;
  if (!returnPlayerId) {
    throw new Error('Missing return player while resolving income choice.');
  }
  const nextActivePlayerIndex = pendingIncomeChoices
    ? findPlayerIndexById(state, pendingIncomeChoices[0].playerId)
    : findPlayerIndexById(state, returnPlayerId);

  const next = replacePlayerById(state, action.playerId, updatedPlayer);
  return log(
    {
      ...next,
      activePlayerIndex: nextActivePlayerIndex,
      phase,
      pendingIncomeChoices,
      incomeChoiceReturnPlayerId: pendingIncomeChoices
        ? state.incomeChoiceReturnPlayerId
        : undefined,
    },
    `income choice ${action.cardId}:${action.suit}`,
    action.playerId
  );
}

function trade(
  state: GameState,
  action: Extract<GameAction, { type: 'trade' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  if (action.give === action.receive) {
    throw new Error('Trade must exchange different suits.');
  }
  if (player.resources[action.give] < 3) {
    throw new Error(`Cannot trade ${action.give}; requires at least 3 tokens.`);
  }

  const delta = { [action.give]: -3, [action.receive]: 1 };
  const updated = replaceActivePlayer(state, {
    ...player,
    resources: applyDelta(player.resources, delta),
  });

  return log(updated, `trade ${action.give} for ${action.receive}`);
}

function developDeed(
  state: GameState,
  action: Extract<GameAction, { type: 'develop-deed' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = getPropertyCard(action.cardId);
  const district = findDistrictById(state, action.districtId);
  const stack = district.stacks[player.id];
  if (!stack?.deed) {
    throw new Error('No deed exists in this district for the active player.');
  }
  if (stack.deed.cardId !== action.cardId) {
    throw new Error('Develop deed action card does not match district deed card.');
  }

  const spend = validateSuitSpend(action.tokens, card.suits, 'deed development');
  if (!canAfford(player.resources, spend)) {
    throw new Error('Player cannot afford deed development spend.');
  }

  const target = developmentCost(card);
  const nextProgress = stack.deed.progress + sumTokens(spend);
  if (nextProgress > target) {
    throw new Error('Deed development cannot exceed its completion cost.');
  }

  const resources = applyDelta(player.resources, negate(spend));
  const districts = updateDistricts(state, action.districtId, player.id, (current) => {
    const existingDeed = current.deed;
    if (!existingDeed || existingDeed.cardId !== action.cardId) {
      throw new Error('Invalid deed state during development.');
    }

    if (nextProgress === target) {
      return {
        developed: [...current.developed, action.cardId],
      };
    }

    return {
      developed: current.developed,
      deed: {
        cardId: action.cardId,
        progress: nextProgress,
        tokens: mergeTokens(existingDeed.tokens, spend),
      },
    };
  });

  const updated = replaceActivePlayer(state, { ...player, resources });
  return log({ ...updated, districts }, `advance ${action.cardId}`);
}

function developOutright(
  state: GameState,
  action: Extract<GameAction, { type: 'develop-outright' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = getPropertyCard(action.cardId);

  assertPlayerHasCard(player, action.cardId);
  const district = findDistrictById(state, action.districtId);
  if (!placementAllowed(card, district, player.id)) {
    throw new Error('Outright development placement is not allowed in this district.');
  }

  const payment = validateSuitSpend(action.payment, card.suits, 'outright development');
  const totalPayment = sumTokens(payment);
  const required = developmentCost(card);
  if (totalPayment !== required) {
    throw new Error(`Outright development requires exactly ${required} total tokens.`);
  }
  if (!card.suits.every((suit) => (payment[suit] ?? 0) >= 1)) {
    throw new Error('Outright development payment must include at least one token of each card suit.');
  }
  if (!canAfford(player.resources, payment)) {
    throw new Error('Player cannot afford outright development payment.');
  }

  const resources = applyDelta(player.resources, negate(payment));
  const hand = player.hand.filter((id) => id !== action.cardId);
  const districts = updateDistricts(state, action.districtId, player.id, (stack) => ({
    developed: [...stack.developed, action.cardId],
  }));

  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  return log(
    {
      ...updated,
      districts,
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
    },
    `develop ${action.cardId}`
  );
}

function buyDeed(
  state: GameState,
  action: Extract<GameAction, { type: 'buy-deed' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  const card = getPropertyCard(action.cardId);

  assertPlayerHasCard(player, action.cardId);
  const district = findDistrictById(state, action.districtId);
  if (!placementAllowed(card, district, player.id)) {
    throw new Error('Deed placement is not allowed in this district.');
  }

  const cost = deedCost(card);
  if (!canAfford(player.resources, cost)) {
    throw new Error('Player cannot afford deed cost.');
  }

  const resources = applyDelta(player.resources, negate(cost));
  const hand = player.hand.filter((id) => id !== action.cardId);
  const districts = updateDistricts(state, action.districtId, player.id, (stack) => ({
    developed: stack.developed,
    deed: { cardId: action.cardId, progress: 0, tokens: {} },
  }));

  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  return log(
    {
      ...updated,
      districts,
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
    },
    `buy deed ${action.cardId}`
  );
}

function sellCard(
  state: GameState,
  action: Extract<GameAction, { type: 'sell-card' }>
): GameState {
  const player = state.players[state.activePlayerIndex];
  assertPlayerHasCard(player, action.cardId);

  const card = CARD_BY_ID[action.cardId];
  if (card.kind !== 'Property') {
    throw new Error('Only property cards can be sold from hand.');
  }

  const gain = gainFor(card.suits);
  const resources = applyDelta(player.resources, gain);
  const hand = player.hand.filter((id) => id !== action.cardId);
  const updated = replaceActivePlayer(state, { ...player, hand, resources });
  const deck = discard(state.deck, action.cardId);

  return log(
    {
      ...updated,
      deck,
      phase: 'ActionWindow',
      cardPlayedThisTurn: true,
    },
    `sell ${action.cardId}`
  );
}

function validateSuitSpend(
  tokens: Partial<Record<Suit, number>>,
  allowedSuits: readonly Suit[],
  label: string
): Partial<Record<Suit, number>> {
  const next: Partial<Record<Suit, number>> = {};
  const allowed = new Set(allowedSuits);
  let total = 0;

  SUITS.forEach((suit) => {
    const value = tokens[suit] ?? 0;
    if (value === 0) {
      return;
    }
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${label} tokens must be positive integers.`);
    }
    if (!allowed.has(suit)) {
      throw new Error(`${label} includes suit ${suit} which is not on the card.`);
    }
    next[suit] = value;
    total += value;
  });

  if (total < 1) {
    throw new Error(`${label} requires at least one token.`);
  }

  return next;
}

function getPropertyCard(cardId: CardId): PropertyCard {
  const card = findProperty(cardId);
  if (!card) {
    throw new Error(`Card ${cardId} is not a property card.`);
  }
  return card;
}

function assertPlayerHasCard(player: PlayerState, cardId: CardId): void {
  if (!player.hand.includes(cardId)) {
    throw new Error(`Card ${cardId} is not in the active player's hand.`);
  }
}

function findDistrictById(state: GameState, districtId: string): DistrictState {
  const district = state.districts.find((item) => item.id === districtId);
  if (!district) {
    throw new Error(`Unknown district: ${districtId}`);
  }
  return district;
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

function replacePlayerById(
  state: GameState,
  playerId: PlayerId,
  updated: PlayerState
): GameState {
  const players = state.players.map((player) =>
    player.id === playerId ? updated : player
  );
  return { ...state, players };
}

function findPlayerById(state: GameState, playerId: PlayerId): PlayerState {
  const player = state.players.find((entry) => entry.id === playerId);
  if (!player) {
    throw new Error(`Unknown player: ${playerId}`);
  }
  return player;
}

function findPlayerIndexById(state: GameState, playerId: PlayerId): number {
  const index = state.players.findIndex((entry) => entry.id === playerId);
  if (index < 0) {
    throw new Error(`Unknown player: ${playerId}`);
  }
  return index;
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

function log(state: GameState, summary: string, playerId?: PlayerId): GameState {
  const entryPlayerId = playerId ?? state.players[state.activePlayerIndex]?.id;
  if (!entryPlayerId) {
    throw new Error('Cannot append log entry without an active player.');
  }

  const entry = {
    turn: state.turn,
    player: entryPlayerId,
    phase: state.phase,
    summary,
  };
  return { ...state, log: [...state.log, entry] };
}
