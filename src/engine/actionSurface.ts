import { legalActions } from './actionBuilders';
import { SUITS } from './stateHelpers';
import type { ActionId, GameAction, GameState, Suit } from './types';

export const ACTION_IDS: readonly ActionId[] = [
  'buy-deed',
  'choose-income-suit',
  'develop-deed',
  'develop-outright',
  'end-turn',
  'sell-card',
  'trade',
];

export interface KeyedAction {
  actionId: ActionId;
  actionKey: string;
  action: GameAction;
}

export function paymentSignature(tokens: Partial<Record<Suit, number>>): string {
  return SUITS.map((suit) => `${suit}:${tokens[suit] ?? 0}`).join('|');
}

export function actionStableKey(action: GameAction): string {
  switch (action.type) {
    case 'end-turn':
      return 'end-turn';
    case 'trade':
      return `trade:${action.give}:${action.receive}`;
    case 'sell-card':
      return `sell-card:${action.cardId}`;
    case 'buy-deed':
      return `buy-deed:${action.cardId}:${action.districtId}`;
    case 'develop-deed':
      return `develop-deed:${action.cardId}:${action.districtId}:${paymentSignature(action.tokens)}`;
    case 'develop-outright':
      return `develop-outright:${action.cardId}:${action.districtId}:${paymentSignature(action.payment)}`;
    case 'choose-income-suit':
      return `choose-income-suit:${action.playerId}:${action.districtId}:${action.cardId}:${action.suit}`;
  }
}

export function toKeyedActions(actions: readonly GameAction[]): KeyedAction[] {
  return [...actions]
    .map((action) => ({
      actionId: action.type,
      actionKey: actionStableKey(action),
      action,
    }))
    .sort((left, right) =>
      left.actionKey < right.actionKey
        ? -1
        : left.actionKey > right.actionKey
          ? 1
          : 0
    );
}

export function legalActionsCanonical(state: GameState): KeyedAction[] {
  return toKeyedActions(legalActions(state));
}

