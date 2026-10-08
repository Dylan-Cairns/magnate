import { legalActions } from './actionBuilders';
import { SUITS } from './stateHelpers';
import { ActionId, type GameAction, type GameState, type Suit } from './types';

export { ACTION_IDS } from './values';

export interface KeyedAction {
  actionId: ActionId;
  actionKey: string;
  action: GameAction;
}

export function paymentSignature(
  tokens: Partial<Record<Suit, number>>
): string {
  return SUITS.map((suit) => `${suit}:${tokens[suit] ?? 0}`).join('|');
}

export function actionStableKey(action: GameAction): string {
  switch (action.type) {
    case ActionId.EndTurn:
      return ActionId.EndTurn;
    case ActionId.Trade:
      return `${ActionId.Trade}:${action.give}:${action.receive}`;
    case ActionId.SellCard:
      return `${ActionId.SellCard}:${action.cardId}`;
    case ActionId.BuyDeed:
      return `${ActionId.BuyDeed}:${action.cardId}:${action.districtId}`;
    case ActionId.DevelopDeed:
      return `${ActionId.DevelopDeed}:${action.cardId}:${action.districtId}:${paymentSignature(action.tokens)}`;
    case ActionId.DevelopOutright:
      return `${ActionId.DevelopOutright}:${action.cardId}:${action.districtId}:${paymentSignature(action.payment)}`;
    case ActionId.ChooseIncomeSuit:
      return `${ActionId.ChooseIncomeSuit}:${action.playerId}:${action.districtId}:${action.cardId}:${action.suit}`;
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
