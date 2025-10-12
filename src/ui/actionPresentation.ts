import { CARD_BY_ID, type CardId } from '../engine/cards';
import { SUITS } from '../engine/stateHelpers';
import type { GameAction, Suit } from '../engine/types';

type TradeAction = Extract<GameAction, { type: 'trade' }>;
type BuyDeedAction = Extract<GameAction, { type: 'buy-deed' }>;
type DevelopDeedAction = Extract<GameAction, { type: 'develop-deed' }>;
type DevelopOutrightAction = Extract<GameAction, { type: 'develop-outright' }>;
type NonGroupedAction = Exclude<
  GameAction,
  TradeAction | BuyDeedAction | DevelopDeedAction | DevelopOutrightAction
>;

export type HumanActionListItem =
  | { kind: 'action'; action: NonGroupedAction }
  | { kind: 'trade-group'; give: Suit; options: TradeAction[] }
  | { kind: 'buy-deed-group'; cardId: CardId; options: BuyDeedAction[] }
  | { kind: 'develop-deed-group'; cardId: CardId; districtId: string; options: DevelopDeedAction[] }
  | {
      kind: 'develop-outright-group';
      cardId: CardId;
      payment: Partial<Record<Suit, number>>;
      paymentKey: string;
      options: DevelopOutrightAction[];
    };

export type ActionPickerQuery =
  | {
      kind: 'trade';
      give: Suit;
    }
  | {
      kind: 'district';
      actionType: 'buy-deed' | 'develop-outright';
      cardId: CardId;
      payment?: Partial<Record<Suit, number>>;
      paymentKey?: string;
    }
  | {
      kind: 'deed-payment';
      cardId: CardId;
      districtId: string;
    };

export interface PickerOption {
  id: string;
  label: string;
  action: GameAction;
}

export function buildHumanActionList(actions: readonly GameAction[]): HumanActionListItem[] {
  const result: HumanActionListItem[] = [];
  const tradeGroups = new Map<Suit, { options: TradeAction[] }>();
  const buyDeedGroups = new Map<CardId, { options: BuyDeedAction[] }>();
  const developDeedGroups = new Map<string, { options: DevelopDeedAction[] }>();
  const developOutrightGroups = new Map<string, { options: DevelopOutrightAction[] }>();

  for (const action of actions) {
    if (action.type === 'trade') {
      const existing = tradeGroups.get(action.give);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        tradeGroups.set(action.give, { options });
        result.push({ kind: 'trade-group', give: action.give, options });
      }
      continue;
    }

    if (action.type === 'buy-deed') {
      const existing = buyDeedGroups.get(action.cardId);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        buyDeedGroups.set(action.cardId, { options });
        result.push({ kind: 'buy-deed-group', cardId: action.cardId, options });
      }
      continue;
    }

    if (action.type === 'develop-deed') {
      const groupKey = `${action.cardId}|${action.districtId}`;
      const existing = developDeedGroups.get(groupKey);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        developDeedGroups.set(groupKey, { options });
        result.push({
          kind: 'develop-deed-group',
          cardId: action.cardId,
          districtId: action.districtId,
          options,
        });
      }
      continue;
    }

    if (action.type === 'develop-outright') {
      const paymentKey = paymentSignature(action.payment);
      const groupKey = `${action.cardId}|${paymentKey}`;
      const existing = developOutrightGroups.get(groupKey);

      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        developOutrightGroups.set(groupKey, { options });
        result.push({
          kind: 'develop-outright-group',
          cardId: action.cardId,
          payment: action.payment,
          paymentKey,
          options,
        });
      }
      continue;
    }

    result.push({ kind: 'action', action });
  }

  return result;
}

export function pickerStillLegal(picker: ActionPickerQuery, actions: readonly GameAction[]): boolean {
  if (picker.kind === 'trade') {
    return actions.some(
      (action): action is TradeAction => action.type === 'trade' && action.give === picker.give
    );
  }

  if (picker.kind === 'deed-payment') {
    const options = actions.filter(
      (action): action is DevelopDeedAction =>
        action.type === 'develop-deed' &&
        action.cardId === picker.cardId &&
        action.districtId === picker.districtId
    );
    return options.length > 1;
  }

  if (picker.actionType === 'buy-deed') {
    const options = actions.filter(
      (action): action is BuyDeedAction =>
        action.type === 'buy-deed' && action.cardId === picker.cardId
    );
    return options.length > 1;
  }

  const options = actions.filter(
    (action): action is DevelopOutrightAction =>
      action.type === 'develop-outright' &&
      action.cardId === picker.cardId &&
      paymentSignature(action.payment) === picker.paymentKey
  );
  return options.length > 1;
}

export function buildPickerOptions(
  picker: ActionPickerQuery,
  actions: readonly GameAction[],
  suitEmoji: Record<Suit, string>
): PickerOption[] {
  if (picker.kind === 'trade') {
    return actions
      .filter((action): action is TradeAction => action.type === 'trade' && action.give === picker.give)
      .map((action) => ({
        id: actionStableKey(action),
        label: `${suitEmoji[action.receive]} x1`,
        action,
      }));
  }

  if (picker.kind === 'deed-payment') {
    return actions
      .filter(
        (action): action is DevelopDeedAction =>
          action.type === 'develop-deed' &&
          action.cardId === picker.cardId &&
          action.districtId === picker.districtId
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: formatTokens(action.tokens, suitEmoji),
        action,
      }));
  }

  if (picker.actionType === 'buy-deed') {
    return actions
      .filter(
        (action): action is BuyDeedAction =>
          action.type === 'buy-deed' && action.cardId === picker.cardId
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: action.districtId,
        action,
      }));
  }

  return actions
    .filter(
      (action): action is DevelopOutrightAction =>
        action.type === 'develop-outright' &&
        action.cardId === picker.cardId &&
        paymentSignature(action.payment) === picker.paymentKey
    )
    .map((action) => ({
      id: actionStableKey(action),
      label: action.districtId,
      action,
    }));
}

export function pickerTitle(
  picker: ActionPickerQuery,
  suitEmoji: Record<Suit, string>
): string {
  if (picker.kind === 'trade') {
    return `Trade ${suitEmoji[picker.give]}x3 for`;
  }

  if (picker.kind === 'deed-payment') {
    return `Develop deed ${cardSummary(picker.cardId, suitEmoji)} in ${picker.districtId} with`;
  }

  if (picker.actionType === 'buy-deed') {
    return `Buy deed ${cardSummary(picker.cardId, suitEmoji)} in`;
  }

  return `Develop ${cardSummary(picker.cardId, suitEmoji)} (${formatTokens(
    picker.payment ?? {},
    suitEmoji
  )}) in`;
}

export function describeAction(action: GameAction, suitEmoji: Record<Suit, string>): string {
  switch (action.type) {
    case 'end-turn':
      return 'Draw card and end turn';
    case 'trade':
      return `Trade ${suitEmoji[action.give]}x3 for ${suitEmoji[action.receive]}x1`;
    case 'sell-card':
      return `Sell ${cardSummary(action.cardId, suitEmoji)}`;
    case 'buy-deed':
      return `Buy deed ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId}`;
    case 'develop-deed':
      return `Develop deed ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId} (${formatTokens(
        action.tokens,
        suitEmoji
      )})`;
    case 'develop-outright':
      return `Develop ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId} (${formatTokens(
        action.payment,
        suitEmoji
      )})`;
    case 'choose-income-suit':
      return `Choose ${suitEmoji[action.suit]} income for ${cardSummary(
        action.cardId,
        suitEmoji
      )} in ${action.districtId}`;
  }
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

export function paymentSignature(tokens: Partial<Record<Suit, number>>): string {
  return SUITS.map((suit) => `${suit}:${tokens[suit] ?? 0}`).join('|');
}

export function formatTokens(
  tokens: Partial<Record<Suit, number>>,
  suitEmoji: Record<Suit, string>
): string {
  const entries = tokenEntries(tokens);
  if (entries.length === 0) {
    return '-';
  }
  return entries.map((entry) => `${suitEmoji[entry.suit]}x${entry.count}`).join(' ');
}

export function cardSummary(cardId: CardId, suitEmoji: Record<Suit, string>): string {
  const card = CARD_BY_ID[cardId];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const suits = card.kind === 'Excuse' ? '' : card.suits.map((suit) => suitEmoji[suit]).join('');
  return `${rank}${suits}`;
}

function tokenEntries(tokens: Partial<Record<Suit, number>>): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
}
