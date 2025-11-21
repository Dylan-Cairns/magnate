import { CARD_BY_ID, type CardId } from '../engine/cards';
import { actionStableKey, paymentSignature } from '../engine/actionSurface';
import { SUITS } from '../engine/stateHelpers';
import type { GameAction, Suit } from '../engine/types';

export { actionStableKey, paymentSignature };

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
  | {
      kind: 'develop-deed-group';
      cardId: CardId;
      districtId: string;
      options: DevelopDeedAction[];
    }
  | {
      kind: 'develop-outright-group';
      cardId: CardId;
      options: DevelopOutrightAction[];
    };

export type ActionPickerQuery =
  | {
      kind: 'trade';
      give: Suit;
    }
  | {
      kind: 'district';
      actionType: 'buy-deed';
      cardId: CardId;
    }
  | {
      kind: 'develop-outright-district';
      cardId: CardId;
    }
  | {
      kind: 'develop-outright-payment';
      cardId: CardId;
      districtId: string;
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

export interface TradeSourceGroup {
  give: Suit;
  options: TradeAction[];
}

export function buildTradeSourceGroups(
  actions: readonly GameAction[]
): TradeSourceGroup[] {
  const groups: TradeSourceGroup[] = [];
  const byGive = new Map<Suit, TradeAction[]>();

  for (const action of actions) {
    if (action.type !== 'trade') {
      continue;
    }

    const existing = byGive.get(action.give);
    if (existing) {
      existing.push(action);
      continue;
    }

    const options = [action];
    byGive.set(action.give, options);
    groups.push({ give: action.give, options });
  }

  return groups;
}

export function buildHumanActionList(
  actions: readonly GameAction[]
): HumanActionListItem[] {
  const tradeItems: Extract<HumanActionListItem, { kind: 'trade-group' }>[] = [];
  const buyDeedItems: Extract<
    HumanActionListItem,
    { kind: 'buy-deed-group' }
  >[] = [];
  const developDeedItems: Extract<
    HumanActionListItem,
    { kind: 'develop-deed-group' }
  >[] = [];
  const developOutrightItems: Extract<
    HumanActionListItem,
    { kind: 'develop-outright-group' }
  >[] = [];
  const nonGroupedByType = new Map<NonGroupedAction['type'], NonGroupedAction[]>();
  const tradeGroups = new Map<Suit, { options: TradeAction[] }>();
  const buyDeedGroups = new Map<CardId, { options: BuyDeedAction[] }>();
  const developDeedGroups = new Map<string, { options: DevelopDeedAction[] }>();
  const developOutrightGroups = new Map<
    CardId,
    { options: DevelopOutrightAction[] }
  >();

  for (const action of actions) {
    if (action.type === 'trade') {
      const existing = tradeGroups.get(action.give);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        tradeGroups.set(action.give, { options });
        tradeItems.push({ kind: 'trade-group', give: action.give, options });
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
        buyDeedItems.push({
          kind: 'buy-deed-group',
          cardId: action.cardId,
          options,
        });
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
        developDeedItems.push({
          kind: 'develop-deed-group',
          cardId: action.cardId,
          districtId: action.districtId,
          options,
        });
      }
      continue;
    }

    if (action.type === 'develop-outright') {
      const existing = developOutrightGroups.get(action.cardId);

      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        developOutrightGroups.set(action.cardId, { options });
        developOutrightItems.push({
          kind: 'develop-outright-group',
          cardId: action.cardId,
          options,
        });
      }
      continue;
    }

    const existing = nonGroupedByType.get(action.type);
    if (existing) {
      existing.push(action);
    } else {
      nonGroupedByType.set(action.type, [action]);
    }
  }

  const sellCardItems = toActionItems(nonGroupedByType.get('sell-card'));
  const endTurnItems = toActionItems(nonGroupedByType.get('end-turn'));
  const otherActionItems: Extract<HumanActionListItem, { kind: 'action' }>[] =
    [];

  for (const [type, grouped] of nonGroupedByType.entries()) {
    if (type === 'sell-card' || type === 'end-turn') {
      continue;
    }
    otherActionItems.push(...toActionItems(grouped));
  }

  return [
    ...developOutrightItems,
    ...buyDeedItems,
    ...sellCardItems,
    ...developDeedItems,
    ...tradeItems,
    ...otherActionItems,
    ...endTurnItems,
  ];
}

export function pickerStillLegal(
  picker: ActionPickerQuery,
  actions: readonly GameAction[]
): boolean {
  if (picker.kind === 'trade') {
    return actions.some(
      (action): action is TradeAction =>
        action.type === 'trade' && action.give === picker.give
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

  if (picker.kind === 'district' && picker.actionType === 'buy-deed') {
    const options = actions.filter(
      (action): action is BuyDeedAction =>
        action.type === 'buy-deed' && action.cardId === picker.cardId
    );
    return options.length > 1;
  }

  if (picker.kind === 'develop-outright-district') {
    const districtIds = new Set<string>();
    for (const action of actions) {
      if (action.type !== 'develop-outright' || action.cardId !== picker.cardId) {
        continue;
      }
      districtIds.add(action.districtId);
    }
    return districtIds.size > 1;
  }

  if (picker.kind !== 'develop-outright-payment') {
    return false;
  }

  const options = actions.filter(
    (action): action is DevelopOutrightAction =>
      action.type === 'develop-outright' &&
      action.cardId === picker.cardId &&
      action.districtId === picker.districtId
  );
  return options.length > 0;
}

export function buildPickerOptions(
  picker: ActionPickerQuery,
  actions: readonly GameAction[],
  suitEmoji: Record<Suit, string>
): PickerOption[] {
  if (picker.kind === 'trade') {
    return actions
      .filter(
        (action): action is TradeAction =>
          action.type === 'trade' && action.give === picker.give
      )
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

  if (picker.kind === 'district' && picker.actionType === 'buy-deed') {
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

  if (picker.kind === 'develop-outright-district') {
    const firstActionByDistrict = new Map<string, DevelopOutrightAction>();
    for (const action of actions) {
      if (
        action.type !== 'develop-outright'
        || action.cardId !== picker.cardId
        || firstActionByDistrict.has(action.districtId)
      ) {
        continue;
      }
      firstActionByDistrict.set(action.districtId, action);
    }
    return [...firstActionByDistrict.values()].map((action) => ({
      id: `develop-outright-district:${picker.cardId}:${action.districtId}`,
      label: action.districtId,
      action,
    }));
  }

  if (picker.kind !== 'develop-outright-payment') {
    return [];
  }

  return actions
    .filter(
      (action): action is DevelopOutrightAction =>
        action.type === 'develop-outright' &&
        action.cardId === picker.cardId &&
        action.districtId === picker.districtId
    )
    .map((action) => ({
      id: actionStableKey(action),
      label: formatTokens(action.payment, suitEmoji),
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

  if (picker.kind === 'district' && picker.actionType === 'buy-deed') {
    return `Buy deed ${cardSummary(picker.cardId, suitEmoji)} in`;
  }

  if (picker.kind === 'develop-outright-district') {
    return `Develop ${cardSummary(picker.cardId, suitEmoji)} in`;
  }

  if (picker.kind !== 'develop-outright-payment') {
    return 'Select option';
  }

  return `Develop ${cardSummary(
    picker.cardId,
    suitEmoji
  )} in ${picker.districtId} with`;
}

export function describeAction(
  action: GameAction,
  suitEmoji: Record<Suit, string>
): string {
  switch (action.type) {
    case 'end-turn':
      return 'End turn';
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

export function formatTokens(
  tokens: Partial<Record<Suit, number>>,
  suitEmoji: Record<Suit, string>
): string {
  const entries = tokenEntries(tokens);
  if (entries.length === 0) {
    return '-';
  }
  return entries
    .map((entry) => `${suitEmoji[entry.suit]}x${entry.count}`)
    .join(' ');
}

export function cardSummary(
  cardId: CardId,
  suitEmoji: Record<Suit, string>
): string {
  const card = CARD_BY_ID[cardId];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const suits =
    card.kind === 'Excuse'
      ? ''
      : card.suits.map((suit) => suitEmoji[suit]).join('');
  return `${rank}${suits}`;
}

function tokenEntries(
  tokens: Partial<Record<Suit, number>>
): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
}

function toActionItems(
  actions: readonly NonGroupedAction[] | undefined
): Array<Extract<HumanActionListItem, { kind: 'action' }>> {
  if (!actions || actions.length === 0) {
    return [];
  }
  return actions.map((action) => ({ kind: 'action', action }));
}
