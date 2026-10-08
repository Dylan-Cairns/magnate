import { ActionListItemKind, ActionPickerKind } from './actionValues';
import { ActionId, CardKind } from '../engine/values';
import { CARD_BY_ID, type CardId } from '../engine/cards';
import { actionStableKey, paymentSignature } from '../engine/actionSurface';
import { SUITS } from '../engine/stateHelpers';
import type { GameAction, Suit } from '../engine/types';

export { actionStableKey, paymentSignature };

type TradeAction = Extract<GameAction, { type: typeof ActionId.Trade }>;
type BuyDeedAction = Extract<GameAction, { type: typeof ActionId.BuyDeed }>;
type DevelopDeedAction = Extract<
  GameAction,
  { type: typeof ActionId.DevelopDeed }
>;
type DevelopOutrightAction = Extract<
  GameAction,
  { type: typeof ActionId.DevelopOutright }
>;
type ChooseIncomeSuitAction = Extract<
  GameAction,
  { type: typeof ActionId.ChooseIncomeSuit }
>;
type NonGroupedAction = Exclude<
  GameAction,
  | TradeAction
  | BuyDeedAction
  | DevelopDeedAction
  | DevelopOutrightAction
  | ChooseIncomeSuitAction
>;
type DirectAction = NonGroupedAction | ChooseIncomeSuitAction;

export type HumanActionListItem =
  | { kind: typeof ActionListItemKind.Action; action: DirectAction }
  | {
      kind: typeof ActionListItemKind.TradeGroup;
      give: Suit;
      options: TradeAction[];
    }
  | {
      kind: typeof ActionListItemKind.BuyDeedGroup;
      cardId: CardId;
      options: BuyDeedAction[];
    }
  | {
      kind: typeof ActionListItemKind.DevelopDeedGroup;
      cardId: CardId;
      districtId: string;
      options: DevelopDeedAction[];
    }
  | {
      kind: typeof ActionListItemKind.DevelopOutrightGroup;
      cardId: CardId;
      options: DevelopOutrightAction[];
    }
  | {
      kind: typeof ActionListItemKind.IncomeChoiceGroup;
      playerId: ChooseIncomeSuitAction['playerId'];
      districtId: string;
      cardId: CardId;
      options: ChooseIncomeSuitAction[];
    };

export type ActionPickerQuery =
  | {
      kind: typeof ActionPickerKind.Trade;
      give: Suit;
    }
  | {
      kind: typeof ActionPickerKind.District;
      actionType: typeof ActionId.BuyDeed;
      cardId: CardId;
    }
  | {
      kind: typeof ActionPickerKind.DevelopOutrightPayment;
      cardId: CardId;
      districtId: string;
    }
  | {
      kind: typeof ActionPickerKind.DeedPayment;
      cardId: CardId;
      districtId: string;
    }
  | {
      kind: typeof ActionPickerKind.IncomeChoice;
      playerId: ChooseIncomeSuitAction['playerId'];
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
    if (action.type !== ActionId.Trade) {
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
  const tradeItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.TradeGroup }
  >[] = [];
  const buyDeedItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.BuyDeedGroup }
  >[] = [];
  const developDeedItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.DevelopDeedGroup }
  >[] = [];
  const developOutrightItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.DevelopOutrightGroup }
  >[] = [];
  const incomeChoiceItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.IncomeChoiceGroup }
  >[] = [];
  const nonGroupedByType = new Map<
    NonGroupedAction['type'],
    NonGroupedAction[]
  >();
  const tradeGroups = new Map<Suit, { options: TradeAction[] }>();
  const buyDeedGroups = new Map<CardId, { options: BuyDeedAction[] }>();
  const developDeedGroups = new Map<string, { options: DevelopDeedAction[] }>();
  const developOutrightGroups = new Map<
    CardId,
    { options: DevelopOutrightAction[] }
  >();
  const incomeChoiceGroups = new Map<
    string,
    {
      playerId: ChooseIncomeSuitAction['playerId'];
      districtId: string;
      cardId: CardId;
      options: ChooseIncomeSuitAction[];
    }
  >();

  for (const action of actions) {
    if (action.type === ActionId.ChooseIncomeSuit) {
      const groupKey = `${action.playerId}|${action.districtId}|${action.cardId}`;
      const existing = incomeChoiceGroups.get(groupKey);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        incomeChoiceGroups.set(groupKey, {
          playerId: action.playerId,
          districtId: action.districtId,
          cardId: action.cardId,
          options,
        });
      }
      continue;
    }

    if (action.type === ActionId.Trade) {
      const existing = tradeGroups.get(action.give);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        tradeGroups.set(action.give, { options });
        tradeItems.push({
          kind: ActionListItemKind.TradeGroup,
          give: action.give,
          options,
        });
      }
      continue;
    }

    if (action.type === ActionId.BuyDeed) {
      const existing = buyDeedGroups.get(action.cardId);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        buyDeedGroups.set(action.cardId, { options });
        buyDeedItems.push({
          kind: ActionListItemKind.BuyDeedGroup,
          cardId: action.cardId,
          options,
        });
      }
      continue;
    }

    if (action.type === ActionId.DevelopDeed) {
      const groupKey = `${action.cardId}|${action.districtId}`;
      const existing = developDeedGroups.get(groupKey);
      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        developDeedGroups.set(groupKey, { options });
        developDeedItems.push({
          kind: ActionListItemKind.DevelopDeedGroup,
          cardId: action.cardId,
          districtId: action.districtId,
          options,
        });
      }
      continue;
    }

    if (action.type === ActionId.DevelopOutright) {
      const existing = developOutrightGroups.get(action.cardId);

      if (existing) {
        existing.options.push(action);
      } else {
        const options = [action];
        developOutrightGroups.set(action.cardId, { options });
        developOutrightItems.push({
          kind: ActionListItemKind.DevelopOutrightGroup,
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

  const sellCardItems = toActionItems(nonGroupedByType.get(ActionId.SellCard));
  const endTurnItems = toActionItems(nonGroupedByType.get(ActionId.EndTurn));
  const otherActionItems: Extract<
    HumanActionListItem,
    { kind: typeof ActionListItemKind.Action }
  >[] = [];
  const incomeGroups = [...incomeChoiceGroups.values()];
  const incomeActionItems =
    incomeGroups.length > 1 ? [] : toActionItems(incomeGroups[0]?.options);
  if (incomeGroups.length > 1) {
    incomeChoiceItems.push(
      ...incomeGroups.map((group) => ({
        kind: ActionListItemKind.IncomeChoiceGroup,
        playerId: group.playerId,
        districtId: group.districtId,
        cardId: group.cardId,
        options: group.options,
      }))
    );
  }

  for (const [type, grouped] of nonGroupedByType.entries()) {
    if (type === ActionId.SellCard || type === ActionId.EndTurn) {
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
    ...incomeChoiceItems,
    ...incomeActionItems,
    ...otherActionItems,
    ...endTurnItems,
  ];
}

export function pickerStillLegal(
  picker: ActionPickerQuery,
  actions: readonly GameAction[]
): boolean {
  if (picker.kind === ActionPickerKind.Trade) {
    return actions.some(
      (action): action is TradeAction =>
        action.type === ActionId.Trade && action.give === picker.give
    );
  }

  if (picker.kind === ActionPickerKind.DeedPayment) {
    const options = actions.filter(
      (action): action is DevelopDeedAction =>
        action.type === ActionId.DevelopDeed &&
        action.cardId === picker.cardId &&
        action.districtId === picker.districtId
    );
    return options.length > 1;
  }

  if (picker.kind === ActionPickerKind.IncomeChoice) {
    const options = actions.filter(
      (action): action is ChooseIncomeSuitAction =>
        action.type === ActionId.ChooseIncomeSuit &&
        action.playerId === picker.playerId &&
        action.cardId === picker.cardId &&
        action.districtId === picker.districtId
    );
    return options.length > 0;
  }

  if (
    picker.kind === ActionPickerKind.District &&
    picker.actionType === ActionId.BuyDeed
  ) {
    const options = actions.filter(
      (action): action is BuyDeedAction =>
        action.type === ActionId.BuyDeed && action.cardId === picker.cardId
    );
    return options.length > 1;
  }

  if (picker.kind !== ActionPickerKind.DevelopOutrightPayment) {
    return false;
  }

  const options = actions.filter(
    (action): action is DevelopOutrightAction =>
      action.type === ActionId.DevelopOutright &&
      action.cardId === picker.cardId &&
      action.districtId === picker.districtId
  );
  return options.length > 1;
}

export function buildPickerOptions(
  picker: ActionPickerQuery,
  actions: readonly GameAction[],
  suitEmoji: Record<Suit, string>
): PickerOption[] {
  if (picker.kind === ActionPickerKind.Trade) {
    return actions
      .filter(
        (action): action is TradeAction =>
          action.type === ActionId.Trade && action.give === picker.give
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: `${suitEmoji[action.receive]} x1`,
        action,
      }));
  }

  if (picker.kind === ActionPickerKind.DeedPayment) {
    return actions
      .filter(
        (action): action is DevelopDeedAction =>
          action.type === ActionId.DevelopDeed &&
          action.cardId === picker.cardId &&
          action.districtId === picker.districtId
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: formatTokens(action.tokens, suitEmoji),
        action,
      }));
  }

  if (picker.kind === ActionPickerKind.IncomeChoice) {
    return actions
      .filter(
        (action): action is ChooseIncomeSuitAction =>
          action.type === ActionId.ChooseIncomeSuit &&
          action.playerId === picker.playerId &&
          action.cardId === picker.cardId &&
          action.districtId === picker.districtId
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: `${suitEmoji[action.suit]} x1`,
        action,
      }));
  }

  if (
    picker.kind === ActionPickerKind.District &&
    picker.actionType === ActionId.BuyDeed
  ) {
    return actions
      .filter(
        (action): action is BuyDeedAction =>
          action.type === ActionId.BuyDeed && action.cardId === picker.cardId
      )
      .map((action) => ({
        id: actionStableKey(action),
        label: action.districtId,
        action,
      }));
  }

  if (picker.kind !== ActionPickerKind.DevelopOutrightPayment) {
    return [];
  }

  return actions
    .filter(
      (action): action is DevelopOutrightAction =>
        action.type === ActionId.DevelopOutright &&
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
  if (picker.kind === ActionPickerKind.Trade) {
    return `Trade ${suitEmoji[picker.give]}x3 for`;
  }

  if (picker.kind === ActionPickerKind.DeedPayment) {
    return `Develop deed ${cardSummary(picker.cardId, suitEmoji)} in ${picker.districtId} with`;
  }

  if (picker.kind === ActionPickerKind.IncomeChoice) {
    return `Choose income ${cardSummary(picker.cardId, suitEmoji)} in ${picker.districtId}`;
  }

  if (
    picker.kind === ActionPickerKind.District &&
    picker.actionType === ActionId.BuyDeed
  ) {
    return `Buy deed ${cardSummary(picker.cardId, suitEmoji)} in`;
  }

  if (picker.kind !== ActionPickerKind.DevelopOutrightPayment) {
    return 'Select option';
  }

  return `Develop ${cardSummary(
    picker.cardId,
    suitEmoji
  )} in ${picker.districtId} with`;
}

export function pickerGroupLabel(picker: ActionPickerQuery): string {
  switch (picker.kind) {
    case ActionPickerKind.Trade:
      return 'Receive x1';
    case ActionPickerKind.District:
      return 'District';
    case ActionPickerKind.DeedPayment:
      return 'Payment';
    case ActionPickerKind.DevelopOutrightPayment:
      return 'Payment';
    case ActionPickerKind.IncomeChoice:
      return 'Suit';
  }
}

export function describeAction(
  action: GameAction,
  suitEmoji: Record<Suit, string>
): string {
  switch (action.type) {
    case ActionId.EndTurn:
      return 'End turn';
    case ActionId.Trade:
      return `Trade ${suitEmoji[action.give]}x3 for ${suitEmoji[action.receive]}x1`;
    case ActionId.SellCard:
      return `Sell ${cardSummary(action.cardId, suitEmoji)}`;
    case ActionId.BuyDeed:
      return `Buy deed ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId}`;
    case ActionId.DevelopDeed:
      return `Develop deed ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId} (${formatTokens(
        action.tokens,
        suitEmoji
      )})`;
    case ActionId.DevelopOutright:
      return `Develop ${cardSummary(action.cardId, suitEmoji)} in ${action.districtId} (${formatTokens(
        action.payment,
        suitEmoji
      )})`;
    case ActionId.ChooseIncomeSuit:
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
    card.kind === CardKind.Property ||
    card.kind === CardKind.Crown ||
    card.kind === CardKind.Court
      ? String(card.rank)
      : card.kind === CardKind.Pawn
        ? 'P'
        : 'X';
  const suits =
    card.kind === CardKind.Excuse
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
  actions: readonly DirectAction[] | undefined
): Array<
  Extract<HumanActionListItem, { kind: typeof ActionListItemKind.Action }>
> {
  if (!actions || actions.length === 0) {
    return [];
  }
  return actions.map((action) => ({ kind: ActionListItemKind.Action, action }));
}
