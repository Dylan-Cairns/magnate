import { ActionListItemKind } from './actionValues';
import { ActionId } from '../engine/values';
import type { GameAction, PlayerId } from '../engine/types';
import type { HumanActionListItem } from './actionPresentation';

type DevelopOutrightAction = Extract<
  GameAction,
  { type: typeof ActionId.DevelopOutright }
>;

export function hasVisibleIncomeChoiceActions(
  items: readonly HumanActionListItem[]
): boolean {
  return items.some(
    (item) =>
      (item.kind === ActionListItemKind.Action &&
        item.action.type === ActionId.ChooseIncomeSuit) ||
      item.kind === ActionListItemKind.IncomeChoiceGroup
  );
}

export function isHumanInputActive({
  terminal,
  activePlayerId,
  humanPlayerId,
  visibleActionItems,
  humanActionUiBlockedByAnimation,
  isIncomeChoicePhase,
}: {
  terminal: boolean;
  activePlayerId: PlayerId;
  humanPlayerId: PlayerId;
  visibleActionItems: readonly HumanActionListItem[];
  humanActionUiBlockedByAnimation: boolean;
  isIncomeChoicePhase: boolean;
}): boolean {
  if (terminal || humanActionUiBlockedByAnimation) {
    return false;
  }
  // Canonical play can advance ahead of the displayed phase, so the human
  // owning the turn is not enough: the bot's income choice on the human's turn
  // must not light the human input area before any human actions exist.
  if (visibleActionItems.length === 0) {
    return false;
  }
  return isIncomeChoicePhase
    ? hasVisibleIncomeChoiceActions(visibleActionItems)
    : activePlayerId === humanPlayerId;
}

export function actionCategoryForItem(item: HumanActionListItem): string {
  switch (item.kind) {
    case ActionListItemKind.TradeGroup:
      return ActionId.Trade;
    case ActionListItemKind.BuyDeedGroup:
      return ActionId.BuyDeed;
    case ActionListItemKind.DevelopDeedGroup:
      return ActionId.DevelopDeed;
    case ActionListItemKind.DevelopOutrightGroup:
      return ActionId.DevelopOutright;
    case ActionListItemKind.IncomeChoiceGroup:
      return ActionId.ChooseIncomeSuit;
    case ActionListItemKind.Action:
      return item.action.type;
  }
}

export function actionCategoryLabel(category: string): string {
  switch (category) {
    case ActionId.Trade:
      return 'Trade';
    case ActionId.BuyDeed:
      return 'Buy Deed';
    case ActionId.DevelopDeed:
      return 'Develop Deed';
    case ActionId.DevelopOutright:
      return 'Develop Outright';
    case ActionId.SellCard:
      return 'Sell Card';
    case ActionId.ChooseIncomeSuit:
      return 'Choose Income';
    case ActionId.EndTurn:
      return 'End Turn';
    default:
      return category;
  }
}

// A card's outright develop options collapse to a single district in the button
// label only when that district is forced; a forced payment is never promoted
// while the district choice is still pending.
export function buildDevelopOutrightGroupPresentation(
  options: readonly DevelopOutrightAction[]
): {
  singleDistrictId?: string;
} {
  const districtIds = new Set<string>();

  for (const option of options) {
    districtIds.add(option.districtId);
  }

  return {
    singleDistrictId:
      districtIds.size === 1 ? districtIds.values().next().value : undefined,
  };
}
