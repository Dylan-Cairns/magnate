import { ActionPickerKind } from './actionValues';
import { ActionId } from '../engine/values';
import type { CardId } from '../engine/cards';
import type { GameAction, Suit } from '../engine/types';
import {
  cardSummary,
  paymentSignature,
  pickerTitle,
  type ActionPickerQuery,
} from './actionPresentation';

type TradeAction = Extract<GameAction, { type: typeof ActionId.Trade }>;
type DevelopOutrightAction = Extract<
  GameAction,
  { type: typeof ActionId.DevelopOutright }
>;

export type TradeCompositePicker = {
  selectedGive?: Suit;
  selectedReceive?: Suit;
};

export type DevelopOutrightCompositePicker = {
  cardId: CardId;
  selectedDistrictId?: string;
  selectedPaymentKey?: string;
};

type Positioned<T> = T & {
  top: number;
  left: number;
};

export type StandardActionPickerState = Positioned<ActionPickerQuery>;

export type ActionPickerState =
  | StandardActionPickerState
  | Positioned<
      TradeCompositePicker & {
        kind: typeof ActionPickerKind.TradeCombined;
      }
    >
  | Positioned<
      DevelopOutrightCompositePicker & {
        kind: typeof ActionPickerKind.DevelopOutrightCombined;
      }
    >;

export function tradeActionsForPicker(
  actions: readonly GameAction[]
): TradeAction[] {
  return actions.filter(
    (action): action is TradeAction => action.type === ActionId.Trade
  );
}

// Candidate actions drive persistent submenu highlights as well as hover previews.
export function actionsForOpenPicker(
  picker: ActionPickerState,
  actions: readonly GameAction[]
): GameAction[] {
  switch (picker.kind) {
    case ActionPickerKind.TradeCombined:
      return tradeActionsForPicker(actions).filter(
        (action) =>
          (!picker.selectedGive || action.give === picker.selectedGive) &&
          (!picker.selectedReceive || action.receive === picker.selectedReceive)
      );
    case ActionPickerKind.DevelopOutrightCombined:
      return buildDevelopOutrightCompositeOptions(
        actions,
        picker.cardId
      ).outrightOptions.filter(
        (action) =>
          (!picker.selectedDistrictId ||
            action.districtId === picker.selectedDistrictId) &&
          (!picker.selectedPaymentKey ||
            paymentSignature(action.payment) === picker.selectedPaymentKey)
      );
    case ActionPickerKind.Trade:
      return tradeActionsForPicker(actions).filter(
        (action) => action.give === picker.give
      );
    case ActionPickerKind.District:
      return actions.filter(
        (action) =>
          action.type === ActionId.BuyDeed && action.cardId === picker.cardId
      );
    case ActionPickerKind.DevelopOutrightPayment:
      return buildDevelopOutrightCompositeOptions(
        actions,
        picker.cardId
      ).outrightOptions.filter(
        (action) => action.districtId === picker.districtId
      );
    case ActionPickerKind.DeedPayment:
      return actions.filter(
        (action) =>
          action.type === ActionId.DevelopDeed &&
          action.cardId === picker.cardId &&
          action.districtId === picker.districtId
      );
    case ActionPickerKind.IncomeChoice:
      return actions.filter(
        (action) =>
          action.type === ActionId.ChooseIncomeSuit &&
          action.cardId === picker.cardId &&
          action.districtId === picker.districtId &&
          action.playerId === picker.playerId
      );
  }
}

export function tradeReceiveOptions(actions: readonly TradeAction[]): Suit[] {
  return [...new Set(actions.map((action) => action.receive))];
}

export function resolveTradeCompositeAction(
  actions: readonly TradeAction[],
  selection: TradeCompositePicker
): TradeAction | undefined {
  if (!selection.selectedGive || !selection.selectedReceive) {
    return undefined;
  }
  return actions.find(
    (action) =>
      action.give === selection.selectedGive &&
      action.receive === selection.selectedReceive
  );
}

export function tradeCompositePickerStillLegal(
  picker: TradeCompositePicker,
  actions: readonly GameAction[]
): boolean {
  const tradeActions = tradeActionsForPicker(actions);
  const giveSuits = new Set(tradeActions.map((action) => action.give));
  if (giveSuits.size <= 1) {
    return false;
  }
  return !picker.selectedGive || giveSuits.has(picker.selectedGive);
}

export function buildDevelopOutrightCompositeOptions(
  actions: readonly GameAction[],
  cardId: CardId
): {
  outrightOptions: DevelopOutrightAction[];
  districtOptions: DevelopOutrightAction[];
  paymentOptions: Array<[string, DevelopOutrightAction]>;
} {
  const outrightOptions = actions.filter(
    (action): action is DevelopOutrightAction =>
      action.type === ActionId.DevelopOutright && action.cardId === cardId
  );
  const firstByDistrict = new Map<string, DevelopOutrightAction>();
  const firstByPayment = new Map<string, DevelopOutrightAction>();

  for (const option of outrightOptions) {
    if (!firstByDistrict.has(option.districtId)) {
      firstByDistrict.set(option.districtId, option);
    }
    const paymentKey = paymentSignature(option.payment);
    if (!firstByPayment.has(paymentKey)) {
      firstByPayment.set(paymentKey, option);
    }
  }

  return {
    outrightOptions,
    districtOptions: [...firstByDistrict.values()],
    paymentOptions: [...firstByPayment.entries()],
  };
}

export function resolveDevelopOutrightCompositeAction(
  actions: readonly DevelopOutrightAction[],
  selection: DevelopOutrightCompositePicker
): DevelopOutrightAction | undefined {
  if (!selection.selectedDistrictId || !selection.selectedPaymentKey) {
    return undefined;
  }
  return actions.find(
    (action) =>
      action.districtId === selection.selectedDistrictId &&
      paymentSignature(action.payment) === selection.selectedPaymentKey
  );
}

export function developOutrightCompositePickerStillLegal(
  picker: DevelopOutrightCompositePicker,
  actions: readonly GameAction[]
): boolean {
  const { outrightOptions } = buildDevelopOutrightCompositeOptions(
    actions,
    picker.cardId
  );
  if (outrightOptions.length <= 1) {
    return false;
  }
  if (
    picker.selectedDistrictId &&
    !outrightOptions.some(
      (option) => option.districtId === picker.selectedDistrictId
    )
  ) {
    return false;
  }
  if (
    picker.selectedPaymentKey &&
    !outrightOptions.some(
      (option) => paymentSignature(option.payment) === picker.selectedPaymentKey
    )
  ) {
    return false;
  }
  return true;
}

export function toPickerQuery(
  picker: StandardActionPickerState
): ActionPickerQuery {
  if (picker.kind === ActionPickerKind.Trade) {
    return { kind: ActionPickerKind.Trade, give: picker.give };
  }
  if (picker.kind === ActionPickerKind.DeedPayment) {
    return {
      kind: ActionPickerKind.DeedPayment,
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  if (picker.kind === ActionPickerKind.DevelopOutrightPayment) {
    return {
      kind: ActionPickerKind.DevelopOutrightPayment,
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  if (picker.kind === ActionPickerKind.IncomeChoice) {
    return {
      kind: ActionPickerKind.IncomeChoice,
      playerId: picker.playerId,
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  return {
    kind: ActionPickerKind.District,
    actionType: picker.actionType,
    cardId: picker.cardId,
  };
}

export function actionPickerTitle(
  picker: ActionPickerState,
  suitTokens: Record<Suit, string>
): string {
  if (picker.kind === ActionPickerKind.TradeCombined) {
    return 'Trade resources';
  }
  if (picker.kind === ActionPickerKind.DevelopOutrightCombined) {
    return `Develop ${cardSummary(picker.cardId, suitTokens)}`;
  }
  return pickerTitle(toPickerQuery(picker), suitTokens);
}
