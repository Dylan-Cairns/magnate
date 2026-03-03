import type { CardId } from '../engine/cards';
import type { GameAction, Suit } from '../engine/types';
import {
  cardSummary,
  paymentSignature,
  pickerTitle,
  type ActionPickerQuery,
} from './actionPresentation';

type TradeAction = Extract<GameAction, { type: 'trade' }>;
type DevelopOutrightAction = Extract<GameAction, { type: 'develop-outright' }>;

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
        kind: 'trade-combined';
      }
    >
  | Positioned<
      DevelopOutrightCompositePicker & {
        kind: 'develop-outright-combined';
      }
    >;

export function tradeActionsForPicker(
  actions: readonly GameAction[]
): TradeAction[] {
  return actions.filter(
    (action): action is TradeAction => action.type === 'trade'
  );
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
      action.type === 'develop-outright' && action.cardId === cardId
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
  if (picker.kind === 'trade') {
    return { kind: 'trade', give: picker.give };
  }
  if (picker.kind === 'deed-payment') {
    return {
      kind: 'deed-payment',
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  if (picker.kind === 'develop-outright-district') {
    return {
      kind: 'develop-outright-district',
      cardId: picker.cardId,
    };
  }
  if (picker.kind === 'develop-outright-payment') {
    return {
      kind: 'develop-outright-payment',
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  return {
    kind: 'district',
    actionType: picker.actionType,
    cardId: picker.cardId,
  };
}

export function actionPickerTitle(
  picker: ActionPickerState,
  suitTokens: Record<Suit, string>
): string {
  if (picker.kind === 'trade-combined') {
    return 'Trade resources';
  }
  if (picker.kind === 'develop-outright-combined') {
    return `Develop ${cardSummary(picker.cardId, suitTokens)}`;
  }
  return pickerTitle(toPickerQuery(picker), suitTokens);
}
