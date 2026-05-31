import type { GameAction } from '../engine/types';
import {
  paymentSignature,
  type HumanActionListItem,
} from './actionPresentation';

type DevelopOutrightAction = Extract<GameAction, { type: 'develop-outright' }>;

export function actionCategoryForItem(item: HumanActionListItem): string {
  switch (item.kind) {
    case 'trade-group':
      return 'trade';
    case 'buy-deed-group':
      return 'buy-deed';
    case 'develop-deed-group':
      return 'develop-deed';
    case 'develop-outright-group':
      return 'develop-outright';
    case 'action':
      return item.action.type;
  }
}

export function actionCategoryLabel(category: string): string {
  switch (category) {
    case 'trade':
      return 'Trade';
    case 'buy-deed':
      return 'Buy Deed';
    case 'develop-deed':
      return 'Develop Deed';
    case 'develop-outright':
      return 'Develop Outright';
    case 'sell-card':
      return 'Sell Card';
    case 'choose-income-suit':
      return 'Choose Income';
    case 'end-turn':
      return 'End Turn';
    default:
      return category;
  }
}

export function buildDevelopOutrightGroupPresentation(
  options: readonly DevelopOutrightAction[]
): {
  districtCount: number;
  hasSinglePaymentPattern: boolean;
  firstPayment?: DevelopOutrightAction['payment'];
} {
  const paymentOptions = new Map<string, DevelopOutrightAction>();
  const districtIds = new Set<string>();

  for (const option of options) {
    districtIds.add(option.districtId);
    const paymentKey = paymentSignature(option.payment);
    if (!paymentOptions.has(paymentKey)) {
      paymentOptions.set(paymentKey, option);
    }
  }

  const firstPayment = paymentOptions.values().next().value?.payment;
  return {
    districtCount: districtIds.size,
    hasSinglePaymentPattern: paymentOptions.size === 1,
    firstPayment,
  };
}
