// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const ActionListItemKind = {
  Action: 'action',
  TradeGroup: 'trade-group',
  BuyDeedGroup: 'buy-deed-group',
  DevelopDeedGroup: 'develop-deed-group',
  DevelopOutrightGroup: 'develop-outright-group',
  IncomeChoiceGroup: 'income-choice-group',
} as const;
export type ActionListItemKind =
  (typeof ActionListItemKind)[keyof typeof ActionListItemKind];

export const ActionPickerKind = {
  Trade: 'trade',
  District: 'district',
  DevelopOutrightPayment: 'develop-outright-payment',
  DeedPayment: 'deed-payment',
  IncomeChoice: 'income-choice',
  TradeCombined: 'trade-combined',
  DevelopOutrightCombined: 'develop-outright-combined',
} as const;
export type ActionPickerKind =
  (typeof ActionPickerKind)[keyof typeof ActionPickerKind];
