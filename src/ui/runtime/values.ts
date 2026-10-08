// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const RuntimeModeType = {
  Idle: 'idle',
  Animating: 'animating',
  AwaitingInput: 'awaiting-input',
} as const;
export type RuntimeModeType =
  (typeof RuntimeModeType)[keyof typeof RuntimeModeType];

export const DicePhase = {
  Rolling: 'rolling',
  Settled: 'settled',
  Hidden: 'hidden',
  Dimmed: 'dimmed',
} as const;
export type DicePhase = (typeof DicePhase)[keyof typeof DicePhase];

export const IncomeTokenSourceKind = {
  DistrictCard: 'district-card',
  Crown: 'crown',
  IncomeChoice: 'income-choice',
} as const;
export type IncomeTokenSourceKind =
  (typeof IncomeTokenSourceKind)[keyof typeof IncomeTokenSourceKind];

export const GamePresentationEventType = {
  ActionStarted: 'action-started',
  DrawCard: 'draw-card',
  CardSold: 'card-sold',
  SellResourceGained: 'sell-resource-gained',
  ResourcePaymentStarted: 'resource-payment-started',
  ResourcePaymentApplied: 'resource-payment-applied',
  CardPlayedToDistrict: 'card-played-to-district',
  DeedTokenPaid: 'deed-token-paid',
  DeedProgressApplied: 'deed-progress-applied',
  DeedCompleted: 'deed-completed',
  TradeResourcesApplied: 'trade-resources-applied',
  IncomeRoll: 'income-roll',
  TaxResolved: 'tax-resolved',
  TaxTokenLost: 'tax-token-lost',
  IncomeTokenGained: 'income-token-gained',
  IncomeChoiceRequired: 'income-choice-required',
  IncomeChoiceSubmitted: 'income-choice-submitted',
  ActivePlayerChanged: 'active-player-changed',
  PhaseChanged: 'phase-changed',
  TransactionSettled: 'transaction-settled',
} as const;
export type GamePresentationEventType =
  (typeof GamePresentationEventType)[keyof typeof GamePresentationEventType];

export const AnimationStepType = {
  HoldPreviousState: 'hold-previous-state',
  DrawCardFlight: 'draw-card-flight',
  StageSoldCard: 'stage-sold-card',
  LaunchSellTokenFlights: 'launch-sell-token-flights',
  LandSellToken: 'land-sell-token',
  LaunchPaymentTokenFlights: 'launch-payment-token-flights',
  ApplyResourcePaymentToken: 'apply-resource-payment-token',
  ApplyResourcePayment: 'apply-resource-payment',
  LaunchCardToDistrictFlight: 'launch-card-to-district-flight',
  PlaceCardInDistrict: 'place-card-in-district',
  LaunchDeedTokenFlights: 'launch-deed-token-flights',
  ApplyDeedTokens: 'apply-deed-tokens',
  ApplyDeedProgress: 'apply-deed-progress',
  RevealDeedCompletion: 'reveal-deed-completion',
  LaunchTradeTokenFlights: 'launch-trade-token-flights',
  ApplyTradeTokenLoss: 'apply-trade-token-loss',
  LandTradeToken: 'land-trade-token',
  ApplyTradeTokenGain: 'apply-trade-token-gain',
  RollIncomeDice: 'roll-income-dice',
  RollTaxDie: 'roll-tax-die',
  HoldBeforeTaxFlights: 'hold-before-tax-flights',
  LaunchTaxTokenFlights: 'launch-tax-token-flights',
  ApplyTaxTokenLoss: 'apply-tax-token-loss',
  StageGap: 'stage-gap',
  HoldBeforeIncomeFlights: 'hold-before-income-flights',
  HighlightIncomeSources: 'highlight-income-sources',
  LaunchIncomeTokenFlights: 'launch-income-token-flights',
  LandIncomeToken: 'land-income-token',
  PostIncomeHold: 'post-income-hold',
  RevealIncomeChoiceRequest: 'reveal-income-choice-request',
  RevealIncomeChoiceSubmission: 'reveal-income-choice-submission',
  CommitViewState: 'commit-view-state',
} as const;
export type AnimationStepType =
  (typeof AnimationStepType)[keyof typeof AnimationStepType];

export const AnimationVisualCommandType = {
  LaunchDrawCardFlight: 'launch-draw-card-flight',
  LaunchSoldCardFlight: 'launch-sold-card-flight',
  LaunchSellTokenFlights: 'launch-sell-token-flights',
  LaunchCardToDistrictFlight: 'launch-card-to-district-flight',
  LaunchPaymentTokenFlights: 'launch-payment-token-flights',
  LaunchTradeTokenFlights: 'launch-trade-token-flights',
  LaunchDeedTokenFlights: 'launch-deed-token-flights',
  PulseTaxResources: 'pulse-tax-resources',
  LaunchTaxTokenFlights: 'launch-tax-token-flights',
  LaunchIncomeTokenFlights: 'launch-income-token-flights',
} as const;
export type AnimationVisualCommandType =
  (typeof AnimationVisualCommandType)[keyof typeof AnimationVisualCommandType];
