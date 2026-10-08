// Shared finite values. Preserve serialized spellings and explicit compatibility orders.

export const Suit = {
  Moons: 'Moons',
  Suns: 'Suns',
  Waves: 'Waves',
  Leaves: 'Leaves',
  Wyrms: 'Wyrms',
  Knots: 'Knots',
} as const;
export type Suit = (typeof Suit)[keyof typeof Suit];

export const CardKind = {
  Property: 'Property',
  Court: 'Court',
  Crown: 'Crown',
  Pawn: 'Pawn',
  Excuse: 'Excuse',
} as const;
export type CardKind = (typeof CardKind)[keyof typeof CardKind];

export const PlayerId = {
  PlayerA: 'PlayerA',
  PlayerB: 'PlayerB',
} as const;
export type PlayerId = (typeof PlayerId)[keyof typeof PlayerId];

export const Ruleset = {
  Standard: 'standard',
  Extended: 'extended',
} as const;
export type Ruleset = (typeof Ruleset)[keyof typeof Ruleset];

export const GamePhase = {
  StartTurn: 'StartTurn',
  TaxCheck: 'TaxCheck',
  CollectIncome: 'CollectIncome',
  ActionWindow: 'ActionWindow',
  DrawCard: 'DrawCard',
  GameOver: 'GameOver',
} as const;
export type GamePhase = (typeof GamePhase)[keyof typeof GamePhase];

export const ActionId = {
  BuyDeed: 'buy-deed',
  ChooseIncomeSuit: 'choose-income-suit',
  DevelopDeed: 'develop-deed',
  DevelopOutright: 'develop-outright',
  EndTurn: 'end-turn',
  SellCard: 'sell-card',
  Trade: 'trade',
} as const;
export type ActionId = (typeof ActionId)[keyof typeof ActionId];

export const Winner = {
  PlayerA: PlayerId.PlayerA,
  PlayerB: PlayerId.PlayerB,
  Draw: 'Draw',
} as const;
export type Winner = (typeof Winner)[keyof typeof Winner];

export const WinnerDecider = {
  Districts: 'districts',
  RankTotal: 'rank-total',
  Resources: 'resources',
  Draw: 'draw',
} as const;
export type WinnerDecider = (typeof WinnerDecider)[keyof typeof WinnerDecider];

export const SUITS = [
  Suit.Moons,
  Suit.Suns,
  Suit.Waves,
  Suit.Leaves,
  Suit.Wyrms,
  Suit.Knots,
] as const;

export const PLAYER_IDS = [PlayerId.PlayerA, PlayerId.PlayerB] as const;

export const GAME_PHASES = [
  GamePhase.StartTurn,
  GamePhase.TaxCheck,
  GamePhase.CollectIncome,
  GamePhase.ActionWindow,
  GamePhase.DrawCard,
  GamePhase.GameOver,
] as const;

export const ACTION_IDS = [
  ActionId.BuyDeed,
  ActionId.ChooseIncomeSuit,
  ActionId.DevelopDeed,
  ActionId.DevelopOutright,
  ActionId.EndTurn,
  ActionId.SellCard,
  ActionId.Trade,
] as const;
