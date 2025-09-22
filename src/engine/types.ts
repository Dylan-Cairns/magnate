import type { CardId, CardName } from './cards';
export type Suit = 'Moons' | 'Suns' | 'Waves' | 'Leaves' | 'Wyrms' | 'Knots';

export type Rank = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

export type CardKind = 'Property' | 'Crown' | 'Pawn' | 'Excuse';

export interface CardBase {
  id: CardId;
  name: CardName;
  kind: CardKind;
}

export interface PropertyCard extends CardBase {
  kind: 'Property';
  rank: Exclude<Rank, 10>;
  suits: readonly Suit[];
}

export interface CrownCard extends CardBase {
  kind: 'Crown';
  rank: 10;
  suits: readonly [Suit];
}

export interface PawnCard extends CardBase {
  kind: 'Pawn';
  suits: readonly [Suit, Suit, Suit];
}

export interface ExcuseCard extends CardBase {
  kind: 'Excuse';
}

export type Card = PropertyCard | CrownCard | PawnCard | ExcuseCard;

export interface DeckState {
  draw: CardId[];
  discard: CardId[];
  reshuffles: 0 | 1 | 2;
}


export interface DistrictStack {
  developed: CardId[];
  deed?: DeedState;
}

export interface DistrictState {
  id: DistrictId;
  markerSuitMask: readonly Suit[];
  stacks: Record<PlayerId, DistrictStack>;
}

export type DistrictLine = ReadonlyArray<DistrictState>;
export type PlayerId = 'PlayerA' | 'PlayerB';

export type ResourcePool = Record<Suit, number>;

export type DistrictId = string;

export interface DeedState {
  cardId: CardId;
  progress: number;
  tokens: Partial<Record<Suit, number>>;
}

export interface PlayerState {
  id: PlayerId;
  hand: CardId[];
  crowns: CardId[];
  resources: ResourcePool;
}

export type GamePhase =
  | 'StartTurn'
  | 'TaxCheck'
  | 'IncomeRoll'
  | 'CollectIncome'
  | 'OptionalTrade'
  | 'OptionalDevelop'
  | 'PlayCard'
  | 'DrawCard'
  | 'GameOver';

export interface IncomeRollResult {
  die1: number;
  die2: number;
}

export interface IncomeChoice {
  playerId: PlayerId;
  districtId: DistrictId;
  cardId: CardId;
  suits: readonly Suit[];
}

export interface GameLogEntry {
  turn: number;
  player: PlayerId;
  phase: GamePhase;
  summary: string;
  details?: Record<string, unknown>;
}

export type Winner = PlayerId | 'Draw';
export type WinnerDecider =
  | 'districts'
  | 'rank-total'
  | 'resources'
  | 'draw';

export interface FinalScore {
  districtPoints: Record<PlayerId, number>;
  rankTotals: Record<PlayerId, number>;
  resourceTotals: Record<PlayerId, number>;
  winner: Winner;
  decidedBy: WinnerDecider;
}

export interface GameState {
  schemaVersion: number;
  seed: string;
  rngCursor: number;
  deck: DeckState;
  players: ReadonlyArray<PlayerState>;
  activePlayerIndex: number;
  turn: number;
  phase: GamePhase;
  districts: DistrictLine;
  cardPlayedThisTurn: boolean;
  exhaustionStage: 0 | 1 | 2;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  pendingIncomeChoices?: ReadonlyArray<IncomeChoice>;
  incomeChoiceReturnPlayerIndex?: number;
  finalScore?: FinalScore;
  log: ReadonlyArray<GameLogEntry>;
}

export type ActionId =
  | 'buy-deed'
  | 'choose-income-suit'
  | 'develop-deed'
  | 'develop-outright'
  | 'end-optional-develop'
  | 'end-optional-trade'
  | 'end-turn'
  | 'sell-card'
  | 'trade';

export interface BuyDeedAction {
  type: 'buy-deed';
  cardId: CardId;
  districtId: DistrictId;
}

export interface DevelopDeedAction {
  type: 'develop-deed';
  districtId: DistrictId;
  cardId: CardId;
  tokens: Partial<Record<Suit, number>>;
}

export interface DevelopOutrightAction {
  type: 'develop-outright';
  cardId: CardId;
  districtId: DistrictId;
  payment: Partial<Record<Suit, number>>;
}

export interface SellCardAction {
  type: 'sell-card';
  cardId: CardId;
}

export interface TradeAction {
  type: 'trade';
  give: Suit;
  receive: Suit;
}

export interface EndOptionalTradeAction {
  type: 'end-optional-trade';
}

export interface EndOptionalDevelopAction {
  type: 'end-optional-develop';
}

export interface EndTurnAction {
  type: 'end-turn';
}

export interface ChooseIncomeSuitAction {
  type: 'choose-income-suit';
  playerId: PlayerId;
  districtId: DistrictId;
  cardId: CardId;
  suit: Suit;
}

export type GameAction =
  | BuyDeedAction
  | ChooseIncomeSuitAction
  | DevelopDeedAction
  | DevelopOutrightAction
  | EndOptionalDevelopAction
  | EndOptionalTradeAction
  | EndTurnAction
  | SellCardAction
  | TradeAction;
