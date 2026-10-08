import {
  Suit,
  CardKind,
  PlayerId,
  Ruleset,
  GamePhase,
  Winner,
  WinnerDecider,
  ActionId,
} from './values';
import type { CardId, CardName } from './cards';

export type Rank = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

export interface CardBase {
  id: CardId;
  name: CardName;
  kind: CardKind;
}

export interface PropertyCard extends CardBase {
  kind: typeof CardKind.Property;
  rank: Exclude<Rank, 10>;
  suits: readonly Suit[];
}

// Courts are the extended-deck property cards. They are developable like
// properties but always rank 10 with three suits and never provide income.
export interface CourtCard extends CardBase {
  kind: typeof CardKind.Court;
  rank: 10;
  suits: readonly [Suit, Suit, Suit];
}

export type DevelopableCard = PropertyCard | CourtCard;

export interface CrownCard extends CardBase {
  kind: typeof CardKind.Crown;
  rank: 10;
  suits: readonly [Suit];
}

export interface PawnCard extends CardBase {
  kind: typeof CardKind.Pawn;
  suits: readonly [Suit, Suit, Suit];
}

export interface ExcuseCard extends CardBase {
  kind: typeof CardKind.Excuse;
}

export type Card = PropertyCard | CourtCard | CrownCard | PawnCard | ExcuseCard;

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

export interface IncomeRollResult {
  die1: number;
  die2: number;
  rollId?: number;
}

export interface IncomeChoice {
  playerId: PlayerId;
  districtId: DistrictId;
  cardId: CardId;
  suits: readonly Suit[];
}

export interface SubmittedIncomeChoice {
  playerId: PlayerId;
  districtId: DistrictId;
  cardId: CardId;
  suit: Suit;
}

export interface GameLogEntry {
  turn: number;
  player: PlayerId;
  phase: GamePhase;
  summary: string;
  details?: Record<string, unknown>;
}

export interface FinalScore {
  districtPoints: Record<PlayerId, number>;
  rankTotals: Record<PlayerId, number>;
  resourceTotals: Record<PlayerId, number>;
  winner: Winner;
  decidedBy: WinnerDecider;
}

export interface ObservedPlayerState {
  id: PlayerId;
  crowns: CardId[];
  resources: ResourcePool;
  hand: CardId[];
  handCount: number;
  handHidden: boolean;
}

export interface PublicDeckView {
  drawCount: number;
  discard: CardId[];
  reshuffles: 0 | 1 | 2;
}

export interface PlayerView {
  viewerId: PlayerId;
  activePlayerId: PlayerId;
  turn: number;
  phase: GamePhase;
  districts: DistrictLine;
  players: ReadonlyArray<ObservedPlayerState>;
  deck: PublicDeckView;
  cardPlayedThisTurn: boolean;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  lastTaxSuit?: Suit;
  pendingIncomeChoices?: ReadonlyArray<IncomeChoice>;
  submittedIncomeChoices?: ReadonlyArray<SubmittedIncomeChoice>;
  incomeChoiceReturnPlayerId?: PlayerId;
  finalScore?: FinalScore;
  log: ReadonlyArray<GameLogEntry>;
}

export interface GameState {
  schemaVersion: number;
  seed: string;
  rngCursor: number;
  ruleset: Ruleset;
  deck: DeckState;
  players: ReadonlyArray<PlayerState>;
  activePlayerIndex: number;
  turn: number;
  phase: GamePhase;
  districts: DistrictLine;
  cardPlayedThisTurn: boolean;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  lastTaxSuit?: Suit;
  pendingIncomeChoices?: ReadonlyArray<IncomeChoice>;
  submittedIncomeChoices?: ReadonlyArray<SubmittedIncomeChoice>;
  incomeChoiceReturnPlayerId?: PlayerId;
  finalScore?: FinalScore;
  log: ReadonlyArray<GameLogEntry>;
}

export interface BuyDeedAction {
  type: typeof ActionId.BuyDeed;
  cardId: CardId;
  districtId: DistrictId;
}

export interface DevelopDeedAction {
  type: typeof ActionId.DevelopDeed;
  districtId: DistrictId;
  cardId: CardId;
  tokens: Partial<Record<Suit, number>>;
}

export interface DevelopOutrightAction {
  type: typeof ActionId.DevelopOutright;
  cardId: CardId;
  districtId: DistrictId;
  payment: Partial<Record<Suit, number>>;
}

export interface SellCardAction {
  type: typeof ActionId.SellCard;
  cardId: CardId;
}

export interface TradeAction {
  type: typeof ActionId.Trade;
  give: Suit;
  receive: Suit;
}

export interface EndTurnAction {
  type: typeof ActionId.EndTurn;
}

export interface ChooseIncomeSuitAction {
  type: typeof ActionId.ChooseIncomeSuit;
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
  | EndTurnAction
  | SellCardAction
  | TradeAction;

export {
  Suit,
  CardKind,
  PlayerId,
  Ruleset,
  GamePhase,
  Winner,
  WinnerDecider,
  ActionId,
} from './values';
