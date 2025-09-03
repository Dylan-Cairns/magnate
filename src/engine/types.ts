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

export interface GameLogEntry {
  turn: number;
  player: PlayerId;
  phase: GamePhase;
  summary: string;
  details?: Record<string, unknown>;
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
  exhaustionStage: 0 | 1 | 2;
  finalTurnsRemaining?: number;
  lastIncomeRoll?: IncomeRollResult;
  log: ReadonlyArray<GameLogEntry>;
}
