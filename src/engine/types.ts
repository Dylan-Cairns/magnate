export type Suit = 'Moons' | 'Suns' | 'Waves' | 'Leaves' | 'Wyrms' | 'Knots';

export type Rank = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10;

export type CardKind = 'Property' | 'Crown' | 'Pawn' | 'Excuse';

export interface CardBase {
  id: string;
  name: string;
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
  draw: string[];
  discard: string[];
  reshuffles: 0 | 1 | 2;
}
