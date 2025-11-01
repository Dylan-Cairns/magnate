import type {
  Card,
  Suit,
  PropertyCard,
  CrownCard,
  PawnCard,
  ExcuseCard,
} from './types';

type RawCardObject = {
  name: string;
  id: string;
  rank: 'Ace' | 'Numeral' | 'Crown' | 'Excuse' | 'Pawn';
  value:
    | '0'
    | '1'
    | '2'
    | '3'
    | '4'
    | '5'
    | '6'
    | '7'
    | '8'
    | '9'
    | '10'
    | '11';
  suit1: '' | Suit;
  suit2: '' | Suit;
  suit3: '' | Suit;
};

const RAW_CARD_OBJECTS: readonly RawCardObject[] = [
  {
    name: 'Ace of Knots',
    id: '0',
    rank: 'Ace',
    value: '1',
    suit1: 'Knots',
    suit2: '',
    suit3: '',
  },
  {
    name: 'Ace of Leaves',
    id: '1',
    rank: 'Ace',
    value: '1',
    suit1: 'Leaves',
    suit2: '',
    suit3: '',
  },
  {
    name: 'Ace of Moons',
    id: '2',
    rank: 'Ace',
    value: '1',
    suit1: 'Moons',
    suit2: '',
    suit3: '',
  },
  {
    name: 'Ace of Suns',
    id: '3',
    rank: 'Ace',
    value: '1',
    suit1: 'Suns',
    suit2: '',
    suit3: '',
  },
  {
    name: 'Ace of Waves',
    id: '4',
    rank: 'Ace',
    value: '1',
    suit1: 'Waves',
    suit2: '',
    suit3: '',
  },
  {
    name: 'Ace of Wyrms',
    id: '5',
    rank: 'Ace',
    value: '1',
    suit1: 'Wyrms',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Author',
    id: '6',
    rank: 'Numeral',
    value: '2',
    suit1: 'Moons',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Desert',
    id: '7',
    rank: 'Numeral',
    value: '2',
    suit1: 'Suns',
    suit2: 'Wyrms',
    suit3: '',
  },
  {
    name: 'The Origin',
    id: '8',
    rank: 'Numeral',
    value: '2',
    suit1: 'Waves',
    suit2: 'Leaves',
    suit3: '',
  },
  {
    name: 'The Journey',
    id: '9',
    rank: 'Numeral',
    value: '3',
    suit1: 'Moons',
    suit2: 'Waves',
    suit3: '',
  },
  {
    name: 'The Painter',
    id: '10',
    rank: 'Numeral',
    value: '3',
    suit1: 'Suns',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Savage',
    id: '11',
    rank: 'Numeral',
    value: '3',
    suit1: 'Leaves',
    suit2: 'Wyrms',
    suit3: '',
  },
  {
    name: 'The Battle',
    id: '12',
    rank: 'Numeral',
    value: '4',
    suit1: 'Wyrms',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Mountain',
    id: '13',
    rank: 'Numeral',
    value: '4',
    suit1: 'Moons',
    suit2: 'Suns',
    suit3: '',
  },
  {
    name: 'The Sailor',
    id: '14',
    rank: 'Numeral',
    value: '4',
    suit1: 'Waves',
    suit2: 'Leaves',
    suit3: '',
  },
  {
    name: 'The Discovery',
    id: '15',
    rank: 'Numeral',
    value: '5',
    suit1: 'Suns',
    suit2: 'Waves',
    suit3: '',
  },
  {
    name: 'The Forest',
    id: '16',
    rank: 'Numeral',
    value: '5',
    suit1: 'Moons',
    suit2: 'Leaves',
    suit3: '',
  },
  {
    name: 'The Soldier',
    id: '17',
    rank: 'Numeral',
    value: '5',
    suit1: 'Wyrms',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Lunatic',
    id: '18',
    rank: 'Numeral',
    value: '6',
    suit1: 'Moons',
    suit2: 'Waves',
    suit3: '',
  },
  {
    name: 'The Market',
    id: '19',
    rank: 'Numeral',
    value: '6',
    suit1: 'Leaves',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Penitent',
    id: '20',
    rank: 'Numeral',
    value: '6',
    suit1: 'Suns',
    suit2: 'Wyrms',
    suit3: '',
  },
  {
    name: 'The Castle',
    id: '21',
    rank: 'Numeral',
    value: '7',
    suit1: 'Suns',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Cave',
    id: '22',
    rank: 'Numeral',
    value: '7',
    suit1: 'Waves',
    suit2: 'Wyrms',
    suit3: '',
  },
  {
    name: 'The Chance Meeting',
    id: '23',
    rank: 'Numeral',
    value: '7',
    suit1: 'Moons',
    suit2: 'Leaves',
    suit3: '',
  },
  {
    name: 'The Betrayal',
    id: '24',
    rank: 'Numeral',
    value: '8',
    suit1: 'Wyrms',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Diplomat',
    id: '25',
    rank: 'Numeral',
    value: '8',
    suit1: 'Moons',
    suit2: 'Suns',
    suit3: '',
  },
  {
    name: 'The Mill',
    id: '26',
    rank: 'Numeral',
    value: '8',
    suit1: 'Waves',
    suit2: 'Leaves',
    suit3: '',
  },
  {
    name: 'The Darkness',
    id: '27',
    rank: 'Numeral',
    value: '9',
    suit1: 'Waves',
    suit2: 'Wyrms',
    suit3: '',
  },
  {
    name: 'The Merchant',
    id: '28',
    rank: 'Numeral',
    value: '9',
    suit1: 'Leaves',
    suit2: 'Knots',
    suit3: '',
  },
  {
    name: 'The Pact',
    id: '29',
    rank: 'Numeral',
    value: '9',
    suit1: 'Moons',
    suit2: 'Suns',
    suit3: '',
  },
  {
    name: 'The Windfall',
    id: '30',
    rank: 'Crown',
    value: '10',
    suit1: 'Knots',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The End',
    id: '31',
    rank: 'Crown',
    value: '10',
    suit1: 'Leaves',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Huntress',
    id: '32',
    rank: 'Crown',
    value: '10',
    suit1: 'Moons',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Bard',
    id: '33',
    rank: 'Crown',
    value: '10',
    suit1: 'Suns',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Sea',
    id: '34',
    rank: 'Crown',
    value: '10',
    suit1: 'Waves',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Calamity',
    id: '35',
    rank: 'Crown',
    value: '10',
    suit1: 'Wyrms',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Excuse',
    id: '36',
    rank: 'Excuse',
    value: '0',
    suit1: '',
    suit2: '',
    suit3: '',
  },
  {
    name: 'The Borderland',
    id: '37',
    rank: 'Pawn',
    value: '11',
    suit1: 'Waves',
    suit2: 'Leaves',
    suit3: 'Wyrms',
  },
  {
    name: 'The Harvest',
    id: '38',
    rank: 'Pawn',
    value: '11',
    suit1: 'Moons',
    suit2: 'Suns',
    suit3: 'Leaves',
  },
  {
    name: 'The Light Keeper',
    id: '39',
    rank: 'Pawn',
    value: '11',
    suit1: 'Suns',
    suit2: 'Waves',
    suit3: 'Knots',
  },
  {
    name: 'The Watchman',
    id: '40',
    rank: 'Pawn',
    value: '11',
    suit1: 'Moons',
    suit2: 'Wyrms',
    suit3: 'Knots',
  },
] as const;

export type CardId = (typeof RAW_CARD_OBJECTS)[number]['id'];
export type CardName = (typeof RAW_CARD_OBJECTS)[number]['name'];

const toCard = (raw: RawCardObject): Card => {
  if (raw.rank === 'Excuse') {
    const card: ExcuseCard = {
      id: raw.id as CardId,
      name: raw.name as CardName,
      kind: 'Excuse',
    };
    return card;
  }
  if (raw.rank === 'Pawn') {
    const suits = [raw.suit1, raw.suit2, raw.suit3] as [Suit, Suit, Suit];
    const card: PawnCard = {
      id: raw.id as CardId,
      name: raw.name as CardName,
      kind: 'Pawn',
      suits,
    };
    return card;
  }
  if (raw.rank === 'Crown') {
    const rank = 10 as const;
    const suits = [raw.suit1 as Suit] as [Suit];
    const card: CrownCard = {
      id: raw.id as CardId,
      name: raw.name as CardName,
      kind: 'Crown',
      rank,
      suits,
    };
    return card;
  }
  const rank =
    raw.rank === 'Ace'
      ? 1
      : (parseInt(raw.value, 10) as 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9);
  const suits: Suit[] = [raw.suit1, raw.suit2, raw.suit3].filter(
    Boolean
  ) as Suit[];
  const card: PropertyCard = {
    id: raw.id as CardId,
    name: raw.name as CardName,
    kind: 'Property',
    rank,
    suits,
  };
  return card;
};

export const ALL_CARDS: Card[] = RAW_CARD_OBJECTS.map(toCard);

export const CARD_BY_ID: Record<CardId, Card> = ALL_CARDS.reduce((acc, c) => {
  acc[c.id] = c;
  return acc;
}, Object.create(null) as Record<CardId, Card>);

export const PROPERTY_CARDS = ALL_CARDS.filter(
  (c): c is PropertyCard => c.kind === 'Property'
);
export const CROWN_CARDS = ALL_CARDS.filter(
  (c): c is CrownCard => c.kind === 'Crown'
);
export const PAWN_CARDS = ALL_CARDS.filter(
  (c): c is PawnCard => c.kind === 'Pawn'
);
export const EXCUSE_CARD = ALL_CARDS.find(
  (c): c is ExcuseCard => c.kind === 'Excuse'
)!;
