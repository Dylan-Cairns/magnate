import type {
  Card,
  CrownCard,
  ExcuseCard,
  PawnCard,
  PropertyCard,
  Suit,
} from './types';

export type CardId = string;

// Decktet facts below are authored from the author's own Jacynth extraction:
// jacynth/src/public/javascript/model/decktet_cards.csv (data rows 2-42).
// Magnate's playable deck is that 41-card set. The four Courts documented in
// the same table are intentionally out of scope here, but their art is retained
// for a planned extended ruleset. Display casing follows Magnate, not Jacynth.

// Aces are grouped by suit in ASCII order.
const ACE_SUITS = [
  'Knots',
  'Leaves',
  'Moons',
  'Suns',
  'Waves',
  'Wyrms',
] as const satisfies readonly Suit[];

// Numeral properties are grouped by rank; each rank has three cards.
const NUMERALS_BY_RANK = [
  {
    rank: 2,
    cards: [
      { name: 'The Author', suits: ['Moons', 'Knots'] },
      { name: 'The Desert', suits: ['Suns', 'Wyrms'] },
      { name: 'The Origin', suits: ['Waves', 'Leaves'] },
    ],
  },
  {
    rank: 3,
    cards: [
      { name: 'The Journey', suits: ['Moons', 'Waves'] },
      { name: 'The Painter', suits: ['Suns', 'Knots'] },
      { name: 'The Savage', suits: ['Leaves', 'Wyrms'] },
    ],
  },
  {
    rank: 4,
    cards: [
      { name: 'The Battle', suits: ['Wyrms', 'Knots'] },
      { name: 'The Mountain', suits: ['Moons', 'Suns'] },
      { name: 'The Sailor', suits: ['Waves', 'Leaves'] },
    ],
  },
  {
    rank: 5,
    cards: [
      { name: 'The Discovery', suits: ['Suns', 'Waves'] },
      { name: 'The Forest', suits: ['Moons', 'Leaves'] },
      { name: 'The Soldier', suits: ['Wyrms', 'Knots'] },
    ],
  },
  {
    rank: 6,
    cards: [
      { name: 'The Lunatic', suits: ['Moons', 'Waves'] },
      { name: 'The Market', suits: ['Leaves', 'Knots'] },
      { name: 'The Penitent', suits: ['Suns', 'Wyrms'] },
    ],
  },
  {
    rank: 7,
    cards: [
      { name: 'The Castle', suits: ['Suns', 'Knots'] },
      { name: 'The Cave', suits: ['Waves', 'Wyrms'] },
      { name: 'The Chance Meeting', suits: ['Moons', 'Leaves'] },
    ],
  },
  {
    rank: 8,
    cards: [
      { name: 'The Betrayal', suits: ['Wyrms', 'Knots'] },
      { name: 'The Diplomat', suits: ['Moons', 'Suns'] },
      { name: 'The Mill', suits: ['Waves', 'Leaves'] },
    ],
  },
  {
    rank: 9,
    cards: [
      { name: 'The Darkness', suits: ['Waves', 'Wyrms'] },
      { name: 'The Merchant', suits: ['Leaves', 'Knots'] },
      { name: 'The Pact', suits: ['Moons', 'Suns'] },
    ],
  },
] as const;

// Crowns are defined as name-to-suit pairs.
const CROWNS = [
  { name: 'The Windfall', suit: 'Knots' },
  { name: 'The End', suit: 'Leaves' },
  { name: 'The Huntress', suit: 'Moons' },
  { name: 'The Bard', suit: 'Suns' },
  { name: 'The Sea', suit: 'Waves' },
  { name: 'The Calamity', suit: 'Wyrms' },
] as const;

// Pawns are defined as name-to-suit-set entries.
const PAWNS = [
  { name: 'The Borderland', suits: ['Waves', 'Leaves', 'Wyrms'] },
  { name: 'The Harvest', suits: ['Moons', 'Suns', 'Leaves'] },
  { name: 'The Light Keeper', suits: ['Suns', 'Waves', 'Knots'] },
  { name: 'The Watchman', suits: ['Moons', 'Wyrms', 'Knots'] },
] as const;

const EXCUSE_NAME = 'The Excuse' as const;

// CardId and CardName intentionally remain `string`, matching the historical
// surface. IDs are still generated contiguously as "0"-"40" and locked by tests.
export type CardName = string;

type NumericRank = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

type CardSpec = {
  name: CardName;
  kind: Card['kind'];
  rank?: NumericRank;
  suits: readonly Suit[];
};

const compareAscii = (a: string, b: string): number =>
  a < b ? -1 : a > b ? 1 : 0;

const aceSpecs: CardSpec[] = [...ACE_SUITS].sort(compareAscii).map((suit) => ({
  name: `Ace of ${suit}`,
  kind: 'Property',
  rank: 1,
  suits: [suit],
}));

const numeralSpecs: CardSpec[] = [...NUMERALS_BY_RANK]
  .sort((a, b) => a.rank - b.rank)
  .flatMap((group) =>
    [...group.cards]
      .sort((a, b) => compareAscii(a.name, b.name))
      .map((card) => ({
        name: card.name,
        kind: 'Property' as const,
        rank: group.rank,
        suits: card.suits,
      }))
  );

const crownSpecs: CardSpec[] = [...CROWNS]
  .sort((a, b) => compareAscii(a.suit, b.suit))
  .map((card) => ({
    name: card.name,
    kind: 'Crown',
    suits: [card.suit],
  }));

const excuseSpec: CardSpec = {
  name: EXCUSE_NAME,
  kind: 'Excuse',
  suits: [],
};

const pawnSpecs: CardSpec[] = [...PAWNS]
  .sort((a, b) => compareAscii(a.name, b.name))
  .map((card) => ({
    name: card.name,
    kind: 'Pawn',
    suits: card.suits,
  }));

// Canonical ordering reproduces Magnate's existing card IDs 0-40.
const CARD_SPECS: readonly CardSpec[] = [
  ...aceSpecs,
  ...numeralSpecs,
  ...crownSpecs,
  excuseSpec,
  ...pawnSpecs,
];

const toCard = (id: CardId, spec: CardSpec): Card => {
  switch (spec.kind) {
    case 'Excuse': {
      const card: ExcuseCard = {
        id,
        name: spec.name,
        kind: 'Excuse',
      };
      return card;
    }
    case 'Pawn': {
      const card: PawnCard = {
        id,
        name: spec.name,
        kind: 'Pawn',
        suits: spec.suits as readonly [Suit, Suit, Suit],
      };
      return card;
    }
    case 'Crown': {
      const card: CrownCard = {
        id,
        name: spec.name,
        kind: 'Crown',
        rank: 10,
        suits: spec.suits as readonly [Suit],
      };
      return card;
    }
    case 'Property': {
      if (spec.rank === undefined) {
        throw new Error(`Property card ${spec.name} is missing a rank.`);
      }
      const card: PropertyCard = {
        id,
        name: spec.name,
        kind: 'Property',
        rank: spec.rank,
        suits: spec.suits,
      };
      return card;
    }
  }
};

export const ALL_CARDS: Card[] = CARD_SPECS.map((spec, index) =>
  toCard(String(index) as CardId, spec)
);

function assertCatalog(cards: readonly Card[]): void {
  const PROPERTY_COUNT = 30;
  const ACE_COUNT = 6;
  const NUMERALS_PER_RANK = 3;
  const CROWN_COUNT = 6;
  const PAWN_COUNT = 4;
  const EXCUSE_COUNT = 1;

  if (
    cards.length !==
    PROPERTY_COUNT + CROWN_COUNT + PAWN_COUNT + EXCUSE_COUNT
  ) {
    throw new Error(
      `Magnate catalog must contain exactly 41 cards, found ${cards.length}.`
    );
  }

  const ids = new Set<string>();
  const names = new Set<string>();
  const numeralsByRank = new Map<number, number>();
  let aces = 0;
  let crowns = 0;
  let pawns = 0;
  let excuses = 0;

  for (const card of cards) {
    if (ids.has(card.id)) {
      throw new Error(`Duplicate card id in Magnate catalog: ${card.id}`);
    }
    if (names.has(card.name)) {
      throw new Error(`Duplicate card name in Magnate catalog: ${card.name}`);
    }
    ids.add(card.id);
    names.add(card.name);

    switch (card.kind) {
      case 'Excuse':
        excuses += 1;
        break;
      case 'Pawn': {
        pawns += 1;
        if (card.suits.length !== 3 || new Set(card.suits).size !== 3) {
          throw new Error(`Pawn ${card.name} must have three distinct suits.`);
        }
        break;
      }
      case 'Crown': {
        crowns += 1;
        if (card.suits.length !== 1) {
          throw new Error(`Crown ${card.name} must have exactly one suit.`);
        }
        break;
      }
      case 'Property': {
        if (new Set(card.suits).size !== card.suits.length) {
          throw new Error(`Property ${card.name} has duplicate suits.`);
        }
        if (card.rank === 1) {
          aces += 1;
          if (card.suits.length !== 1) {
            throw new Error(`Ace ${card.name} must have exactly one suit.`);
          }
        } else {
          if (card.suits.length !== 2) {
            throw new Error(
              `Numeral ${card.name} must have exactly two suits.`
            );
          }
          numeralsByRank.set(
            card.rank,
            (numeralsByRank.get(card.rank) ?? 0) + 1
          );
        }
        break;
      }
    }
  }

  if (aces !== ACE_COUNT) {
    throw new Error(`Expected ${ACE_COUNT} Aces, found ${aces}.`);
  }
  for (let rank = 2; rank <= 9; rank += 1) {
    if ((numeralsByRank.get(rank) ?? 0) !== NUMERALS_PER_RANK) {
      throw new Error(
        `Expected ${NUMERALS_PER_RANK} numeral cards at rank ${rank}, found ${
          numeralsByRank.get(rank) ?? 0
        }.`
      );
    }
  }
  if (crowns !== CROWN_COUNT) {
    throw new Error(`Expected ${CROWN_COUNT} Crowns, found ${crowns}.`);
  }
  if (pawns !== PAWN_COUNT) {
    throw new Error(`Expected ${PAWN_COUNT} Pawns, found ${pawns}.`);
  }
  if (excuses !== EXCUSE_COUNT) {
    throw new Error(`Expected ${EXCUSE_COUNT} Excuse, found ${excuses}.`);
  }
}

assertCatalog(ALL_CARDS);

export const CARD_BY_ID: Record<CardId, Card> = ALL_CARDS.reduce(
  (acc, c) => {
    acc[c.id] = c;
    return acc;
  },
  Object.create(null) as Record<CardId, Card>
);

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
