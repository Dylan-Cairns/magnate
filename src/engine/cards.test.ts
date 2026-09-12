import { describe, expect, it } from 'vitest';

import {
  ALL_CARDS,
  CARD_BY_ID,
  CROWN_CARDS,
  EXCUSE_CARD,
  PAWN_CARDS,
  PROPERTY_CARDS,
  type CardId,
} from './cards';

// Compact fingerprint authored from the Jacynth Decktet fact table
// (decktet_cards.csv rows 2-42) using Magnate's display casing. This is an
// independent oracle, not a copy of the implementation.
const EXPECTED_CATALOG = `
0 Ace of Knots | Knots
1 Ace of Leaves | Leaves
2 Ace of Moons | Moons
3 Ace of Suns | Suns
4 Ace of Waves | Waves
5 Ace of Wyrms | Wyrms
6 The Author | Moons,Knots
7 The Desert | Suns,Wyrms
8 The Origin | Waves,Leaves
9 The Journey | Moons,Waves
10 The Painter | Suns,Knots
11 The Savage | Leaves,Wyrms
12 The Battle | Wyrms,Knots
13 The Mountain | Moons,Suns
14 The Sailor | Waves,Leaves
15 The Discovery | Suns,Waves
16 The Forest | Moons,Leaves
17 The Soldier | Wyrms,Knots
18 The Lunatic | Moons,Waves
19 The Market | Leaves,Knots
20 The Penitent | Suns,Wyrms
21 The Castle | Suns,Knots
22 The Cave | Waves,Wyrms
23 The Chance Meeting | Moons,Leaves
24 The Betrayal | Wyrms,Knots
25 The Diplomat | Moons,Suns
26 The Mill | Waves,Leaves
27 The Darkness | Waves,Wyrms
28 The Merchant | Leaves,Knots
29 The Pact | Moons,Suns
30 The Windfall | Knots
31 The End | Leaves
32 The Huntress | Moons
33 The Bard | Suns
34 The Sea | Waves
35 The Calamity | Wyrms
36 The Excuse |
37 The Borderland | Waves,Leaves,Wyrms
38 The Harvest | Moons,Suns,Leaves
39 The Light Keeper | Suns,Waves,Knots
40 The Watchman | Moons,Wyrms,Knots
`
  .trim()
  .split('\n')
  .join(';');

const describeCard = (card: (typeof ALL_CARDS)[number]): string => {
  const suits = 'suits' in card ? card.suits.join(',') : '';
  return `${card.id} ${card.name} |${suits ? ` ${suits}` : ''}`;
};

describe('card catalog', () => {
  it('matches the Jacynth-derived compatibility fingerprint', () => {
    expect(ALL_CARDS.map(describeCard).join(';')).toBe(EXPECTED_CATALOG);
  });

  it('orders ALL_CARDS as the contiguous id sequence "0" through "40"', () => {
    const expectedIds = Array.from({ length: 41 }, (_, index) => String(index));
    expect(ALL_CARDS.map((card) => card.id)).toEqual(expectedIds);
  });

  it('has unique ids and names', () => {
    const ids = new Set(ALL_CARDS.map((card) => card.id));
    const names = new Set(ALL_CARDS.map((card) => card.name));
    expect(ids.size).toBe(ALL_CARDS.length);
    expect(names.size).toBe(ALL_CARDS.length);
  });

  it('has the expected card-class counts', () => {
    expect(ALL_CARDS).toHaveLength(41);
    expect(PROPERTY_CARDS).toHaveLength(30);
    expect(CROWN_CARDS).toHaveLength(6);
    expect(PAWN_CARDS).toHaveLength(4);
    expect(EXCUSE_CARD.kind).toBe('Excuse');
  });

  it('has six aces and three numeral cards at each rank 2 through 9', () => {
    const propertyRanks = PROPERTY_CARDS.map((card) => card.rank);
    expect(propertyRanks.filter((rank) => rank === 1)).toHaveLength(6);
    for (let rank = 2; rank <= 9; rank += 1) {
      expect(propertyRanks.filter((value) => value === rank)).toHaveLength(3);
    }
  });

  it('respects per-class suit counts and distinctness', () => {
    const aces = PROPERTY_CARDS.filter((card) => card.rank === 1);
    for (const ace of aces) {
      expect(ace.suits).toHaveLength(1);
    }
    for (const numeral of PROPERTY_CARDS.filter((card) => card.rank !== 1)) {
      expect(new Set(numeral.suits).size).toBe(2);
    }
    for (const crown of CROWN_CARDS) {
      expect(crown.suits).toHaveLength(1);
    }
    for (const pawn of PAWN_CARDS) {
      expect(new Set(pawn.suits).size).toBe(3);
    }
  });

  it('indexes every card by id and preserves grouping order', () => {
    for (const card of ALL_CARDS) {
      expect(CARD_BY_ID[card.id]).toBe(card);
    }
    expect(PROPERTY_CARDS.map((card) => card.id)).toEqual(
      ALL_CARDS.filter((card) => card.kind === 'Property').map(
        (card) => card.id
      )
    );
  });

  it('uses Magnate display casing for "the" cards', () => {
    const author = CARD_BY_ID['6'];
    const chanceMeeting = CARD_BY_ID['23'];
    const lightKeeper = CARD_BY_ID['39'];
    expect(author.name).toBe('The Author');
    expect(chanceMeeting.name).toBe('The Chance Meeting');
    expect(lightKeeper.name).toBe('The Light Keeper');
    expect(ALL_CARDS.some((card) => /^the /.test(card.name))).toBe(false);
  });

  it('exposes representative cards from each category', () => {
    expect(CARD_BY_ID['0']).toMatchObject({
      name: 'Ace of Knots',
      kind: 'Property',
      rank: 1,
      suits: ['Knots'],
    });
    expect(CARD_BY_ID['6']).toMatchObject({
      name: 'The Author',
      kind: 'Property',
      rank: 2,
      suits: ['Moons', 'Knots'],
    });
    expect(CARD_BY_ID['30']).toMatchObject({
      name: 'The Windfall',
      kind: 'Crown',
      rank: 10,
      suits: ['Knots'],
    });
    expect(CARD_BY_ID['36']).toMatchObject({
      name: 'The Excuse',
      kind: 'Excuse',
    });
    expect(CARD_BY_ID['37']).toMatchObject({
      name: 'The Borderland',
      kind: 'Pawn',
      suits: ['Waves', 'Leaves', 'Wyrms'],
    });
    expect(EXCUSE_CARD.id).toBe('36');
  });

  it('accepts the canonical boundary ids', () => {
    const first: CardId = '0';
    const last: CardId = '40';
    expect(CARD_BY_ID[first].name).toBe('Ace of Knots');
    expect(CARD_BY_ID[last].name).toBe('The Watchman');
  });
});
