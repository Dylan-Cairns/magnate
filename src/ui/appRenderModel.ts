import type { CardId } from '../engine/cards';
import { PROPERTY_CARDS } from '../engine/cards';
import type { GameState, Suit } from '../engine/types';

const ALL_SUITS: Suit[] = [
  'Moons',
  'Suns',
  'Waves',
  'Leaves',
  'Wyrms',
  'Knots',
];

const SUIT_CARD_IDS = new Map<Suit, ReadonlyArray<CardId>>(
  ALL_SUITS.map((suit) => [
    suit,
    PROPERTY_CARDS.filter((card) => card.suits.includes(suit)).map(
      (card) => card.id
    ),
  ])
);

export type DeckMapDimming = {
  dimmedCardIds: Set<CardId>;
  dimmedSuits: Set<Suit>;
};

export function isVisibleIncomeChoicePhase(viewState: GameState): boolean {
  return (
    viewState.phase === 'CollectIncome' &&
    (viewState.pendingIncomeChoices?.length ?? 0) > 0
  );
}

export function buildDeckMapDimming({
  deckMapInteractive,
  viewState,
}: {
  deckMapInteractive: boolean;
  viewState: GameState;
}): DeckMapDimming {
  if (!deckMapInteractive) {
    return {
      dimmedCardIds: new Set<CardId>(),
      dimmedSuits: new Set<Suit>(),
    };
  }

  const inCirculation = new Set<CardId>([
    ...viewState.deck.draw,
    ...viewState.players.flatMap((player) => player.hand),
    ...(viewState.deck.reshuffles === 0 ? viewState.deck.discard : []),
  ]);
  const dimmedCardIds = new Set<CardId>();
  for (const card of PROPERTY_CARDS) {
    if (card.suits.length === 2 && !inCirculation.has(card.id)) {
      dimmedCardIds.add(card.id);
    }
  }

  const dimmedSuits = new Set<Suit>();
  for (const [suit, cardIds] of SUIT_CARD_IDS) {
    if (cardIds.every((id) => !inCirculation.has(id))) {
      dimmedSuits.add(suit);
    }
  }
  return { dimmedCardIds, dimmedSuits };
}
