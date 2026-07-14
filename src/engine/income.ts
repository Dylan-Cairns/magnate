import { CARD_BY_ID } from './cards';
import { findProperty } from './stateHelpers';
import type { GameState, IncomeChoice, PlayerId, Rank, Suit } from './types';

export interface IncomeForResult {
  readonly fixedDelta: Partial<Record<Suit, number>>;
  readonly pendingChoices: readonly IncomeChoice[];
}

/**
 * Returns the exact income sources for one effective die result.
 *
 * This is canonical rules logic shared by turn resolution and read-only
 * strategic summaries. It deliberately contains no die probabilities or
 * future-income forecast.
 */
export function incomeForResult(
  state: GameState,
  playerId: PlayerId,
  result: Rank
): IncomeForResult {
  const fixedDelta: Partial<Record<Suit, number>> = {};
  const pendingChoices: IncomeChoice[] = [];

  if (result === 10) {
    awardCrownIncome(state, playerId, fixedDelta);
    return { fixedDelta, pendingChoices };
  }

  if (result === 1) {
    awardAceIncome(state, playerId, fixedDelta);
    return { fixedDelta, pendingChoices };
  }

  awardRankIncome(state, playerId, result, fixedDelta, pendingChoices);
  return { fixedDelta, pendingChoices };
}

function awardCrownIncome(
  state: GameState,
  playerId: PlayerId,
  delta: Partial<Record<Suit, number>>
): void {
  const player = state.players.find((item) => item.id === playerId);
  if (!player) {
    return;
  }

  player.crowns.forEach((cardId) => {
    const card = CARD_BY_ID[cardId];
    if (card.kind === 'Crown') {
      addSuit(delta, card.suits[0], 1);
    }
  });
}

function awardAceIncome(
  state: GameState,
  playerId: PlayerId,
  delta: Partial<Record<Suit, number>>
): void {
  state.districts.forEach((district) => {
    const stack = district.stacks[playerId];
    stack.developed.forEach((cardId) => {
      const property = findProperty(cardId);
      if (property?.rank === 1) {
        property.suits.forEach((suit) => addSuit(delta, suit, 1));
      }
    });

    const deed = stack.deed ? findProperty(stack.deed.cardId) : undefined;
    if (deed?.rank === 1) {
      addSuit(delta, deed.suits[0], 1);
    }
  });
}

function awardRankIncome(
  state: GameState,
  playerId: PlayerId,
  rank: Exclude<Rank, 1 | 10>,
  delta: Partial<Record<Suit, number>>,
  pendingChoices: IncomeChoice[]
): void {
  state.districts.forEach((district) => {
    const stack = district.stacks[playerId];
    stack.developed.forEach((cardId) => {
      const property = findProperty(cardId);
      if (property?.rank === rank) {
        property.suits.forEach((suit) => addSuit(delta, suit, 1));
      }
    });

    const deed = stack.deed ? findProperty(stack.deed.cardId) : undefined;
    if (deed?.rank !== rank) {
      return;
    }
    if (deed.suits.length === 1) {
      addSuit(delta, deed.suits[0], 1);
      return;
    }
    pendingChoices.push({
      playerId,
      districtId: district.id,
      cardId: deed.id,
      suits: deed.suits,
    });
  });
}

function addSuit(
  delta: Partial<Record<Suit, number>>,
  suit: Suit,
  amount: number
): void {
  delta[suit] = (delta[suit] ?? 0) + amount;
}
