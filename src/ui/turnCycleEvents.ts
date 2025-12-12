import { CARD_BY_ID, type CardId } from '../engine/cards';
import { findProperty } from '../engine/stateHelpers';
import type {
  GameAction,
  GameState,
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  Suit,
} from '../engine/types';

export interface TurnCycleTaxSummary {
  suit: Suit;
  lossesByPlayer: ReadonlyArray<{
    playerId: PlayerId;
    count: number;
  }>;
}

export interface TurnCycleIncomeToken {
  playerId: PlayerId;
  suit: Suit;
  source:
    | {
        kind: 'district-card';
        cardId: CardId;
        districtId: string;
      }
    | {
        kind: 'crown';
        cardId: CardId;
      };
}

export interface TurnCycleIncomeHighlight {
  playerId: PlayerId;
  cardId: CardId;
  districtId: string;
}

export interface TurnCycleEvents {
  cycleOwner: PlayerId;
  roll: IncomeRollResult;
  incomeRank: number;
  tax: TurnCycleTaxSummary | null;
  incomeTokens: ReadonlyArray<TurnCycleIncomeToken>;
  incomeHighlights: ReadonlyArray<TurnCycleIncomeHighlight>;
  pendingChoices: ReadonlyArray<IncomeChoice>;
}

export function deriveTurnCycleEvents(
  previousState: GameState,
  nextState: GameState,
  action?: GameAction
): TurnCycleEvents | null {
  if (!action || action.type !== 'end-turn') {
    return null;
  }

  const roll = nextState.lastIncomeRoll;
  if (!roll) {
    return null;
  }

  const cycleOwner = resolveCycleOwner(previousState, nextState);
  if (!cycleOwner) {
    return null;
  }

  const incomeRank = Math.max(roll.die1, roll.die2);
  const tax = summarizeTax(previousState, nextState);
  const { incomeTokens, incomeHighlights } = resolveIncome(previousState, incomeRank);

  return {
    cycleOwner,
    roll,
    incomeRank,
    tax,
    incomeTokens,
    incomeHighlights,
    pendingChoices: nextState.pendingIncomeChoices ?? [],
  };
}

function resolveCycleOwner(
  previousState: GameState,
  nextState: GameState
): PlayerId | undefined {
  if (nextState.incomeChoiceReturnPlayerId) {
    return nextState.incomeChoiceReturnPlayerId;
  }

  const nextActive = nextState.players[nextState.activePlayerIndex];
  if (nextActive) {
    return nextActive.id;
  }

  return previousState.players[previousState.activePlayerIndex]?.id;
}

function summarizeTax(
  previousState: GameState,
  nextState: GameState
): TurnCycleTaxSummary | null {
  const taxSuit = nextState.lastTaxSuit;
  if (!taxSuit) {
    return null;
  }

  return {
    suit: taxSuit,
    lossesByPlayer: previousState.players.map((player) => ({
      playerId: player.id,
      count: Math.max(player.resources[taxSuit] - 1, 0),
    })),
  };
}

function resolveIncome(
  state: GameState,
  incomeRank: number
): {
  incomeTokens: ReadonlyArray<TurnCycleIncomeToken>;
  incomeHighlights: ReadonlyArray<TurnCycleIncomeHighlight>;
} {
  const tokens: TurnCycleIncomeToken[] = [];
  const highlights: TurnCycleIncomeHighlight[] = [];
  const highlightedCardIds = new Set<CardId>();

  const addHighlight = (playerId: PlayerId, districtId: string, cardId: CardId): void => {
    if (highlightedCardIds.has(cardId)) {
      return;
    }
    highlightedCardIds.add(cardId);
    highlights.push({ playerId, districtId, cardId });
  };

  const addDistrictCardToken = (
    playerId: PlayerId,
    districtId: string,
    cardId: CardId,
    suit: Suit
  ): void => {
    tokens.push({
      playerId,
      suit,
      source: {
        kind: 'district-card',
        cardId,
        districtId,
      },
    });
  };

  if (incomeRank === 10) {
    for (const player of state.players) {
      for (const crownId of player.crowns) {
        const crown = CARD_BY_ID[crownId];
        if (crown.kind !== 'Crown') {
          continue;
        }
        tokens.push({
          playerId: player.id,
          suit: crown.suits[0],
          source: {
            kind: 'crown',
            cardId: crown.id,
          },
        });
      }
    }
    return {
      incomeTokens: tokens,
      incomeHighlights: highlights,
    };
  }

  for (const player of state.players) {
    for (const district of state.districts) {
      const stack = district.stacks[player.id];
      for (const developedCardId of stack.developed) {
        const property = findProperty(developedCardId);
        if (!property || property.rank !== incomeRank) {
          continue;
        }
        addHighlight(player.id, district.id, developedCardId);
        for (const suit of property.suits) {
          addDistrictCardToken(player.id, district.id, developedCardId, suit);
        }
      }

      const deedProperty = stack.deed ? findProperty(stack.deed.cardId) : undefined;
      if (!deedProperty || deedProperty.rank !== incomeRank) {
        continue;
      }

      addHighlight(player.id, district.id, deedProperty.id);
      if (deedProperty.suits.length === 1) {
        addDistrictCardToken(player.id, district.id, deedProperty.id, deedProperty.suits[0]);
      }
    }
  }

  return {
    incomeTokens: tokens,
    incomeHighlights: highlights,
  };
}
