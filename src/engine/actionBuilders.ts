import { CARD_BY_ID } from './cards';
import {
  SUITS,
  canAfford,
  deedCost,
  developmentCost,
  enumerateOutrightPayments,
  placementAllowed,
} from './stateHelpers';
import type { GameAction, GamePhase, GameState } from './types';

type PhaseBuilderMap = Partial<Record<GamePhase, (state: GameState) => GameAction[]>>;

const builders: PhaseBuilderMap = {
  CollectIncome: collectIncomeChoiceActions,
  ActionWindow: actionWindowActions,
};

export function legalActions(state: GameState): readonly GameAction[] {
  const phaseBuilder = builders[state.phase];
  return phaseBuilder ? phaseBuilder(state) : [];
}

function actionWindowActions(state: GameState): GameAction[] {
  const trades = tradeActions(state);
  const develops = developActions(state);
  if (state.cardPlayedThisTurn) {
    return [...trades, ...develops, { type: 'end-turn' as const }];
  }

  return [...trades, ...develops, ...playActions(state)];
}

function tradeActions(
  state: GameState
): Extract<GameAction, { type: 'trade' }>[] {
  const player = state.players[state.activePlayerIndex];
  return SUITS.flatMap((give) =>
    player.resources[give] < 3
      ? []
      : SUITS.filter((receive) => receive !== give).map((receive) => ({
          type: 'trade' as const,
          give,
          receive,
        }))
  );
}

function developActions(
  state: GameState
): Extract<GameAction, { type: 'develop-deed' }>[] {
  const player = state.players[state.activePlayerIndex];
  const playerId = player.id;

  return state.districts.flatMap(
    (district) => {
      const deed = district.stacks[playerId]?.deed;
      if (!deed) {
        return [];
      }

      const card = CARD_BY_ID[deed.cardId];
      if (!card || card.kind !== 'Property') {
        return [];
      }
      if (deed.progress >= developmentCost(card)) {
        return [];
      }

      return card.suits
        .filter((suit) => player.resources[suit] > 0)
        .map((suit) => ({
          type: 'develop-deed' as const,
          districtId: district.id,
          cardId: deed.cardId,
          tokens: { [suit]: 1 },
        }));
    }
  );
}

function collectIncomeChoiceActions(state: GameState): GameAction[] {
  const [choice] = state.pendingIncomeChoices ?? [];
  if (!choice) {
    return [];
  }

  const activePlayer = state.players[state.activePlayerIndex];
  if (!activePlayer || activePlayer.id !== choice.playerId) {
    return [];
  }

  return choice.suits.map((suit) => ({
    type: 'choose-income-suit' as const,
    playerId: choice.playerId,
    districtId: choice.districtId,
    cardId: choice.cardId,
    suit,
  }));
}

function playActions(state: GameState): GameAction[] {
  if (state.cardPlayedThisTurn) {
    return [];
  }

  const player = state.players[state.activePlayerIndex];
  const playerId = player.id;

  return player.hand.flatMap((cardId) => {
    const card = CARD_BY_ID[cardId];
    if (!card || card.kind !== 'Property') {
      return [];
    }

    const sell = { type: 'sell-card' as const, cardId };
    const placements = state.districts.filter((district) =>
      placementAllowed(card, district, playerId)
    );

    const canBuyDeed = canAfford(player.resources, deedCost(card));
    const deedable = canBuyDeed
      ? placements
          .filter((district) => !district.stacks[playerId]?.deed)
          .map((district) => ({
            type: 'buy-deed' as const,
            cardId,
            districtId: district.id,
          }))
      : [];

    const developOutright = placements.flatMap((district) =>
      enumerateOutrightPayments(card, player.resources).map((payment) => ({
        type: 'develop-outright' as const,
        cardId,
        districtId: district.id,
        payment,
      }))
    );

    return [sell, ...deedable, ...developOutright];
  });
}
