import { CARD_BY_ID } from './cards';
import {
  SUITS,
  canAfford,
  deedCost,
  developmentCost,
  enumerateOutrightPayments,
  placementAllowed,
} from './stateHelpers';
import type { GameState, GameAction, GamePhase } from './types';

type PhaseBuilderMap = Partial<
  Record<GamePhase, (state: GameState) => GameAction[]>
>;

const builders: PhaseBuilderMap = {
  OptionalTrade: tradeActions,
  OptionalDevelop: developActions,
  PlayCard: playActions,
};

export function legalActions(state: GameState): readonly GameAction[] {
  const phaseBuilder = builders[state.phase];
  return phaseBuilder ? phaseBuilder(state) : [];
}

function tradeActions(state: GameState): GameAction[] {
  const player = state.players[state.activePlayerIndex];
  return SUITS.flatMap((give) =>
    player.resources[give] < 3
      ? []
      : SUITS.filter((receive) => receive !== give).map((receive) => ({
          type: 'trade',
          give,
          receive,
        }))
  );
}

function developActions(state: GameState): GameAction[] {
  const playerId = state.players[state.activePlayerIndex].id;
  const player = state.players[state.activePlayerIndex];
  return state.districts.flatMap((district) => {
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
        type: 'develop-deed',
        districtId: district.id,
        cardId: deed.cardId,
        tokens: { [suit]: 1 },
      }));
  });
}

function playActions(state: GameState): GameAction[] {
  const player = state.players[state.activePlayerIndex];
  const playerId = player.id;
  return player.hand.flatMap((cardId) => {
    const card = CARD_BY_ID[cardId];
    if (!card) {
      return [];
    }
    if (card.kind !== 'Property') {
      return [];
    }
    const sell = { type: 'sell-card' as const, cardId };
    const placements = state.districts.filter((district) =>
      placementAllowed(card, district, playerId)
    );
    const canBuyDeed = canAfford(player.resources, deedCost(card));
    const deedable = placements
      .filter((district) => !district.stacks[playerId]?.deed)
      .filter(() => canBuyDeed)
      .map((district) => ({
        type: 'buy-deed' as const,
        cardId,
        districtId: district.id,
      }));

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
