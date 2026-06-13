import type { CardId } from '../../engine/cards';
import type {
  GameAction,
  GameState,
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  ResourcePool,
  Suit,
} from '../../engine/types';

export type ActionResourcePaymentReason =
  | 'buy-deed'
  | 'develop-outright'
  | 'develop-deed';

export type RuntimeMode =
  | { type: 'idle' }
  | { type: 'animating'; transactionId: string; elapsedMs: number }
  | { type: 'awaiting-input'; actorId: PlayerId };

export type AnimationOverlayState = {
  incomeHighlightCardIds: readonly CardId[];
  incomeHighlightCrowns: readonly { playerId: PlayerId; suit: Suit }[];
  activePlayerHighlightOverride: PlayerId | null;
};

export type InteractionState = {
  legalActions: readonly GameAction[];
  acceptingInput: boolean;
};

export type GameRuntimeSnapshot = {
  viewState: GameState;
  overlays: AnimationOverlayState;
  interaction: InteractionState;
  mode: RuntimeMode;
};

export type IncomeTokenSource =
  | {
      kind: 'district-card';
      cardId: CardId;
      districtId: string;
    }
  | {
      kind: 'crown';
      cardId: CardId;
    }
  | {
      kind: 'income-choice';
      cardId: CardId;
      districtId: string;
    };

export type GamePresentationEvent =
  | {
      type: 'action-started';
      action: GameAction;
      actingPlayerId: PlayerId;
    }
  | {
      type: 'draw-card';
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: 'card-sold';
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: 'sell-resource-gained';
      playerId: PlayerId;
      cardId: CardId;
      suit: Suit;
      tokenIndex: number;
    }
  | {
      type: 'resource-payment-started';
      playerId: PlayerId;
      reason: ActionResourcePaymentReason;
      cardId: CardId;
      districtId: string;
      payment: Partial<Record<Suit, number>>;
    }
  | {
      type: 'resource-payment-applied';
      playerId: PlayerId;
      reason: ActionResourcePaymentReason;
      cardId: CardId;
      districtId: string;
      payment: Partial<Record<Suit, number>>;
    }
  | {
      type: 'card-played-to-district';
      playerId: PlayerId;
      cardId: CardId;
      districtId: string;
      placement: 'deed' | 'developed';
    }
  | {
      type: 'deed-token-paid';
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      suit: Suit;
      tokenIndex: number;
    }
  | {
      type: 'deed-progress-applied';
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      previousProgress: number;
      nextProgress: number;
      targetProgress: number;
      completed: boolean;
    }
  | {
      type: 'deed-completed';
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
    }
  | {
      type: 'trade-resources-applied';
      playerId: PlayerId;
      give: Suit;
      receive: Suit;
      giveCount: number;
      receiveCount: number;
    }
  | {
      type: 'income-roll';
      playerId: PlayerId;
      roll: IncomeRollResult;
      incomeRank: number;
    }
  | {
      type: 'tax-token-lost';
      playerId: PlayerId;
      suit: Suit;
      tokenIndex: number;
    }
  | {
      type: 'income-token-gained';
      playerId: PlayerId;
      suit: Suit;
      source: IncomeTokenSource;
    }
  | {
      type: 'income-choice-required';
      choices: readonly IncomeChoice[];
    }
  | {
      type: 'income-choice-submitted';
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      suit: Suit;
    }
  | {
      type: 'active-player-changed';
      previousPlayerId: PlayerId | null;
      nextPlayerId: PlayerId | null;
    }
  | {
      type: 'phase-changed';
      previousPhase: GameState['phase'];
      nextPhase: GameState['phase'];
    }
  | {
      type: 'transaction-settled';
    };

export type GameTransaction = {
  id: string;
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  events: readonly GamePresentationEvent[];
};

export function cloneResourcePool(resources: ResourcePool): ResourcePool {
  return { ...resources };
}
