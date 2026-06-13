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

export type PresentationTimelineEvent =
  | {
      atMs: number;
      type: 'hold-previous-state';
    }
  | {
      atMs: number;
      type: 'reveal-drawn-card';
      event: Extract<GamePresentationEvent, { type: 'draw-card' }>;
    }
  | {
      atMs: number;
      type: 'stage-sold-card';
      event: Extract<GamePresentationEvent, { type: 'card-sold' }>;
    }
  | {
      atMs: number;
      type: 'show-income-roll';
      event: Extract<GamePresentationEvent, { type: 'income-roll' }>;
    }
  | {
      atMs: number;
      type: 'apply-tax-token-loss';
      event: Extract<GamePresentationEvent, { type: 'tax-token-lost' }>;
    }
  | {
      atMs: number;
      type: 'launch-income-token-flight';
      event: Extract<GamePresentationEvent, { type: 'income-token-gained' }>;
    }
  | {
      atMs: number;
      type: 'show-income-highlights';
      cardIds: readonly CardId[];
      crowns: readonly { playerId: PlayerId; suit: Suit }[];
    }
  | {
      atMs: number;
      type: 'clear-income-highlights';
    }
  | {
      atMs: number;
      type: 'apply-income-token-gain';
      event: Extract<GamePresentationEvent, { type: 'income-token-gained' }>;
    }
  | {
      atMs: number;
      type: 'reveal-income-choice-request';
      event: Extract<GamePresentationEvent, { type: 'income-choice-required' }>;
    }
  | {
      atMs: number;
      type: 'reveal-income-choice-submission';
      event: Extract<
        GamePresentationEvent,
        { type: 'income-choice-submitted' }
      >;
    }
  | {
      atMs: number;
      type: 'commit-view-to-next-state';
    };

export type PresentationTimeline = {
  transactionId: string;
  durationMs: number;
  events: readonly PresentationTimelineEvent[];
};

export function cloneResourcePool(resources: ResourcePool): ResourcePool {
  return { ...resources };
}
