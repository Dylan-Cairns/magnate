import {
  RuntimeModeType,
  DicePhase,
  IncomeTokenSourceKind,
  GamePresentationEventType,
} from './values';
import { ActionId } from '../../engine/values';
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
  | typeof ActionId.BuyDeed
  | typeof ActionId.DevelopOutright
  | typeof ActionId.DevelopDeed;

export type RuntimeMode =
  | { type: typeof RuntimeModeType.Idle }
  | {
      type: typeof RuntimeModeType.Animating;
      transactionId: string;
      elapsedMs: number;
    }
  | { type: typeof RuntimeModeType.AwaitingInput; actorId: PlayerId };

export type TradeProgress = {
  transactionId: string;
  playerId: PlayerId;
  suit: Suit;
  landed: number;
  total: number;
};

export type AnimationOverlayState = {
  tradeProgress?: TradeProgress;
  incomeHighlightCardIds: readonly CardId[];
  incomeHighlightCrowns: readonly { playerId: PlayerId; suit: Suit }[];
  activePlayerHighlightOverride: PlayerId | null;
  dice: DiceVisualState | null;
};

export type IncomeDicePhase =
  | typeof DicePhase.Rolling
  | typeof DicePhase.Settled;
export type TaxDicePhase =
  | typeof DicePhase.Hidden
  | typeof DicePhase.Rolling
  | typeof DicePhase.Settled
  | typeof DicePhase.Dimmed;

export type DiceVisualState = {
  incomeRoll: IncomeRollResult;
  taxSuit: Suit | undefined;
  incomePhase: IncomeDicePhase;
  taxPhase: TaxDicePhase;
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
      kind: typeof IncomeTokenSourceKind.DistrictCard;
      cardId: CardId;
      districtId: string;
    }
  | {
      kind: typeof IncomeTokenSourceKind.Crown;
      cardId: CardId;
    }
  | {
      kind: typeof IncomeTokenSourceKind.IncomeChoice;
      cardId: CardId;
      districtId: string;
    };

export type GamePresentationEvent =
  | {
      type: typeof GamePresentationEventType.ActionStarted;
      action: GameAction;
      actingPlayerId: PlayerId;
    }
  | {
      type: typeof GamePresentationEventType.DrawCard;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: typeof GamePresentationEventType.CardSold;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      type: typeof GamePresentationEventType.SellResourceGained;
      playerId: PlayerId;
      cardId: CardId;
      suit: Suit;
      tokenIndex: number;
    }
  | {
      type: typeof GamePresentationEventType.ResourcePaymentStarted;
      playerId: PlayerId;
      reason: ActionResourcePaymentReason;
      cardId: CardId;
      districtId: string;
      payment: Partial<Record<Suit, number>>;
    }
  | {
      type: typeof GamePresentationEventType.ResourcePaymentApplied;
      playerId: PlayerId;
      reason: ActionResourcePaymentReason;
      cardId: CardId;
      districtId: string;
      payment: Partial<Record<Suit, number>>;
    }
  | {
      type: typeof GamePresentationEventType.CardPlayedToDistrict;
      playerId: PlayerId;
      cardId: CardId;
      districtId: string;
      placement: 'deed' | 'developed';
    }
  | {
      type: typeof GamePresentationEventType.DeedTokenPaid;
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      suit: Suit;
      tokenIndex: number;
      previousTokens: Partial<Record<Suit, number>>;
      nextTokens: Partial<Record<Suit, number>>;
    }
  | {
      type: typeof GamePresentationEventType.DeedProgressApplied;
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      previousProgress: number;
      nextProgress: number;
      targetProgress: number;
      completed: boolean;
    }
  | {
      type: typeof GamePresentationEventType.DeedCompleted;
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
    }
  | {
      type: typeof GamePresentationEventType.TradeResourcesApplied;
      playerId: PlayerId;
      give: Suit;
      receive: Suit;
      giveCount: number;
      receiveCount: number;
    }
  | {
      type: typeof GamePresentationEventType.IncomeRoll;
      playerId: PlayerId;
      turn: number;
      roll: IncomeRollResult;
      incomeRank: number;
    }
  | {
      type: typeof GamePresentationEventType.TaxResolved;
      suit: Suit;
    }
  | {
      type: typeof GamePresentationEventType.TaxTokenLost;
      playerId: PlayerId;
      suit: Suit;
      tokenIndex: number;
    }
  | {
      type: typeof GamePresentationEventType.IncomeTokenGained;
      playerId: PlayerId;
      suit: Suit;
      source: IncomeTokenSource;
    }
  | {
      type: typeof GamePresentationEventType.IncomeChoiceRequired;
      choices: readonly IncomeChoice[];
      returnPlayerId: PlayerId | undefined;
    }
  | {
      type: typeof GamePresentationEventType.IncomeChoiceSubmitted;
      playerId: PlayerId;
      districtId: string;
      cardId: CardId;
      suit: Suit;
    }
  | {
      type: typeof GamePresentationEventType.ActivePlayerChanged;
      previousPlayerId: PlayerId | null;
      nextPlayerId: PlayerId | null;
    }
  | {
      type: typeof GamePresentationEventType.PhaseChanged;
      previousPhase: GameState['phase'];
      nextPhase: GameState['phase'];
    }
  | {
      type: typeof GamePresentationEventType.TransactionSettled;
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
