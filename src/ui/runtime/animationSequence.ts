import { ActionId } from '../../engine/values';
import {
  AnimationStepType,
  GamePresentationEventType,
  IncomeTokenSourceKind,
} from './values';
import type { CardId } from '../../engine/cards';
import { SUITS } from '../../engine/stateHelpers';
import type {
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  Suit,
} from '../../engine/types';
import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  CARD_FLIGHT_DURATION_MS,
  DEED_PROGRESS_REVEAL_MS,
  PAYMENT_FLIGHT_DURATION_MS,
  PAYMENT_FLIGHT_STAGGER_MS,
  RESOURCE_FLIGHT_DURATION_MS,
  RESOURCE_FLIGHT_STAGGER_MS,
  SELL_FLIGHT_DURATION_MS,
  SELL_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from '../animations/timing';
import type { GamePresentationEvent, GameTransaction } from './types';

export type AnimationDurations = {
  cardFlightMs: number;
  commitBufferMs: number;
  actionResourceFlightMs: number;
  actionResourceFlightStaggerMs: number;
  paymentFlightMs: number;
  paymentFlightStaggerMs: number;
  sellFlightMs: number;
  sellFlightStaggerMs: number;
  dieRollMs: number;
  taxDieRollMs: number;
  taxPreFlightHoldMs: number;
  taxFlightMs: number;
  taxFlightStaggerMs: number;
  stageGapMs: number;
  incomePreFlightHoldMs: number;
  incomeFlightMs: number;
  incomeFlightStaggerMs: number;
  postIncomeHoldMs: number;
  deedProgressRevealMs: number;
};

export const DEFAULT_ANIMATION_DURATIONS: AnimationDurations = {
  cardFlightMs: CARD_FLIGHT_DURATION_MS,
  commitBufferMs: ACTION_FLIGHT_COMMIT_BUFFER_MS,
  actionResourceFlightMs: RESOURCE_FLIGHT_DURATION_MS,
  actionResourceFlightStaggerMs: RESOURCE_FLIGHT_STAGGER_MS,
  paymentFlightMs: PAYMENT_FLIGHT_DURATION_MS,
  paymentFlightStaggerMs: PAYMENT_FLIGHT_STAGGER_MS,
  sellFlightMs: SELL_FLIGHT_DURATION_MS,
  sellFlightStaggerMs: SELL_FLIGHT_STAGGER_MS,
  dieRollMs: 1000,
  taxDieRollMs: 1000,
  taxPreFlightHoldMs: 550,
  taxFlightMs: TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  taxFlightStaggerMs: TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
  stageGapMs: 220,
  incomePreFlightHoldMs: 400,
  incomeFlightMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  incomeFlightStaggerMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  postIncomeHoldMs: 220,
  deedProgressRevealMs: DEED_PROGRESS_REVEAL_MS,
};

export type AnimationStep =
  | {
      id: string;
      type: typeof AnimationStepType.HoldPreviousState;
      durationMs: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.DrawCardFlight;
      durationMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      id: string;
      type: typeof AnimationStepType.StageSoldCard;
      durationMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchSellTokenFlights;
      durationMs: number;
      flightSequenceDurationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.SellResourceGained }
      >[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.LandSellToken;
      durationMs: number;
      gain: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.SellResourceGained }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchPaymentTokenFlights;
      durationMs: number;
      flightSequenceDurationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.ResourcePaymentStarted }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyResourcePaymentToken;
      durationMs: number;
      playerId: PlayerId;
      suit: Suit;
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyResourcePayment;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.ResourcePaymentApplied }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchCardToDistrictFlight;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.CardPlayedToDistrict }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.PlaceCardInDistrict;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.CardPlayedToDistrict }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchDeedTokenFlights;
      durationMs: number;
      tokens: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.DeedTokenPaid }
      >[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyDeedTokens;
      durationMs: number;
      tokens: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.DeedTokenPaid }
      >[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyDeedProgress;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.DeedProgressApplied }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.RevealDeedCompletion;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.DeedCompleted }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchTradeTokenFlights;
      durationMs: number;
      flightSequenceDurationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TradeResourcesApplied }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyTradeTokenLoss;
      durationMs: number;
      playerId: PlayerId;
      suit: Suit;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LandTradeToken;
      durationMs: number;
      landed: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TradeResourcesApplied }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyTradeTokenGain;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TradeResourcesApplied }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.RollIncomeDice;
      durationMs: number;
      playerId: PlayerId;
      turn: number;
      roll: IncomeRollResult;
      incomeRank: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.RollTaxDie;
      durationMs: number;
      suit: Suit;
    }
  | {
      id: string;
      type: typeof AnimationStepType.HoldBeforeTaxFlights;
      durationMs: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchTaxTokenFlights;
      durationMs: number;
      flightSequenceDurationMs: number;
      losses: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TaxTokenLost }
      >[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.ApplyTaxTokenLoss;
      durationMs: number;
      loss: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.TaxTokenLost }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.StageGap;
      durationMs: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.HoldBeforeIncomeFlights;
      durationMs: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.HighlightIncomeSources;
      durationMs: number;
      cardIds: readonly CardId[];
      crowns: readonly { playerId: PlayerId; suit: Suit }[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.LaunchIncomeTokenFlights;
      durationMs: number;
      flightSequenceDurationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.IncomeTokenGained }
      >[];
    }
  | {
      id: string;
      type: typeof AnimationStepType.LandIncomeToken;
      durationMs: number;
      gain: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.IncomeTokenGained }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.PostIncomeHold;
      durationMs: number;
    }
  | {
      id: string;
      type: typeof AnimationStepType.RevealIncomeChoiceRequest;
      durationMs: number;
      choices: readonly IncomeChoice[];
      returnPlayerId: PlayerId | undefined;
    }
  | {
      id: string;
      type: typeof AnimationStepType.RevealIncomeChoiceSubmission;
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.IncomeChoiceSubmitted }
      >;
    }
  | {
      id: string;
      type: typeof AnimationStepType.CommitViewState;
      durationMs: number;
    };

export type ScheduledAnimationStep = AnimationStep & {
  startMs: number;
  endMs: number;
};

export type AnimationSequence = {
  transactionId: string;
  durationMs: number;
  commitMs: number;
  inputUnlockMs: number;
  steps: readonly ScheduledAnimationStep[];
};

export function buildAnimationSequence(
  transaction: GameTransaction,
  durations: AnimationDurations = DEFAULT_ANIMATION_DURATIONS
): AnimationSequence {
  const steps: AnimationStep[] = [
    {
      id: AnimationStepType.HoldPreviousState,
      type: AnimationStepType.HoldPreviousState,
      durationMs: 0,
    },
  ];
  const drawEvent = firstEvent(transaction, GamePresentationEventType.DrawCard);
  const incomeChoiceSubmissions = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.IncomeChoiceSubmitted }
    > => event.type === GamePresentationEventType.IncomeChoiceSubmitted
  );
  const deferIncomeChoiceSubmission = transaction.events.some(
    (event) =>
      event.type === GamePresentationEventType.IncomeTokenGained &&
      event.source.kind === IncomeTokenSourceKind.IncomeChoice
  );
  if (drawEvent) {
    steps.push({
      id: `draw-card-flight:${drawEvent.playerId}:${drawEvent.cardId}`,
      type: AnimationStepType.DrawCardFlight,
      durationMs: durations.cardFlightMs + durations.commitBufferMs,
      playerId: drawEvent.playerId,
      cardId: drawEvent.cardId,
    });
  }

  appendSellSteps(transaction, steps, durations);

  for (const event of transaction.events) {
    if (
      event.type === GamePresentationEventType.IncomeChoiceSubmitted &&
      !deferIncomeChoiceSubmission
    ) {
      appendIncomeChoiceSubmissionStep(steps, event);
    }
  }

  appendCardPlacementSteps(transaction, steps, durations);
  appendActionResourcePaymentSteps(transaction, steps, durations);
  appendTradeSteps(transaction, steps, durations);
  appendDeedDevelopmentSteps(transaction, steps, durations);

  const incomeRoll = firstEvent(
    transaction,
    GamePresentationEventType.IncomeRoll
  );
  if (incomeRoll) {
    steps.push({
      id: `roll-income-dice:${incomeRoll.roll.rollId ?? `${incomeRoll.roll.die1}-${incomeRoll.roll.die2}`}`,
      type: AnimationStepType.RollIncomeDice,
      durationMs: durations.dieRollMs,
      playerId: incomeRoll.playerId,
      turn: incomeRoll.turn,
      roll: incomeRoll.roll,
      incomeRank: incomeRoll.incomeRank,
    });
  }

  const taxLosses = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.TaxTokenLost }
    > => event.type === GamePresentationEventType.TaxTokenLost
  );
  const taxResolved = firstEvent(
    transaction,
    GamePresentationEventType.TaxResolved
  );
  const taxSuit = taxResolved?.suit ?? taxLosses[0]?.suit;
  if (taxSuit) {
    steps.push({
      id: `roll-tax-die:${taxSuit}`,
      type: AnimationStepType.RollTaxDie,
      durationMs: durations.taxDieRollMs,
      suit: taxSuit,
    });
  }
  if (taxLosses.length > 0) {
    const flightSequenceDurationMs = staggeredDuration(
      taxLosses.length,
      durations.taxFlightMs,
      durations.taxFlightStaggerMs
    );
    steps.push({
      id: AnimationStepType.HoldBeforeTaxFlights,
      type: AnimationStepType.HoldBeforeTaxFlights,
      durationMs: durations.taxPreFlightHoldMs,
    });
    steps.push({
      id: AnimationStepType.LaunchTaxTokenFlights,
      type: AnimationStepType.LaunchTaxTokenFlights,
      durationMs: 0,
      flightSequenceDurationMs,
      losses: taxLosses,
    });
    taxLosses.forEach((loss, index) => {
      const isLastLoss = index === taxLosses.length - 1;
      steps.push({
        id: `apply-tax-token-loss:${loss.playerId}:${loss.suit}:${String(loss.tokenIndex)}`,
        type: AnimationStepType.ApplyTaxTokenLoss,
        durationMs: isLastLoss
          ? durations.taxFlightMs
          : durations.taxFlightStaggerMs,
        loss,
      });
    });
  }

  const incomeGains = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.IncomeTokenGained }
    > => event.type === GamePresentationEventType.IncomeTokenGained
  );
  if (
    (taxSuit || taxLosses.length > 0) &&
    (incomeGains.length > 0 ||
      hasEvent(transaction, GamePresentationEventType.IncomeChoiceRequired))
  ) {
    steps.push({
      id: AnimationStepType.StageGap,
      type: AnimationStepType.StageGap,
      durationMs: durations.stageGapMs,
    });
  }
  if (incomeGains.length > 0) {
    const targets = highlightTargetsForIncomeEvents(incomeGains);
    const flightSequenceDurationMs = staggeredDuration(
      incomeGains.length,
      durations.incomeFlightMs,
      durations.incomeFlightStaggerMs
    );
    steps.push({
      id: AnimationStepType.HoldBeforeIncomeFlights,
      type: AnimationStepType.HoldBeforeIncomeFlights,
      durationMs: durations.incomePreFlightHoldMs,
    });
    if (deferIncomeChoiceSubmission) {
      for (const submission of incomeChoiceSubmissions) {
        appendIncomeChoiceSubmissionStep(steps, submission);
      }
    }
    if (targets.cardIds.length > 0 || targets.crowns.length > 0) {
      steps.push({
        id: AnimationStepType.HighlightIncomeSources,
        type: AnimationStepType.HighlightIncomeSources,
        durationMs: 0,
        cardIds: targets.cardIds,
        crowns: targets.crowns,
      });
    }
    steps.push({
      id: AnimationStepType.LaunchIncomeTokenFlights,
      type: AnimationStepType.LaunchIncomeTokenFlights,
      durationMs: 0,
      flightSequenceDurationMs,
      flightDurationMs: durations.incomeFlightMs,
      flightStaggerMs: durations.incomeFlightStaggerMs,
      gains: incomeGains,
    });
    incomeGains.forEach((gain, index) => {
      steps.push({
        id: `land-income-token:${gain.playerId}:${gain.suit}:${String(index)}`,
        type: AnimationStepType.LandIncomeToken,
        durationMs:
          index === 0
            ? durations.incomeFlightMs
            : durations.incomeFlightStaggerMs,
        gain,
      });
    });
    steps.push({
      id: AnimationStepType.PostIncomeHold,
      type: AnimationStepType.PostIncomeHold,
      durationMs: durations.postIncomeHoldMs,
    });
  }

  for (const event of transaction.events) {
    if (event.type === GamePresentationEventType.IncomeChoiceRequired) {
      steps.push({
        id: AnimationStepType.RevealIncomeChoiceRequest,
        type: AnimationStepType.RevealIncomeChoiceRequest,
        durationMs: 0,
        choices: event.choices,
        returnPlayerId: event.returnPlayerId,
      });
    }
  }

  steps.push({
    id: AnimationStepType.CommitViewState,
    type: AnimationStepType.CommitViewState,
    durationMs: durations.commitBufferMs,
  });

  return scheduleSteps(transaction.id, steps);
}

function scheduleSteps(
  transactionId: string,
  steps: readonly AnimationStep[]
): AnimationSequence {
  let cursorMs = 0;
  const scheduled = steps.map((step) => {
    const scheduledStep = {
      ...step,
      startMs: cursorMs,
      endMs: cursorMs + step.durationMs,
    };
    cursorMs = scheduledStep.endMs;
    return scheduledStep;
  });
  return {
    transactionId,
    durationMs: cursorMs,
    commitMs:
      scheduled.find((step) => step.type === AnimationStepType.CommitViewState)
        ?.startMs ?? cursorMs,
    inputUnlockMs:
      scheduled.find((step) => step.type === AnimationStepType.CommitViewState)
        ?.startMs ?? cursorMs,
    steps: scheduled,
  };
}

function staggeredDuration(
  count: number,
  durationMs: number,
  staggerMs: number
): number {
  if (count <= 0) {
    return 0;
  }
  return (count - 1) * staggerMs + durationMs;
}

function appendActionResourcePaymentSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const starts = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.ResourcePaymentStarted }
    > =>
      event.type === GamePresentationEventType.ResourcePaymentStarted &&
      event.reason !== ActionId.DevelopDeed
  );
  for (const start of starts) {
    const paymentTokens = SUITS.flatMap((suit) =>
      Array.from({ length: start.payment[suit] ?? 0 }, () => suit)
    );
    const flightSequenceDurationMs = staggeredDuration(
      paymentTokens.length,
      durations.paymentFlightMs,
      durations.paymentFlightStaggerMs
    );
    steps.push({
      id: `launch-payment-token-flights:${start.reason}:${start.playerId}:${start.cardId}:${start.districtId}`,
      type: AnimationStepType.LaunchPaymentTokenFlights,
      durationMs: 0,
      flightSequenceDurationMs,
      flightDurationMs: durations.paymentFlightMs,
      flightStaggerMs: durations.paymentFlightStaggerMs,
      event: start,
    });
    const apply = transaction.events.find(
      (
        event
      ): event is Extract<
        GamePresentationEvent,
        { type: typeof GamePresentationEventType.ResourcePaymentApplied }
      > =>
        event.type === GamePresentationEventType.ResourcePaymentApplied &&
        event.reason === start.reason &&
        event.playerId === start.playerId &&
        event.cardId === start.cardId &&
        event.districtId === start.districtId
    );
    if (apply) {
      paymentTokens.forEach((suit, index) => {
        const isLastToken = index === paymentTokens.length - 1;
        steps.push({
          id: `apply-resource-payment-token:${apply.reason}:${apply.playerId}:${apply.cardId}:${apply.districtId}:${suit}:${String(index)}`,
          type: AnimationStepType.ApplyResourcePaymentToken,
          durationMs: isLastToken
            ? durations.paymentFlightMs + durations.commitBufferMs
            : durations.paymentFlightStaggerMs,
          playerId: apply.playerId,
          suit,
        });
      });
    }
  }
}

function appendTradeSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const trade = firstEvent(
    transaction,
    GamePresentationEventType.TradeResourcesApplied
  );
  if (!trade) {
    return;
  }

  steps.push({
    id: `launch-trade-token-flights:${trade.playerId}:${trade.give}:${trade.receive}`,
    type: AnimationStepType.LaunchTradeTokenFlights,
    durationMs: 0,
    flightSequenceDurationMs: staggeredDuration(
      trade.giveCount,
      durations.paymentFlightMs,
      durations.paymentFlightStaggerMs
    ),
    flightDurationMs: durations.paymentFlightMs,
    flightStaggerMs: durations.paymentFlightStaggerMs,
    event: trade,
  });
  // Merge launches and arrivals so overlapping flights retain explicit boundaries.
  const boundaries: { atMs: number; step: AnimationStep }[] = [];
  for (let index = 0; index < trade.giveCount; index += 1) {
    const launchMs = index * durations.paymentFlightStaggerMs;
    boundaries.push({
      atMs: launchMs,
      step: {
        id: `apply-trade-token-loss:${trade.playerId}:${trade.give}:${index}`,
        type: AnimationStepType.ApplyTradeTokenLoss,
        durationMs: 0,
        playerId: trade.playerId,
        suit: trade.give,
      },
    });
    boundaries.push({
      atMs: launchMs + durations.paymentFlightMs,
      step: {
        id: `land-trade-token:${trade.playerId}:${index}`,
        type: AnimationStepType.LandTradeToken,
        durationMs: 0,
        landed: index + 1,
        event: trade,
      },
    });
  }
  boundaries.sort((a, b) => a.atMs - b.atMs);
  boundaries.forEach((boundary, index) => {
    steps.push({
      ...boundary.step,
      durationMs:
        (boundaries[index + 1]?.atMs ?? boundary.atMs) - boundary.atMs,
    });
  });
  steps.push({
    id: `apply-trade-token-gain:${trade.playerId}:${trade.receive}`,
    type: AnimationStepType.ApplyTradeTokenGain,
    durationMs: durations.commitBufferMs,
    event: trade,
  });
}

function appendCardPlacementSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const placements = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.CardPlayedToDistrict }
    > => event.type === GamePresentationEventType.CardPlayedToDistrict
  );
  for (const event of placements) {
    steps.push(
      {
        id: `launch-card-to-district-flight:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: AnimationStepType.LaunchCardToDistrictFlight,
        // The placement commit starts at this step's end, so the flight needs
        // the same settle buffer the draw flight has: without it the animation
        // is cut off before its final frame and the card pops to full size.
        durationMs: durations.cardFlightMs + durations.commitBufferMs,
        event,
      },
      {
        id: `place-card-in-district:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: AnimationStepType.PlaceCardInDistrict,
        durationMs: 0,
        event,
      }
    );
  }
}

function appendDeedDevelopmentSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const deedPayment = transaction.events.find(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.ResourcePaymentApplied }
    > =>
      event.type === GamePresentationEventType.ResourcePaymentApplied &&
      event.reason === ActionId.DevelopDeed
  );
  if (deedPayment) {
    steps.push({
      id: `apply-resource-payment:${deedPayment.reason}:${deedPayment.playerId}:${deedPayment.cardId}:${deedPayment.districtId}`,
      type: AnimationStepType.ApplyResourcePayment,
      durationMs: 0,
      event: deedPayment,
    });
  }

  const deedTokens = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.DeedTokenPaid }
    > => event.type === GamePresentationEventType.DeedTokenPaid
  );
  if (deedTokens.length > 0) {
    steps.push({
      id: AnimationStepType.LaunchDeedTokenFlights,
      type: AnimationStepType.LaunchDeedTokenFlights,
      durationMs: staggeredDuration(
        deedTokens.length,
        durations.actionResourceFlightMs,
        durations.actionResourceFlightStaggerMs
      ),
      tokens: deedTokens,
    });
    steps.push({
      id: AnimationStepType.ApplyDeedTokens,
      type: AnimationStepType.ApplyDeedTokens,
      durationMs: durations.commitBufferMs,
      tokens: deedTokens,
    });
  }

  for (const event of transaction.events) {
    if (event.type === GamePresentationEventType.DeedProgressApplied) {
      steps.push({
        id: `apply-deed-progress:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: AnimationStepType.ApplyDeedProgress,
        durationMs: durations.deedProgressRevealMs,
        event,
      });
    }
    if (event.type === GamePresentationEventType.DeedCompleted) {
      steps.push({
        id: `reveal-deed-completion:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: AnimationStepType.RevealDeedCompletion,
        durationMs: 0,
        event,
      });
    }
  }
}

function appendSellSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const sold = firstEvent(transaction, GamePresentationEventType.CardSold);
  if (!sold) {
    return;
  }

  const gains = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: typeof GamePresentationEventType.SellResourceGained }
    > => event.type === GamePresentationEventType.SellResourceGained
  );
  if (gains.length > 0) {
    steps.push({
      id: `launch-sell-token-flights:${sold.playerId}:${sold.cardId}`,
      type: AnimationStepType.LaunchSellTokenFlights,
      durationMs: 0,
      flightSequenceDurationMs: staggeredDuration(
        gains.length,
        durations.sellFlightMs,
        durations.sellFlightStaggerMs
      ),
      flightDurationMs: durations.sellFlightMs,
      flightStaggerMs: durations.sellFlightStaggerMs,
      gains,
    });
    gains.forEach((gain, index) => {
      steps.push({
        id: `land-sell-token:${gain.playerId}:${gain.suit}:${String(gain.tokenIndex)}`,
        type: AnimationStepType.LandSellToken,
        durationMs:
          index === 0 ? durations.sellFlightMs : durations.sellFlightStaggerMs,
        gain,
      });
    });
  }

  steps.push({
    id: `stage-sold-card:${sold.playerId}:${sold.cardId}`,
    type: AnimationStepType.StageSoldCard,
    durationMs: durations.cardFlightMs + durations.commitBufferMs,
    playerId: sold.playerId,
    cardId: sold.cardId,
  });
}

function firstEvent<TType extends GamePresentationEvent['type']>(
  transaction: GameTransaction,
  type: TType
): Extract<GamePresentationEvent, { type: TType }> | undefined {
  return transaction.events.find(
    (event): event is Extract<GamePresentationEvent, { type: TType }> =>
      event.type === type
  );
}

function hasEvent<TType extends GamePresentationEvent['type']>(
  transaction: GameTransaction,
  type: TType
): boolean {
  return transaction.events.some((event) => event.type === type);
}

function highlightTargetsForIncomeEvents(
  incomeEvents: readonly Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.IncomeTokenGained }
  >[]
): {
  cardIds: readonly CardId[];
  crowns: readonly { playerId: PlayerId; suit: Suit }[];
} {
  const cardIds: CardId[] = [];
  const crownTargets: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const seenCardIds = new Set<CardId>();
  const seenCrowns = new Set<string>();
  for (const event of incomeEvents) {
    if (event.source.kind === IncomeTokenSourceKind.IncomeChoice) {
      continue;
    }
    if (event.source.kind === IncomeTokenSourceKind.Crown) {
      const key = `${event.playerId}:${event.suit}`;
      if (!seenCrowns.has(key)) {
        seenCrowns.add(key);
        crownTargets.push({ playerId: event.playerId, suit: event.suit });
      }
      continue;
    }
    if (!seenCardIds.has(event.source.cardId)) {
      seenCardIds.add(event.source.cardId);
      cardIds.push(event.source.cardId);
    }
  }
  return { cardIds, crowns: crownTargets };
}

function appendIncomeChoiceSubmissionStep(
  steps: AnimationStep[],
  event: Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.IncomeChoiceSubmitted }
  >
): void {
  steps.push({
    id: `reveal-income-choice-submission:${event.playerId}:${event.districtId}:${event.cardId}`,
    type: AnimationStepType.RevealIncomeChoiceSubmission,
    durationMs: 0,
    event,
  });
}
