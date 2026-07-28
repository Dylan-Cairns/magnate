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
      type: 'hold-previous-state';
      durationMs: number;
    }
  | {
      id: string;
      type: 'draw-card-flight';
      durationMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      id: string;
      type: 'stage-sold-card';
      durationMs: number;
      playerId: PlayerId;
      cardId: CardId;
    }
  | {
      id: string;
      type: 'apply-sell-resource-gains';
      durationMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: 'sell-resource-gained' }
      >[];
    }
  | {
      id: string;
      type: 'launch-payment-token-flights';
      durationMs: number;
      flightSequenceDurationMs: number;
      flightDurationMs: number;
      flightStaggerMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'resource-payment-started' }
      >;
    }
  | {
      id: string;
      type: 'apply-resource-payment-token';
      durationMs: number;
      playerId: PlayerId;
      suit: Suit;
    }
  | {
      id: string;
      type: 'apply-resource-payment';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'resource-payment-applied' }
      >;
    }
  | {
      id: string;
      type: 'launch-card-to-district-flight';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'card-played-to-district' }
      >;
    }
  | {
      id: string;
      type: 'place-card-in-district';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'card-played-to-district' }
      >;
    }
  | {
      id: string;
      type: 'launch-deed-token-flights';
      durationMs: number;
      tokens: readonly Extract<
        GamePresentationEvent,
        { type: 'deed-token-paid' }
      >[];
    }
  | {
      id: string;
      type: 'apply-deed-tokens';
      durationMs: number;
      tokens: readonly Extract<
        GamePresentationEvent,
        { type: 'deed-token-paid' }
      >[];
    }
  | {
      id: string;
      type: 'apply-deed-progress';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'deed-progress-applied' }
      >;
    }
  | {
      id: string;
      type: 'reveal-deed-completion';
      durationMs: number;
      event: Extract<GamePresentationEvent, { type: 'deed-completed' }>;
    }
  | {
      id: string;
      type: 'apply-trade-resources';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'trade-resources-applied' }
      >;
    }
  | {
      id: string;
      type: 'roll-income-dice';
      durationMs: number;
      playerId: PlayerId;
      turn: number;
      roll: IncomeRollResult;
      incomeRank: number;
    }
  | {
      id: string;
      type: 'roll-tax-die';
      durationMs: number;
      suit: Suit;
    }
  | {
      id: string;
      type: 'hold-before-tax-flights';
      durationMs: number;
    }
  | {
      id: string;
      type: 'launch-tax-token-flights';
      durationMs: number;
      flightSequenceDurationMs: number;
      losses: readonly Extract<
        GamePresentationEvent,
        { type: 'tax-token-lost' }
      >[];
    }
  | {
      id: string;
      type: 'apply-tax-token-loss';
      durationMs: number;
      loss: Extract<GamePresentationEvent, { type: 'tax-token-lost' }>;
    }
  | {
      id: string;
      type: 'stage-gap';
      durationMs: number;
    }
  | {
      id: string;
      type: 'hold-before-income-flights';
      durationMs: number;
    }
  | {
      id: string;
      type: 'highlight-income-sources';
      durationMs: number;
      cardIds: readonly CardId[];
      crowns: readonly { playerId: PlayerId; suit: Suit }[];
    }
  | {
      id: string;
      type: 'launch-income-token-flights';
      durationMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: 'income-token-gained' }
      >[];
    }
  | {
      id: string;
      type: 'apply-income-gains';
      durationMs: number;
      gains: readonly Extract<
        GamePresentationEvent,
        { type: 'income-token-gained' }
      >[];
    }
  | {
      id: string;
      type: 'post-income-hold';
      durationMs: number;
    }
  | {
      id: string;
      type: 'reveal-income-choice-request';
      durationMs: number;
      choices: readonly IncomeChoice[];
      returnPlayerId: PlayerId | undefined;
    }
  | {
      id: string;
      type: 'reveal-income-choice-submission';
      durationMs: number;
      event: Extract<
        GamePresentationEvent,
        { type: 'income-choice-submitted' }
      >;
    }
  | {
      id: string;
      type: 'commit-view-state';
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
    { id: 'hold-previous-state', type: 'hold-previous-state', durationMs: 0 },
  ];
  const drawEvent = firstEvent(transaction, 'draw-card');
  const incomeChoiceSubmissions = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: 'income-choice-submitted' }
    > => event.type === 'income-choice-submitted'
  );
  const deferIncomeChoiceSubmission = transaction.events.some(
    (event) =>
      event.type === 'income-token-gained' &&
      event.source.kind === 'income-choice'
  );
  if (drawEvent) {
    steps.push({
      id: `draw-card-flight:${drawEvent.playerId}:${drawEvent.cardId}`,
      type: 'draw-card-flight',
      durationMs: durations.cardFlightMs + durations.commitBufferMs,
      playerId: drawEvent.playerId,
      cardId: drawEvent.cardId,
    });
  }

  for (const event of transaction.events) {
    if (event.type === 'card-sold') {
      steps.push({
        id: `stage-sold-card:${event.playerId}:${event.cardId}`,
        type: 'stage-sold-card',
        durationMs: durations.cardFlightMs + durations.commitBufferMs,
        playerId: event.playerId,
        cardId: event.cardId,
      });
    }
    if (event.type === 'trade-resources-applied') {
      steps.push({
        id: `apply-trade-resources:${event.playerId}:${event.give}:${event.receive}`,
        type: 'apply-trade-resources',
        durationMs: 0,
        event,
      });
    }
    if (
      event.type === 'income-choice-submitted' &&
      !deferIncomeChoiceSubmission
    ) {
      appendIncomeChoiceSubmissionStep(steps, event);
    }
  }

  appendCardPlacementSteps(transaction, steps, durations);
  appendActionResourcePaymentSteps(transaction, steps, durations);
  appendDeedDevelopmentSteps(transaction, steps, durations);
  appendSellGainSteps(transaction, steps, durations);

  const incomeRoll = firstEvent(transaction, 'income-roll');
  if (incomeRoll) {
    steps.push({
      id: `roll-income-dice:${incomeRoll.roll.rollId ?? `${incomeRoll.roll.die1}-${incomeRoll.roll.die2}`}`,
      type: 'roll-income-dice',
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
    ): event is Extract<GamePresentationEvent, { type: 'tax-token-lost' }> =>
      event.type === 'tax-token-lost'
  );
  const taxResolved = firstEvent(transaction, 'tax-resolved');
  const taxSuit = taxResolved?.suit ?? taxLosses[0]?.suit;
  if (taxSuit) {
    steps.push({
      id: `roll-tax-die:${taxSuit}`,
      type: 'roll-tax-die',
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
      id: 'hold-before-tax-flights',
      type: 'hold-before-tax-flights',
      durationMs: durations.taxPreFlightHoldMs,
    });
    steps.push({
      id: 'launch-tax-token-flights',
      type: 'launch-tax-token-flights',
      durationMs: 0,
      flightSequenceDurationMs,
      losses: taxLosses,
    });
    taxLosses.forEach((loss, index) => {
      const isLastLoss = index === taxLosses.length - 1;
      steps.push({
        id: `apply-tax-token-loss:${loss.playerId}:${loss.suit}:${String(loss.tokenIndex)}`,
        type: 'apply-tax-token-loss',
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
      { type: 'income-token-gained' }
    > => event.type === 'income-token-gained'
  );
  if (
    (taxSuit || taxLosses.length > 0) &&
    (incomeGains.length > 0 || hasEvent(transaction, 'income-choice-required'))
  ) {
    steps.push({
      id: 'stage-gap',
      type: 'stage-gap',
      durationMs: durations.stageGapMs,
    });
  }
  if (incomeGains.length > 0) {
    const targets = highlightTargetsForIncomeEvents(incomeGains);
    steps.push({
      id: 'hold-before-income-flights',
      type: 'hold-before-income-flights',
      durationMs: durations.incomePreFlightHoldMs,
    });
    if (deferIncomeChoiceSubmission) {
      for (const submission of incomeChoiceSubmissions) {
        appendIncomeChoiceSubmissionStep(steps, submission);
      }
    }
    if (targets.cardIds.length > 0 || targets.crowns.length > 0) {
      steps.push({
        id: 'highlight-income-sources',
        type: 'highlight-income-sources',
        durationMs: 0,
        cardIds: targets.cardIds,
        crowns: targets.crowns,
      });
    }
    steps.push({
      id: 'launch-income-token-flights',
      type: 'launch-income-token-flights',
      durationMs: staggeredDuration(
        incomeGains.length,
        durations.incomeFlightMs,
        durations.incomeFlightStaggerMs
      ),
      gains: incomeGains,
    });
    steps.push({
      id: 'apply-income-gains',
      type: 'apply-income-gains',
      durationMs: 0,
      gains: incomeGains,
    });
    steps.push({
      id: 'post-income-hold',
      type: 'post-income-hold',
      durationMs: durations.postIncomeHoldMs,
    });
  }

  for (const event of transaction.events) {
    if (event.type === 'income-choice-required') {
      steps.push({
        id: 'reveal-income-choice-request',
        type: 'reveal-income-choice-request',
        durationMs: 0,
        choices: event.choices,
        returnPlayerId: event.returnPlayerId,
      });
    }
  }

  steps.push({
    id: 'commit-view-state',
    type: 'commit-view-state',
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
    commitMs: scheduled.find((step) => step.type === 'commit-view-state')
      ?.startMs ?? cursorMs,
    inputUnlockMs: scheduled.find((step) => step.type === 'commit-view-state')
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
      { type: 'resource-payment-started' }
    > =>
      event.type === 'resource-payment-started' &&
      event.reason !== 'develop-deed'
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
      type: 'launch-payment-token-flights',
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
        { type: 'resource-payment-applied' }
      > =>
        event.type === 'resource-payment-applied' &&
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
          type: 'apply-resource-payment-token',
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
      { type: 'card-played-to-district' }
    > => event.type === 'card-played-to-district'
  );
  for (const event of placements) {
    steps.push(
      {
        id: `launch-card-to-district-flight:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: 'launch-card-to-district-flight',
        durationMs: durations.cardFlightMs,
        event,
      },
      {
        id: `place-card-in-district:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: 'place-card-in-district',
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
      { type: 'resource-payment-applied' }
    > =>
      event.type === 'resource-payment-applied' &&
      event.reason === 'develop-deed'
  );
  if (deedPayment) {
    steps.push({
      id: `apply-resource-payment:${deedPayment.reason}:${deedPayment.playerId}:${deedPayment.cardId}:${deedPayment.districtId}`,
      type: 'apply-resource-payment',
      durationMs: 0,
      event: deedPayment,
    });
  }

  const deedTokens = transaction.events.filter(
    (
      event
    ): event is Extract<GamePresentationEvent, { type: 'deed-token-paid' }> =>
      event.type === 'deed-token-paid'
  );
  if (deedTokens.length > 0) {
    steps.push({
      id: 'launch-deed-token-flights',
      type: 'launch-deed-token-flights',
      durationMs: staggeredDuration(
        deedTokens.length,
        durations.actionResourceFlightMs,
        durations.actionResourceFlightStaggerMs
      ),
      tokens: deedTokens,
    });
    steps.push({
      id: 'apply-deed-tokens',
      type: 'apply-deed-tokens',
      durationMs: durations.commitBufferMs,
      tokens: deedTokens,
    });
  }

  for (const event of transaction.events) {
    if (event.type === 'deed-progress-applied') {
      steps.push({
        id: `apply-deed-progress:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: 'apply-deed-progress',
        durationMs: durations.deedProgressRevealMs,
        event,
      });
    }
    if (event.type === 'deed-completed') {
      steps.push({
        id: `reveal-deed-completion:${event.playerId}:${event.cardId}:${event.districtId}`,
        type: 'reveal-deed-completion',
        durationMs: 0,
        event,
      });
    }
  }
}

function appendSellGainSteps(
  transaction: GameTransaction,
  steps: AnimationStep[],
  durations: AnimationDurations
): void {
  const gains = transaction.events.filter(
    (
      event
    ): event is Extract<
      GamePresentationEvent,
      { type: 'sell-resource-gained' }
    > => event.type === 'sell-resource-gained'
  );
  if (gains.length === 0) {
    return;
  }
  steps.push({
    id: `apply-sell-resource-gains:${gains[0].playerId}:${gains[0].cardId}`,
    type: 'apply-sell-resource-gains',
    durationMs: durations.commitBufferMs,
    gains,
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
    { type: 'income-token-gained' }
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
    if (event.source.kind === 'income-choice') {
      continue;
    }
    if (event.source.kind === 'crown') {
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
  event: Extract<GamePresentationEvent, { type: 'income-choice-submitted' }>
): void {
  steps.push({
    id: `reveal-income-choice-submission:${event.playerId}:${event.districtId}:${event.cardId}`,
    type: 'reveal-income-choice-submission',
    durationMs: 0,
    event,
  });
}
