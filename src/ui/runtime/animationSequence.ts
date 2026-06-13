import type { CardId } from '../../engine/cards';
import type {
  IncomeChoice,
  IncomeRollResult,
  PlayerId,
  Suit,
} from '../../engine/types';
import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  CARD_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from '../animations/timing';
import type { GamePresentationEvent, GameTransaction } from './types';

export type AnimationDurations = {
  cardFlightMs: number;
  commitBufferMs: number;
  dieRollMs: number;
  diePulseMs: number;
  taxDieRollMs: number;
  taxPulseMs: number;
  taxPreFlightHoldMs: number;
  taxFlightMs: number;
  taxFlightStaggerMs: number;
  stageGapMs: number;
  incomePreFlightHoldMs: number;
  incomeFlightMs: number;
  incomeFlightStaggerMs: number;
  postIncomeHoldMs: number;
};

export const DEFAULT_ANIMATION_DURATIONS: AnimationDurations = {
  cardFlightMs: CARD_FLIGHT_DURATION_MS,
  commitBufferMs: ACTION_FLIGHT_COMMIT_BUFFER_MS,
  dieRollMs: 1000,
  diePulseMs: 900,
  taxDieRollMs: 1000,
  taxPulseMs: 900,
  taxPreFlightHoldMs: 550,
  taxFlightMs: TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  taxFlightStaggerMs: TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
  stageGapMs: 220,
  incomePreFlightHoldMs: 400,
  incomeFlightMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  incomeFlightStaggerMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  postIncomeHoldMs: 220,
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
      type: 'roll-income-dice';
      durationMs: number;
      playerId: PlayerId;
      roll: IncomeRollResult;
      incomeRank: number;
    }
  | {
      id: string;
      type: 'pulse-income-die';
      durationMs: number;
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
      type: 'pulse-tax-die';
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
      losses: readonly Extract<
        GamePresentationEvent,
        { type: 'tax-token-lost' }
      >[];
    }
  | {
      id: string;
      type: 'apply-tax-losses';
      durationMs: number;
      losses: readonly Extract<
        GamePresentationEvent,
        { type: 'tax-token-lost' }
      >[];
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
    if (event.type === 'income-choice-submitted') {
      steps.push({
        id: `reveal-income-choice-submission:${event.playerId}:${event.districtId}:${event.cardId}`,
        type: 'reveal-income-choice-submission',
        durationMs: 0,
        event,
      });
    }
  }

  const incomeRoll = firstEvent(transaction, 'income-roll');
  if (incomeRoll) {
    steps.push({
      id: `roll-income-dice:${incomeRoll.roll.rollId ?? `${incomeRoll.roll.die1}-${incomeRoll.roll.die2}`}`,
      type: 'roll-income-dice',
      durationMs: durations.dieRollMs,
      playerId: incomeRoll.playerId,
      roll: incomeRoll.roll,
      incomeRank: incomeRoll.incomeRank,
    });
    steps.push({
      id: `pulse-income-die:${incomeRoll.roll.rollId ?? `${incomeRoll.roll.die1}-${incomeRoll.roll.die2}`}`,
      type: 'pulse-income-die',
      durationMs: durations.diePulseMs,
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
  const taxSuit = taxLosses[0]?.suit ?? transaction.nextState.lastTaxSuit;
  if (taxSuit) {
    steps.push({
      id: `roll-tax-die:${taxSuit}`,
      type: 'roll-tax-die',
      durationMs: durations.taxDieRollMs,
      suit: taxSuit,
    });
    steps.push({
      id: `pulse-tax-die:${taxSuit}`,
      type: 'pulse-tax-die',
      durationMs: durations.taxPulseMs,
      suit: taxSuit,
    });
  }
  if (taxLosses.length > 0) {
    steps.push({
      id: 'hold-before-tax-flights',
      type: 'hold-before-tax-flights',
      durationMs: durations.taxPreFlightHoldMs,
    });
    steps.push({
      id: 'launch-tax-token-flights',
      type: 'launch-tax-token-flights',
      durationMs: staggeredDuration(
        taxLosses.length,
        durations.taxFlightMs,
        durations.taxFlightStaggerMs
      ),
      losses: taxLosses,
    });
    steps.push({
      id: 'apply-tax-losses',
      type: 'apply-tax-losses',
      durationMs: 0,
      losses: taxLosses,
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
