import {
  ACTION_FLIGHT_COMMIT_BUFFER_MS,
  CARD_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from '../animations/timing';
import { collectTurnCycleAnimationPlan } from '../animations/turnCycleVisualPlan';
import type {
  GamePresentationEvent,
  GameTransaction,
  PresentationTimeline,
  PresentationTimelineEvent,
} from './types';

const DRAW_CARD_REVEAL_MS =
  CARD_FLIGHT_DURATION_MS + ACTION_FLIGHT_COMMIT_BUFFER_MS;

export function buildPresentationTimeline(
  transaction: GameTransaction
): PresentationTimeline {
  const events: PresentationTimelineEvent[] = [
    { atMs: 0, type: 'hold-previous-state' },
  ];
  const turnCyclePlan = collectTurnCycleAnimationPlan(
    transaction.previousState,
    transaction.nextState,
    transaction.action
  );
  const turnCycleStartDelayMs = hasEvent(transaction, 'draw-card')
    ? DRAW_CARD_REVEAL_MS
    : 0;

  for (const event of transaction.events) {
    switch (event.type) {
      case 'draw-card':
        events.push({
          atMs: DRAW_CARD_REVEAL_MS,
          type: 'reveal-drawn-card',
          event,
        });
        break;
      case 'income-roll':
        events.push({
          atMs: turnCycleStartDelayMs,
          type: 'show-income-roll',
          event,
        });
        break;
      case 'tax-token-lost':
        events.push({
          atMs: taxLossTimeMs(event, turnCycleStartDelayMs, turnCyclePlan),
          type: 'apply-tax-token-loss',
          event,
        });
        break;
      case 'income-token-gained':
        addIncomeTokenEvents(
          events,
          event,
          incomeTokenIndex(transaction.events, event),
          turnCycleStartDelayMs,
          turnCyclePlan
        );
        break;
      case 'income-choice-required':
        events.push({
          atMs: turnCyclePlan
            ? turnCycleStartDelayMs + turnCyclePlan.totalDurationMs
            : 0,
          type: 'reveal-income-choice-request',
          event,
        });
        break;
      case 'income-choice-submitted':
        events.push({
          atMs: 0,
          type: 'reveal-income-choice-submission',
          event,
        });
        break;
    }
  }

  events.push({
    atMs: commitTimeMs(transaction, turnCycleStartDelayMs, turnCyclePlan),
    type: 'commit-view-to-next-state',
  });

  const sorted = sortTimelineEvents(events);
  return {
    transactionId: transaction.id,
    durationMs: sorted[sorted.length - 1]?.atMs ?? 0,
    events: sorted,
  };
}

function addIncomeTokenEvents(
  events: PresentationTimelineEvent[],
  event: Extract<GamePresentationEvent, { type: 'income-token-gained' }>,
  index: number,
  turnCycleStartDelayMs: number,
  turnCyclePlan: ReturnType<typeof collectTurnCycleAnimationPlan>
): void {
  const launchAtMs = incomeFlightLaunchTimeMs(
    index,
    turnCycleStartDelayMs,
    turnCyclePlan
  );
  events.push({
    atMs: launchAtMs,
    type: 'launch-income-token-flight',
    event,
  });
  events.push({
    atMs: launchAtMs + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
    type: 'apply-income-token-gain',
    event,
  });
}

function incomeFlightLaunchTimeMs(
  index: number,
  turnCycleStartDelayMs: number,
  turnCyclePlan: ReturnType<typeof collectTurnCycleAnimationPlan>
): number {
  if (!turnCyclePlan) {
    return index * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS;
  }
  return (
    turnCycleStartDelayMs +
    turnCyclePlan.visualPlan.incomeFlightLaunchAtMs +
    index * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS
  );
}

function taxLossTimeMs(
  event: Extract<GamePresentationEvent, { type: 'tax-token-lost' }>,
  turnCycleStartDelayMs: number,
  turnCyclePlan: ReturnType<typeof collectTurnCycleAnimationPlan>
): number {
  if (!turnCyclePlan || turnCyclePlan.visualPlan.taxFlightLaunchAtMs === null) {
    return turnCycleStartDelayMs;
  }
  return (
    turnCycleStartDelayMs +
    turnCyclePlan.visualPlan.taxFlightLaunchAtMs +
    event.tokenIndex * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS
  );
}

function commitTimeMs(
  transaction: GameTransaction,
  turnCycleStartDelayMs: number,
  turnCyclePlan: ReturnType<typeof collectTurnCycleAnimationPlan>
): number {
  if (turnCyclePlan) {
    return turnCycleStartDelayMs + turnCyclePlan.totalDurationMs;
  }

  const incomeEventCount = transaction.events.filter(
    (event) => event.type === 'income-token-gained'
  ).length;
  if (incomeEventCount > 0) {
    return (
      (incomeEventCount - 1) * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS +
      TURN_CYCLE_INCOME_FLIGHT_DURATION_MS +
      ACTION_FLIGHT_COMMIT_BUFFER_MS
    );
  }

  return 0;
}

function incomeTokenIndex(
  events: readonly GamePresentationEvent[],
  target: Extract<GamePresentationEvent, { type: 'income-token-gained' }>
): number {
  return events
    .filter((event) => event.type === 'income-token-gained')
    .findIndex((event) => event === target);
}

function hasEvent<TType extends GamePresentationEvent['type']>(
  transaction: GameTransaction,
  type: TType
): boolean {
  return transaction.events.some((event) => event.type === type);
}

function sortTimelineEvents(
  events: readonly PresentationTimelineEvent[]
): PresentationTimelineEvent[] {
  return [...events].sort((left, right) => {
    if (left.atMs !== right.atMs) {
      return left.atMs - right.atMs;
    }
    return timelineEventPriority(left.type) - timelineEventPriority(right.type);
  });
}

function timelineEventPriority(
  type: PresentationTimelineEvent['type']
): number {
  switch (type) {
    case 'hold-previous-state':
      return 0;
    case 'reveal-drawn-card':
      return 10;
    case 'show-income-roll':
      return 20;
    case 'apply-tax-token-loss':
      return 30;
    case 'launch-income-token-flight':
      return 40;
    case 'apply-income-token-gain':
      return 50;
    case 'reveal-income-choice-submission':
      return 60;
    case 'reveal-income-choice-request':
      return 70;
    case 'commit-view-to-next-state':
      return 100;
  }
}
