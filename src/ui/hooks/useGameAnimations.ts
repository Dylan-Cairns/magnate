import { useCallback, useEffect, useRef, useState } from 'react';

import type { CardId } from '../../engine/cards';
import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import {
  createAnimationTimerRegistry,
  type AnimationTimerRegistry,
} from '../animations/animationTimerRegistry';
import { browserAnimationDomTargets } from '../animations/domTargets';
import {
  buildIncomeFlightsFromDom,
  buildTaxLossFlightsFromDom,
} from '../animations/flightPlans';
import {
  cardFlightSettleMs,
  resourceFlightSettleMs,
  shouldAllowHumanActionsDuringAnimationSettle,
  shouldCommitBeforeAnimationSettle,
} from '../animations/timing';
import type {
  TurnCycleAnimationPlan,
  TurnCycleVisualPlan,
} from '../animations/turnCycleVisualPlan';
import type { CardFlight, ResourceFlight } from '../animations/types';
import {
  derivePresentationSnapshot,
  type PresentationSnapshot,
} from '../runtime/presentation';
import { buildPresentationTimeline } from '../runtime/timeline';
import { deriveGamePresentationEvents } from '../runtime/transactions';
import type {
  GameTransaction,
  PresentationTimeline,
  PresentationTimelineEvent,
} from '../runtime/types';

const ANIMATIONS_STORAGE_KEY = 'magnate:animationsEnabled';

type RunTransitionOptions = {
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  resourceFlights: readonly ResourceFlight[];
  cardFlights: readonly CardFlight[];
  turnCyclePlan: TurnCycleAnimationPlan | null;
  onSettle?: () => void;
};

type UseGameAnimationsOptions = {
  onCommitTransition: (
    previousState: GameState,
    nextState: GameState,
    action: GameAction
  ) => void;
};

export function useGameAnimations({
  onCommitTransition,
}: UseGameAnimationsOptions) {
  const [enabled, setEnabled] = useState<boolean>(() =>
    readAnimationsEnabledPreference()
  );
  const [resourceFlights, setResourceFlights] = useState<
    ReadonlyArray<ResourceFlight>
  >([]);
  const [cardFlights, setCardFlights] = useState<ReadonlyArray<CardFlight>>([]);
  const [incomeHighlightCardIds, setIncomeHighlightCardIds] = useState<
    ReadonlyArray<CardId>
  >([]);
  const [incomeHighlightCrowns, setIncomeHighlightCrowns] = useState<
    ReadonlyArray<{ playerId: PlayerId; suit: Suit }>
  >([]);
  const [presentationSnapshot, setPresentationSnapshot] =
    useState<PresentationSnapshot | null>(null);
  const [pendingDiscardHoldback, setPendingDiscardHoldback] =
    useState<number>(0);
  const [activePlayerHighlightOverride, setActivePlayerHighlightOverride] =
    useState<PlayerId | null>(null);
  const [actionCommitPending, setActionCommitPending] =
    useState<boolean>(false);
  const [
    allowHumanActionsWhileCommitPending,
    setAllowHumanActionsWhileCommitPending,
  ] = useState<boolean>(false);
  const [actionCommitTimers] = useState<AnimationTimerRegistry>(() =>
    createAnimationTimerRegistry()
  );
  const [turnCycleVisualTimers] = useState<AnimationTimerRegistry>(() =>
    createAnimationTimerRegistry()
  );
  const taxPulseElementsRef = useRef<HTMLElement[]>([]);
  const nextResourceFlightId = useRef(0);
  const nextCardFlightId = useRef(0);

  const makeResourceFlightId = useCallback(() => {
    nextResourceFlightId.current += 1;
    return `resource-flight-${nextResourceFlightId.current}`;
  }, []);
  const makeCardFlightId = useCallback(() => {
    nextCardFlightId.current += 1;
    return `card-flight-${nextCardFlightId.current}`;
  }, []);
  const clearTaxPulseElements = useCallback(() => {
    for (const element of taxPulseElementsRef.current) {
      element.classList.remove('is-tax-pulsing');
    }
    taxPulseElementsRef.current = [];
  }, []);
  const applyTaxPulseTargets = useCallback(
    (
      targets: ReadonlyArray<{
        playerId: PlayerId;
        suit: Suit;
      }>
    ) => {
      clearTaxPulseElements();
      const pulsingElements: HTMLElement[] = [];
      for (const target of targets) {
        const element = browserAnimationDomTargets.resourceToken(
          target.playerId,
          target.suit
        );
        if (!element) {
          continue;
        }
        element.classList.add('is-tax-pulsing');
        pulsingElements.push(element);
      }
      taxPulseElementsRef.current = pulsingElements;
    },
    [clearTaxPulseElements]
  );
  const clearTurnCycleVisuals = useCallback(() => {
    turnCycleVisualTimers.clearAll();
    clearTaxPulseElements();
    setIncomeHighlightCardIds([]);
    setIncomeHighlightCrowns([]);
  }, [clearTaxPulseElements, turnCycleVisualTimers]);
  const scheduleTurnCycleVisuals = useCallback(
    (plan: TurnCycleVisualPlan | null, startDelayMs = 0) => {
      clearTurnCycleVisuals();
      if (!plan) {
        return;
      }

      if (plan.taxSuit) {
        if (
          plan.taxPulseStartAtMs !== null &&
          plan.taxPulseTargets.length > 0
        ) {
          turnCycleVisualTimers.schedule(
            startDelayMs + plan.taxPulseStartAtMs,
            () => {
              applyTaxPulseTargets(plan.taxPulseTargets);
            }
          );
        }
        if (plan.taxPulseEndAtMs !== null) {
          turnCycleVisualTimers.schedule(
            startDelayMs + plan.taxPulseEndAtMs,
            () => {
              clearTaxPulseElements();
            }
          );
        }
        if (
          plan.taxFlightLaunchAtMs !== null &&
          plan.taxFlightTokens.length > 0
        ) {
          turnCycleVisualTimers.schedule(
            startDelayMs + plan.taxFlightLaunchAtMs,
            () => {
              const taxFlights = buildTaxLossFlightsFromDom(
                plan.taxFlightTokens,
                makeResourceFlightId
              );
              if (taxFlights.length === 0) {
                return;
              }
              setResourceFlights((existing) => [...existing, ...taxFlights]);
            }
          );
        }
      }

      turnCycleVisualTimers.schedule(
        startDelayMs + plan.incomeFlightLaunchAtMs,
        () => {
          const incomeFlights = buildIncomeFlightsFromDom(
            plan.incomeFlightTokens,
            makeResourceFlightId
          );
          if (incomeFlights.length === 0) {
            return;
          }
          setResourceFlights((existing) => [...existing, ...incomeFlights]);
        }
      );
      turnCycleVisualTimers.schedule(
        startDelayMs + plan.incomeHighlightStartAtMs,
        () => {
          setIncomeHighlightCardIds(plan.highlightCardIds);
          setIncomeHighlightCrowns(plan.highlightCrowns);
        }
      );
      turnCycleVisualTimers.schedule(
        startDelayMs + plan.incomeHighlightEndAtMs,
        () => {
          setIncomeHighlightCardIds([]);
          setIncomeHighlightCrowns([]);
        }
      );
      turnCycleVisualTimers.schedule(startDelayMs + plan.hideAllAtMs, () => {
        clearTaxPulseElements();
        setIncomeHighlightCardIds([]);
        setIncomeHighlightCrowns([]);
      });
    },
    [
      applyTaxPulseTargets,
      clearTaxPulseElements,
      clearTurnCycleVisuals,
      makeResourceFlightId,
      turnCycleVisualTimers,
    ]
  );
  const clearPendingActionCommit = useCallback(() => {
    actionCommitTimers.clearAll();
    setActionCommitPending(false);
    setActivePlayerHighlightOverride(null);
    setAllowHumanActionsWhileCommitPending(false);
    setPresentationSnapshot(null);
  }, [actionCommitTimers]);
  const clearAllFlights = useCallback(() => {
    setResourceFlights([]);
    setCardFlights([]);
    setPendingDiscardHoldback(0);
    setActivePlayerHighlightOverride(null);
    setAllowHumanActionsWhileCommitPending(false);
    setPresentationSnapshot(null);
    clearTurnCycleVisuals();
  }, [clearTurnCycleVisuals]);
  const runTransition = useCallback(
    ({
      previousState,
      nextState,
      action,
      actingPlayerId,
      resourceFlights: queuedResourceFlights,
      cardFlights: queuedCardFlights,
      turnCyclePlan,
      onSettle,
    }: RunTransitionOptions) => {
      const drawFlightsForTimer = queuedCardFlights.filter(
        (f) => f.variant === 'draw' && f.cardId != null
      );
      const turnCycleStartDelayMs = turnCycleStartDelayForTransition(
        action,
        drawFlightsForTimer
      );
      scheduleTurnCycleVisuals(
        turnCyclePlan?.visualPlan ?? null,
        turnCycleStartDelayMs
      );
      const presentationTransaction = buildPresentationTransactionForTransition(
        previousState,
        nextState,
        action,
        actingPlayerId,
        queuedResourceFlights,
        drawFlightsForTimer,
        turnCyclePlan
      );
      const presentationTimeline = presentationTransaction
        ? buildPresentationTimeline(presentationTransaction)
        : null;
      if (presentationTransaction && presentationTimeline) {
        setPresentationSnapshot(
          derivePresentationSnapshot({
            transaction: presentationTransaction,
            timeline: presentationTimeline,
            elapsedMs: 0,
          })
        );
      } else {
        setPresentationSnapshot(null);
      }

      const settleMs = Math.max(
        resourceFlightSettleMs(queuedResourceFlights),
        cardFlightSettleMs(queuedCardFlights),
        turnCyclePlan
          ? turnCycleStartDelayMs + turnCyclePlan.totalDurationMs
          : 0
      );
      if (settleMs <= 0) {
        setPendingDiscardHoldback(0);
        setActivePlayerHighlightOverride(null);
        setPresentationSnapshot(null);
        setAllowHumanActionsWhileCommitPending(
          shouldAllowHumanActionsDuringAnimationSettle(action)
        );
        onCommitTransition(previousState, nextState, action);
        onSettle?.();
        return;
      }

      if (queuedResourceFlights.length > 0) {
        setResourceFlights((existing) => [
          ...existing,
          ...queuedResourceFlights,
        ]);
      }
      if (queuedCardFlights.length > 0) {
        setCardFlights((existing) => [...existing, ...queuedCardFlights]);
      }
      setActivePlayerHighlightOverride(
        activeHighlightOverrideForTransition(
          action,
          previousState,
          drawFlightsForTimer
        )
      );
      setPendingDiscardHoldback(action.type === 'sell-card' ? 1 : 0);
      setAllowHumanActionsWhileCommitPending(
        shouldAllowHumanActionsDuringAnimationSettle(action) &&
          action.type !== 'choose-income-suit'
      );
      setActionCommitPending(true);
      actionCommitTimers.clearAll();
      if (presentationTransaction && presentationTimeline) {
        schedulePresentationSnapshots(
          presentationTransaction,
          presentationTimeline,
          settleMs,
          actionCommitTimers,
          setPresentationSnapshot
        );
      }
      if (shouldCommitBeforeAnimationSettle(action)) {
        onCommitTransition(previousState, nextState, action);
      }
      if (drawFlightsForTimer.length > 0) {
        const drawSettleMs = cardFlightSettleMs(drawFlightsForTimer);
        actionCommitTimers.schedule(drawSettleMs, () => {
          setActivePlayerHighlightOverride(null);
        });
      }
      actionCommitTimers.schedule(settleMs, () => {
        if (!shouldCommitBeforeAnimationSettle(action)) {
          onCommitTransition(previousState, nextState, action);
        }
        setResourceFlights([]);
        setCardFlights([]);
        setPendingDiscardHoldback(0);
        setActivePlayerHighlightOverride(null);
        setAllowHumanActionsWhileCommitPending(false);
        setPresentationSnapshot(null);
        clearTurnCycleVisuals();
        onSettle?.();
        setActionCommitPending(false);
      });
    },
    [
      actionCommitTimers,
      clearTurnCycleVisuals,
      onCommitTransition,
      scheduleTurnCycleVisuals,
    ]
  );

  useEffect(() => {
    persistAnimationsEnabledPreference(enabled);
  }, [enabled]);

  useEffect(() => {
    return () => {
      actionCommitTimers.clearAll();
      turnCycleVisualTimers.clearAll();
      clearTaxPulseElements();
    };
  }, [actionCommitTimers, clearTaxPulseElements, turnCycleVisualTimers]);

  return {
    enabled,
    setEnabled,
    resourceFlights,
    cardFlights,
    incomeHighlightCardIds,
    incomeHighlightCrowns,
    presentationSnapshot,
    pendingDiscardHoldback,
    activePlayerHighlightOverride,
    actionCommitPending,
    allowHumanActionsWhileCommitPending,
    makeResourceFlightId,
    makeCardFlightId,
    clearPendingActionCommit,
    clearAllFlights,
    runTransition,
  };
}

export function activeHighlightOverrideForTransition(
  action: GameAction,
  previousState: GameState,
  drawFlights: readonly CardFlight[]
): PlayerId | null {
  if (action.type !== 'end-turn' || drawFlights.length === 0) {
    return null;
  }
  return previousState.players[previousState.activePlayerIndex]?.id ?? null;
}

export function turnCycleStartDelayForTransition(
  action: GameAction,
  drawFlights: readonly CardFlight[]
): number {
  if (action.type !== 'end-turn' || drawFlights.length === 0) {
    return 0;
  }
  return cardFlightSettleMs(drawFlights);
}

function buildPresentationTransactionForTransition(
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  resourceFlights: readonly ResourceFlight[],
  drawFlights: readonly CardFlight[],
  turnCyclePlan: TurnCycleAnimationPlan | null
): GameTransaction | null {
  const shouldUseRuntime =
    (action.type === 'end-turn' &&
      (turnCyclePlan !== null || drawFlights.length > 0)) ||
    (action.type === 'choose-income-suit' && resourceFlights.length > 0);
  if (!shouldUseRuntime) {
    return null;
  }

  return {
    id: `${previousState.seed}:${previousState.turn}:${previousState.phase}:${action.type}`,
    previousState,
    nextState,
    action,
    actingPlayerId,
    events: deriveGamePresentationEvents(
      previousState,
      nextState,
      action,
      actingPlayerId
    ),
  };
}

function schedulePresentationSnapshots(
  transaction: GameTransaction,
  timeline: PresentationTimeline,
  settleMs: number,
  timers: AnimationTimerRegistry,
  setSnapshot: (snapshot: PresentationSnapshot | null) => void
): void {
  const updateTimes = uniquePresentationUpdateTimes(timeline.events).filter(
    (atMs) => atMs > 0 && atMs < settleMs
  );
  for (const atMs of updateTimes) {
    timers.schedule(atMs, () => {
      setSnapshot(
        derivePresentationSnapshot({
          transaction,
          timeline,
          elapsedMs: atMs,
        })
      );
    });
  }
}

function uniquePresentationUpdateTimes(
  events: readonly PresentationTimelineEvent[]
): number[] {
  return [...new Set(events.map((event) => event.atMs))].sort(
    (left, right) => left - right
  );
}

function readAnimationsEnabledPreference(): boolean {
  if (typeof window === 'undefined') {
    return true;
  }
  try {
    const stored = window.localStorage.getItem(ANIMATIONS_STORAGE_KEY);
    if (stored === null) {
      return true;
    }
    return stored !== 'false';
  } catch {
    return true;
  }
}

function persistAnimationsEnabledPreference(enabled: boolean): void {
  if (typeof window === 'undefined') {
    return;
  }
  try {
    window.localStorage.setItem(
      ANIMATIONS_STORAGE_KEY,
      enabled ? 'true' : 'false'
    );
  } catch {
    // Ignore storage failures (for example private browsing restrictions).
  }
}
