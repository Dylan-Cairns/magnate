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
import {
  applyTurnCycleResourcePreviewEvent,
  buildTurnCycleResourcePreviewPlan,
  type ResourcePreviewByPlayer,
} from '../animations/turnCycleResourcePreview';
import type {
  TurnCycleAnimationPlan,
  TurnCycleVisualPlan,
} from '../animations/turnCycleVisualPlan';
import type {
  CardFlight,
  ResourceFlight,
  TurnCycleOverlayState,
} from '../animations/types';

const ANIMATIONS_STORAGE_KEY = 'magnate:animationsEnabled';

type RunTransitionOptions = {
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
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
  const [turnCycleOverlay, setTurnCycleOverlay] =
    useState<TurnCycleOverlayState | null>(null);
  const [incomeHighlightCardIds, setIncomeHighlightCardIds] = useState<
    ReadonlyArray<CardId>
  >([]);
  const [incomeHighlightCrowns, setIncomeHighlightCrowns] = useState<
    ReadonlyArray<{ playerId: PlayerId; suit: Suit }>
  >([]);
  const [incomeResourcePreviewByPlayer, setIncomeResourcePreviewByPlayer] =
    useState<ResourcePreviewByPlayer>(null);
  const [pendingDiscardHoldback, setPendingDiscardHoldback] =
    useState<number>(0);
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
    setTurnCycleOverlay(null);
    setIncomeHighlightCardIds([]);
    setIncomeHighlightCrowns([]);
    setIncomeResourcePreviewByPlayer(null);
  }, [clearTaxPulseElements, turnCycleVisualTimers]);
  const scheduleTurnCycleVisuals = useCallback(
    (plan: TurnCycleVisualPlan | null) => {
      clearTurnCycleVisuals();
      if (!plan) {
        return;
      }

      if (plan.taxLabelAtMs !== null && plan.taxSuit) {
        const taxSuit = plan.taxSuit;
        turnCycleVisualTimers.schedule(plan.taxLabelAtMs, () => {
          setTurnCycleOverlay({ kind: 'tax', suit: taxSuit });
        });
        if (plan.taxLabelHideAtMs !== null) {
          turnCycleVisualTimers.schedule(plan.taxLabelHideAtMs, () => {
            setTurnCycleOverlay(null);
          });
        }
        if (
          plan.taxPulseStartAtMs !== null &&
          plan.taxPulseTargets.length > 0
        ) {
          turnCycleVisualTimers.schedule(plan.taxPulseStartAtMs, () => {
            applyTaxPulseTargets(plan.taxPulseTargets);
          });
        }
        if (plan.taxPulseEndAtMs !== null) {
          turnCycleVisualTimers.schedule(plan.taxPulseEndAtMs, () => {
            clearTaxPulseElements();
          });
        }
        if (
          plan.taxFlightLaunchAtMs !== null &&
          plan.taxFlightTokens.length > 0
        ) {
          turnCycleVisualTimers.schedule(plan.taxFlightLaunchAtMs, () => {
            const taxFlights = buildTaxLossFlightsFromDom(
              plan.taxFlightTokens,
              makeResourceFlightId
            );
            if (taxFlights.length === 0) {
              return;
            }
            setResourceFlights((existing) => [...existing, ...taxFlights]);
          });
        }
      }

      turnCycleVisualTimers.schedule(plan.incomeLabelAtMs, () => {
        setTurnCycleOverlay({ kind: 'income', rank: plan.incomeRank });
      });
      turnCycleVisualTimers.schedule(plan.incomeLabelHideAtMs, () => {
        setTurnCycleOverlay(null);
      });
      turnCycleVisualTimers.schedule(plan.incomeFlightLaunchAtMs, () => {
        const incomeFlights = buildIncomeFlightsFromDom(
          plan.incomeFlightTokens,
          makeResourceFlightId
        );
        if (incomeFlights.length === 0) {
          return;
        }
        setResourceFlights((existing) => [...existing, ...incomeFlights]);
      });
      turnCycleVisualTimers.schedule(plan.incomeHighlightStartAtMs, () => {
        setIncomeHighlightCardIds(plan.highlightCardIds);
        setIncomeHighlightCrowns(plan.highlightCrowns);
      });
      turnCycleVisualTimers.schedule(plan.incomeHighlightEndAtMs, () => {
        setIncomeHighlightCardIds([]);
        setIncomeHighlightCrowns([]);
      });
      turnCycleVisualTimers.schedule(plan.hideAllAtMs, () => {
        clearTaxPulseElements();
        setTurnCycleOverlay(null);
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
  const scheduleTurnCycleResourcePreview = useCallback(
    (
      previousState: GameState,
      nextState: GameState,
      turnCyclePlan: TurnCycleAnimationPlan
    ) => {
      const previewPlan = buildTurnCycleResourcePreviewPlan(
        previousState,
        nextState,
        turnCyclePlan
      );
      setIncomeResourcePreviewByPlayer(previewPlan.initialPreview);
      for (const event of previewPlan.events) {
        turnCycleVisualTimers.schedule(event.atMs, () => {
          setIncomeResourcePreviewByPlayer((existing) =>
            applyTurnCycleResourcePreviewEvent(existing, event)
          );
        });
      }
    },
    [turnCycleVisualTimers]
  );
  const clearPendingActionCommit = useCallback(() => {
    actionCommitTimers.clearAll();
    setActionCommitPending(false);
    setAllowHumanActionsWhileCommitPending(false);
  }, [actionCommitTimers]);
  const clearAllFlights = useCallback(() => {
    setResourceFlights([]);
    setCardFlights([]);
    setPendingDiscardHoldback(0);
    setAllowHumanActionsWhileCommitPending(false);
    clearTurnCycleVisuals();
  }, [clearTurnCycleVisuals]);
  const runTransition = useCallback(
    ({
      previousState,
      nextState,
      action,
      resourceFlights: queuedResourceFlights,
      cardFlights: queuedCardFlights,
      turnCyclePlan,
      onSettle,
    }: RunTransitionOptions) => {
      scheduleTurnCycleVisuals(turnCyclePlan?.visualPlan ?? null);
      if (action.type === 'end-turn' && turnCyclePlan) {
        scheduleTurnCycleResourcePreview(
          previousState,
          nextState,
          turnCyclePlan
        );
      }

      const settleMs = Math.max(
        resourceFlightSettleMs(queuedResourceFlights),
        cardFlightSettleMs(queuedCardFlights),
        turnCyclePlan?.totalDurationMs ?? 0
      );
      if (settleMs <= 0) {
        setPendingDiscardHoldback(0);
        setAllowHumanActionsWhileCommitPending(false);
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
      setPendingDiscardHoldback(action.type === 'sell-card' ? 1 : 0);
      setAllowHumanActionsWhileCommitPending(
        shouldAllowHumanActionsDuringAnimationSettle(action)
      );
      setActionCommitPending(true);
      if (shouldCommitBeforeAnimationSettle(action)) {
        onCommitTransition(previousState, nextState, action);
      }
      actionCommitTimers.clearAll();
      actionCommitTimers.schedule(settleMs, () => {
        if (!shouldCommitBeforeAnimationSettle(action)) {
          onCommitTransition(previousState, nextState, action);
        }
        setResourceFlights([]);
        setCardFlights([]);
        setPendingDiscardHoldback(0);
        setAllowHumanActionsWhileCommitPending(false);
        clearTurnCycleVisuals();
        onSettle?.();
        setActionCommitPending(false);
      });
    },
    [
      actionCommitTimers,
      clearTurnCycleVisuals,
      onCommitTransition,
      scheduleTurnCycleResourcePreview,
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
    turnCycleOverlay,
    incomeHighlightCardIds,
    incomeHighlightCrowns,
    incomeResourcePreviewByPlayer,
    pendingDiscardHoldback,
    actionCommitPending,
    allowHumanActionsWhileCommitPending,
    makeResourceFlightId,
    makeCardFlightId,
    clearPendingActionCommit,
    clearAllFlights,
    runTransition,
  };
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
