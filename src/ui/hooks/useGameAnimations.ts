import { useCallback, useEffect, useRef, useState } from 'react';

import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import {
  createAnimationTimerRegistry,
  type AnimationTimerRegistry,
} from '../animations/animationTimerRegistry';
import { browserAnimationDomTargets } from '../animations/domTargets';
import {
  buildCardToDistrictFlightFromDom,
  buildDeedResourceFlightsFromDom,
  buildDrawCardFlightFromDom,
  buildIncomeFlightsFromDom,
  buildPaymentFlightsFromDom,
  buildSoldCardFlightFromDom,
  buildTaxLossFlightsFromDom,
  type IncomeFlightToken,
} from '../animations/flightPlans';
import type { CardFlight, ResourceFlight } from '../animations/types';
import {
  deriveAnimationVisualCommands,
  type AnimationVisualCommand,
} from '../runtime/animationVisualCommands';
import {
  derivePresentationSnapshotFromSequence,
  type PresentationSnapshot,
} from '../runtime/presentation';
import {
  buildAnimationSequence,
  type AnimationSequence,
  type ScheduledAnimationStep,
} from '../runtime/animationSequence';
import { deriveGamePresentationEvents } from '../runtime/transactions';
import type { GameTransaction } from '../runtime/types';

const ANIMATIONS_STORAGE_KEY = 'magnate:animationsEnabled';

type RunTransitionOptions = {
  transactionId: string;
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  onInputUnlock?: () => void;
  onSettle?: () => void;
};

type ActivePresentationTransition = {
  options: RunTransitionOptions;
  inputUnlocked: boolean;
  settled: boolean;
};

export function useGameAnimations() {
  const [enabled, setEnabled] = useState<boolean>(() =>
    readAnimationsEnabledPreference()
  );
  const [resourceFlights, setResourceFlights] = useState<
    ReadonlyArray<ResourceFlight>
  >([]);
  const [cardFlights, setCardFlights] = useState<ReadonlyArray<CardFlight>>([]);
  const [presentationSnapshot, setPresentationSnapshot] =
    useState<PresentationSnapshot | null>(null);
  const [presentedState, setPresentedState] = useState<GameState | null>(null);
  const [presentationPending, setPresentationPending] =
    useState<boolean>(false);
  const [presentationTimers] = useState<AnimationTimerRegistry>(() =>
    createAnimationTimerRegistry()
  );
  const [turnCycleVisualTimers] = useState<AnimationTimerRegistry>(() =>
    createAnimationTimerRegistry()
  );
  const taxPulseElementsRef = useRef<HTMLElement[]>([]);
  const nextResourceFlightId = useRef(0);
  const nextCardFlightId = useRef(0);
  const queuedTransitionsRef = useRef<RunTransitionOptions[]>([]);
  const activeTransitionRef = useRef<ActivePresentationTransition | null>(null);
  const startNextTransitionRef = useRef<() => void>(() => undefined);

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
  }, [clearTaxPulseElements, turnCycleVisualTimers]);
  const appendResourceFlightsWithCleanup = useCallback(
    (flights: readonly ResourceFlight[], durationMs: number) => {
      if (flights.length === 0) {
        return;
      }
      setResourceFlights((existing) => [...existing, ...flights]);
      const flightIds = new Set(flights.map((flight) => flight.id));
      turnCycleVisualTimers.schedule(durationMs, () => {
        setResourceFlights((existing) =>
          existing.filter((flight) => !flightIds.has(flight.id))
        );
      });
    },
    [turnCycleVisualTimers]
  );
  const appendCardFlightsWithCleanup = useCallback(
    (flights: readonly CardFlight[], durationMs: number) => {
      if (flights.length === 0) {
        return;
      }
      setCardFlights((existing) => [...existing, ...flights]);
      const flightIds = new Set(flights.map((flight) => flight.id));
      turnCycleVisualTimers.schedule(durationMs, () => {
        setCardFlights((existing) =>
          existing.filter((flight) => !flightIds.has(flight.id))
        );
      });
    },
    [turnCycleVisualTimers]
  );
  const scheduleSequenceVisualCommand = useCallback(
    (command: AnimationVisualCommand) => {
      const scheduleAt = (atMs: number, callback: () => void) => {
        if (atMs <= 0) {
          callback();
          return;
        }
        turnCycleVisualTimers.schedule(atMs, callback);
      };

      switch (command.type) {
        case 'launch-draw-card-flight':
          scheduleAt(command.atMs, () => {
            const flights = buildDrawCardFlightFromDom(
              command.playerId,
              command.cardId,
              makeCardFlightId
            );
            if (flights.length === 0) {
              return;
            }
            setCardFlights((existing) => [...existing, ...flights]);
          });
          return;
        case 'launch-sold-card-flight':
          scheduleAt(command.atMs, () => {
            const flights = buildSoldCardFlightFromDom(
              command.playerId,
              command.cardId,
              makeCardFlightId
            );
            if (flights.length === 0) {
              return;
            }
            setCardFlights((existing) => [...existing, ...flights]);
          });
          return;
        case 'launch-card-to-district-flight':
          scheduleAt(command.atMs, () => {
            const flights = buildCardToDistrictFlightFromDom(
              command.event,
              makeCardFlightId
            );
            if (flights.length === 0) {
              return;
            }
            appendCardFlightsWithCleanup(flights, command.durationMs);
          });
          return;
        case 'launch-payment-token-flights':
          scheduleAt(command.atMs, () => {
            const flights = buildPaymentFlightsFromDom(
              command.event,
              makeResourceFlightId,
              browserAnimationDomTargets,
              {
                durationMs: command.flightDurationMs,
                staggerMs: command.flightStaggerMs,
              }
            );
            if (flights.length === 0) {
              return;
            }
            appendResourceFlightsWithCleanup(flights, command.durationMs);
          });
          return;
        case 'launch-deed-token-flights':
          scheduleAt(command.atMs, () => {
            const flights = buildDeedResourceFlightsFromDom(
              command.tokens,
              makeResourceFlightId
            );
            if (flights.length === 0) {
              return;
            }
            appendResourceFlightsWithCleanup(flights, command.durationMs);
          });
          return;
        case 'pulse-tax-resources':
          scheduleAt(command.startMs, () => {
            applyTaxPulseTargets(command.targets);
          });
          scheduleAt(command.endMs, () => {
            clearTaxPulseElements();
          });
          return;
        case 'launch-tax-token-flights':
          scheduleAt(command.atMs, () => {
            const taxFlights = buildTaxLossFlightsFromDom(
              command.losses.map((loss) => ({
                playerId: loss.playerId,
                suit: loss.suit,
              })),
              makeResourceFlightId
            );
            if (taxFlights.length === 0) {
              return;
            }
            appendResourceFlightsWithCleanup(taxFlights, command.durationMs);
          });
          return;
        case 'launch-income-token-flights':
          scheduleAt(command.atMs, () => {
            const incomeFlights = buildIncomeFlightsFromDom(
              command.gains.map(
                (gain): IncomeFlightToken => ({
                  playerId: gain.playerId,
                  suit: gain.suit,
                  source: gain.source,
                })
              ),
              makeResourceFlightId
            );
            if (incomeFlights.length === 0) {
              return;
            }
            appendResourceFlightsWithCleanup(incomeFlights, command.durationMs);
          });
          return;
      }
    },
    [
      appendCardFlightsWithCleanup,
      appendResourceFlightsWithCleanup,
      applyTaxPulseTargets,
      clearTaxPulseElements,
      makeCardFlightId,
      makeResourceFlightId,
      turnCycleVisualTimers,
    ]
  );
  const scheduleSequenceVisuals = useCallback(
    (
      transaction: GameTransaction | null,
      sequence: AnimationSequence | null
    ) => {
      clearTurnCycleVisuals();
      if (!transaction || !sequence) {
        return;
      }

      for (const command of deriveAnimationVisualCommands(sequence)) {
        scheduleSequenceVisualCommand(command);
      }
    },
    [clearTurnCycleVisuals, scheduleSequenceVisualCommand]
  );
  const clearAllFlights = useCallback(() => {
    setResourceFlights([]);
    setCardFlights([]);
    clearTurnCycleVisuals();
  }, [clearTurnCycleVisuals]);
  const clearPresentationQueue = useCallback(() => {
    presentationTimers.clearAll();
    queuedTransitionsRef.current = [];
    activeTransitionRef.current = null;
    setPresentationPending(false);
    setPresentationSnapshot(null);
    setPresentedState(null);
    clearAllFlights();
  }, [clearAllFlights, presentationTimers]);
  const finishPresentationQueue = useCallback(() => {
    presentationTimers.clearAll();
    const activeTransition = activeTransitionRef.current;
    const queuedTransitions = queuedTransitionsRef.current;
    queuedTransitionsRef.current = [];
    activeTransitionRef.current = null;

    if (activeTransition) {
      unlockPresentationTransition(activeTransition);
      settlePresentationTransition(activeTransition);
    }
    for (const queuedTransition of queuedTransitions) {
      queuedTransition.onInputUnlock?.();
      queuedTransition.onSettle?.();
    }

    setPresentationPending(false);
    setPresentationSnapshot(null);
    setPresentedState(null);
    clearAllFlights();
  }, [clearAllFlights, presentationTimers]);
  const startNextTransition = useCallback(() => {
    if (activeTransitionRef.current) {
      return;
    }

    while (queuedTransitionsRef.current.length > 0) {
      const options = queuedTransitionsRef.current.shift();
      if (!options) {
        continue;
      }
      const {
        transactionId,
        previousState,
        nextState,
        action,
        actingPlayerId,
      } = options;
      const presentationTransaction = buildPresentationTransactionForTransition(
        transactionId,
        previousState,
        nextState,
        action,
        actingPlayerId
      );
      const presentationSequence = presentationTransaction
        ? buildAnimationSequence(presentationTransaction)
        : null;
      const settleMs = presentationSequence?.durationMs ?? 0;
      if (!presentationTransaction || !presentationSequence || settleMs <= 0) {
        setPresentedState(nextState);
        options.onInputUnlock?.();
        options.onSettle?.();
        continue;
      }

      const activeTransition: ActivePresentationTransition = {
        options,
        inputUnlocked: false,
        settled: false,
      };
      activeTransitionRef.current = activeTransition;
      setPresentationPending(true);
      setPresentedState(previousState);
      presentationTimers.clearAll();
      scheduleSequenceVisuals(presentationTransaction, presentationSequence);
      setPresentationSnapshot(
        derivePresentationSnapshotFromSequence({
          transaction: presentationTransaction,
          sequence: presentationSequence,
          elapsedMs: 0,
        })
      );

      const commitMs = presentationSequence.commitMs;
      const inputUnlockMs = presentationSequence.inputUnlockMs;
      scheduleSequencePresentationSnapshots(
        presentationTransaction,
        presentationSequence,
        settleMs,
        commitMs,
        presentationTimers,
        setPresentationSnapshot
      );
      const applyVisibleCommit = () => {
        if (activeTransitionRef.current !== activeTransition) {
          return;
        }
        setPresentedState(nextState);
        setPresentationSnapshot(null);
      };
      if (commitMs <= 0) {
        applyVisibleCommit();
      } else if (commitMs < settleMs) {
        presentationTimers.schedule(commitMs, applyVisibleCommit);
      }
      if (inputUnlockMs <= 0) {
        unlockPresentationTransition(activeTransition);
      } else if (inputUnlockMs < settleMs) {
        presentationTimers.schedule(inputUnlockMs, () => {
          if (activeTransitionRef.current === activeTransition) {
            unlockPresentationTransition(activeTransition);
          }
        });
      }
      presentationTimers.schedule(settleMs, () => {
        if (activeTransitionRef.current !== activeTransition) {
          return;
        }
        applyVisibleCommit();
        unlockPresentationTransition(activeTransition);
        setResourceFlights([]);
        setCardFlights([]);
        setPresentationSnapshot(null);
        clearTurnCycleVisuals();
        settlePresentationTransition(activeTransition);
        activeTransitionRef.current = null;

        if (queuedTransitionsRef.current.length === 0) {
          setPresentationPending(false);
          setPresentedState(null);
          return;
        }
        presentationTimers.schedule(0, () => {
          startNextTransitionRef.current();
        });
      });
      return;
    }

    setPresentationPending(false);
    setPresentationSnapshot(null);
    setPresentedState(null);
  }, [clearTurnCycleVisuals, presentationTimers, scheduleSequenceVisuals]);
  useEffect(() => {
    startNextTransitionRef.current = startNextTransition;
  }, [startNextTransition]);
  const enqueueTransition = useCallback((options: RunTransitionOptions) => {
    queuedTransitionsRef.current.push(options);
    setPresentationPending(true);
    if (!activeTransitionRef.current) {
      startNextTransitionRef.current();
    }
  }, []);
  const setAnimationsEnabled = useCallback(
    (nextEnabled: boolean) => {
      if (!nextEnabled) {
        finishPresentationQueue();
      }
      setEnabled(nextEnabled);
    },
    [finishPresentationQueue]
  );

  useEffect(() => {
    persistAnimationsEnabledPreference(enabled);
  }, [enabled]);

  useEffect(() => {
    return () => {
      presentationTimers.clearAll();
      turnCycleVisualTimers.clearAll();
      clearTaxPulseElements();
      queuedTransitionsRef.current = [];
      activeTransitionRef.current = null;
    };
  }, [clearTaxPulseElements, presentationTimers, turnCycleVisualTimers]);

  const presentationOverlays = presentationSnapshot?.overlays;

  return {
    enabled,
    setEnabled: setAnimationsEnabled,
    resourceFlights,
    cardFlights,
    incomeHighlightCardIds: presentationOverlays?.incomeHighlightCardIds ?? [],
    incomeHighlightCrowns: presentationOverlays?.incomeHighlightCrowns ?? [],
    diceVisualState: presentationOverlays?.dice ?? null,
    presentationSnapshot,
    presentedState,
    activePlayerHighlightOverride:
      presentationOverlays?.activePlayerHighlightOverride ?? null,
    presentationPending,
    clearPresentationQueue,
    clearAllFlights,
    enqueueTransition,
  };
}

function unlockPresentationTransition(
  transition: ActivePresentationTransition
): void {
  if (transition.inputUnlocked) {
    return;
  }
  transition.inputUnlocked = true;
  transition.options.onInputUnlock?.();
}

function settlePresentationTransition(
  transition: ActivePresentationTransition
): void {
  if (transition.settled) {
    return;
  }
  transition.settled = true;
  transition.options.onSettle?.();
}

function buildPresentationTransactionForTransition(
  transactionId: string,
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId
): GameTransaction | null {
  const events = deriveGamePresentationEvents(
    previousState,
    nextState,
    action,
    actingPlayerId
  );
  const shouldUseRuntime = events.some(isAnimatedPresentationEvent);
  if (!shouldUseRuntime) {
    return null;
  }

  return {
    id: transactionId,
    previousState,
    nextState,
    action,
    actingPlayerId,
    events,
  };
}

function isAnimatedPresentationEvent(
  event: ReturnType<typeof deriveGamePresentationEvents>[number]
): boolean {
  return (
    event.type === 'draw-card' ||
    event.type === 'card-sold' ||
    event.type === 'sell-resource-gained' ||
    event.type === 'resource-payment-started' ||
    event.type === 'resource-payment-applied' ||
    event.type === 'card-played-to-district' ||
    event.type === 'deed-token-paid' ||
    event.type === 'deed-progress-applied' ||
    event.type === 'deed-completed' ||
    event.type === 'trade-resources-applied' ||
    event.type === 'income-roll' ||
    event.type === 'tax-resolved' ||
    event.type === 'tax-token-lost' ||
    event.type === 'income-token-gained' ||
    event.type === 'income-choice-required' ||
    event.type === 'income-choice-submitted'
  );
}

function scheduleSequencePresentationSnapshots(
  transaction: GameTransaction,
  sequence: AnimationSequence,
  settleMs: number,
  finalBoundaryMs: number,
  timers: AnimationTimerRegistry,
  setSnapshot: (snapshot: PresentationSnapshot | null) => void
): void {
  const updateTimes = sequencePresentationSnapshotUpdateTimes(
    sequence,
    settleMs,
    finalBoundaryMs
  );
  for (const atMs of updateTimes) {
    timers.schedule(atMs, () => {
      setSnapshot(
        derivePresentationSnapshotFromSequence({
          transaction,
          sequence,
          elapsedMs: atMs,
        })
      );
    });
  }
}

export function sequencePresentationSnapshotUpdateTimes(
  sequence: AnimationSequence,
  settleMs: number,
  finalBoundaryMs: number
): number[] {
  return uniqueSequenceUpdateTimes(sequence.steps).filter(
    (atMs) => atMs > 0 && atMs < settleMs && atMs < finalBoundaryMs
  );
}

function uniqueSequenceUpdateTimes(
  steps: readonly ScheduledAnimationStep[]
): number[] {
  return [...new Set(steps.flatMap((step) => [step.startMs, step.endMs]))].sort(
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
