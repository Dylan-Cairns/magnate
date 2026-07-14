import { useCallback, useEffect, useRef, useState } from 'react';

import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import {
  createAnimationTimerRegistry,
  type AnimationTimerRegistry,
} from '../animations/animationTimerRegistry';
import { browserAnimationDomTargets } from '../animations/domTargets';
import {
  buildIncomeFlightsFromDom,
  buildPaymentFlightsFromDom,
  buildTaxLossFlightsFromDom,
  collectCardPlayFlights,
  collectDeedResourceFlights,
  type IncomeFlightToken,
} from '../animations/flightPlans';
import {
  cardFlightSettleMs,
  resourceFlightSettleMs,
} from '../animations/timing';
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
  previousState: GameState;
  nextState: GameState;
  action: GameAction;
  actingPlayerId: PlayerId;
  resourceFlights: readonly ResourceFlight[];
  cardFlights: readonly CardFlight[];
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
  const [presentationSnapshot, setPresentationSnapshot] =
    useState<PresentationSnapshot | null>(null);
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
  const scheduleSequenceVisualCommand = useCallback(
    (command: AnimationVisualCommand, transaction: GameTransaction) => {
      const scheduleAt = (atMs: number, callback: () => void) => {
        if (atMs <= 0) {
          callback();
          return;
        }
        turnCycleVisualTimers.schedule(atMs, callback);
      };

      switch (command.type) {
        case 'launch-draw-card-flight':
        case 'launch-sold-card-flight':
        case 'launch-card-to-district-flight':
          scheduleAt(command.atMs, () => {
            const flights = collectCardPlayFlights(
              transaction.previousState,
              transaction.nextState,
              transaction.action,
              transaction.actingPlayerId,
              makeCardFlightId
            );
            if (flights.length === 0) {
              return;
            }
            setCardFlights((existing) => [...existing, ...flights]);
          });
          return;
        case 'launch-payment-token-flights':
          scheduleAt(command.atMs, () => {
            const flights = buildPaymentFlightsFromDom(
              command.event,
              makeResourceFlightId
            );
            if (flights.length === 0) {
              return;
            }
            appendResourceFlightsWithCleanup(flights, command.durationMs);
          });
          return;
        case 'launch-deed-token-flights':
          scheduleAt(command.atMs, () => {
            const flights = collectDeedResourceFlights(
              transaction.previousState,
              transaction.action,
              transaction.actingPlayerId,
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
      appendResourceFlightsWithCleanup,
      applyTaxPulseTargets,
      clearTaxPulseElements,
      makeCardFlightId,
      makeResourceFlightId,
      turnCycleVisualTimers,
    ]
  );
  const scheduleSequenceVisuals = useCallback(
    (transaction: GameTransaction | null, sequence: AnimationSequence | null) => {
      clearTurnCycleVisuals();
      if (!transaction || !sequence) {
        return;
      }

      for (const command of deriveAnimationVisualCommands(sequence)) {
        scheduleSequenceVisualCommand(command, transaction);
      }
    },
    [clearTurnCycleVisuals, scheduleSequenceVisualCommand]
  );
  const clearPendingActionCommit = useCallback(() => {
    actionCommitTimers.clearAll();
    setActionCommitPending(false);
    setAllowHumanActionsWhileCommitPending(false);
    setPresentationSnapshot(null);
  }, [actionCommitTimers]);
  const clearAllFlights = useCallback(() => {
    setResourceFlights([]);
    setCardFlights([]);
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
      onSettle,
    }: RunTransitionOptions) => {
      const presentationTransaction = buildPresentationTransactionForTransition(
        previousState,
        nextState,
        action,
        actingPlayerId,
        queuedResourceFlights,
        queuedCardFlights
      );
      const presentationSequence = presentationTransaction
        ? buildAnimationSequence(presentationTransaction)
        : null;
      scheduleSequenceVisuals(presentationTransaction, presentationSequence);
      if (presentationTransaction && presentationSequence) {
        setPresentationSnapshot(
          derivePresentationSnapshotFromSequence({
            transaction: presentationTransaction,
            sequence: presentationSequence,
            elapsedMs: 0,
          })
        );
      } else {
        setPresentationSnapshot(null);
      }

      const settleMs = Math.max(
        resourceFlightSettleMs(queuedResourceFlights),
        cardFlightSettleMs(queuedCardFlights),
        presentationSequence?.durationMs ?? 0
      );
      if (settleMs <= 0) {
        setPresentationSnapshot(null);
        setAllowHumanActionsWhileCommitPending(false);
        onCommitTransition(previousState, nextState, action);
        onSettle?.();
        return;
      }

      const immediateResourceFlights =
        presentationSequence && action.type === 'choose-income-suit'
          ? []
          : queuedResourceFlights;
      if (immediateResourceFlights.length > 0) {
        setResourceFlights((existing) => [
          ...existing,
          ...immediateResourceFlights,
        ]);
      }
      if (queuedCardFlights.length > 0) {
        setCardFlights((existing) => [...existing, ...queuedCardFlights]);
      }
      setAllowHumanActionsWhileCommitPending(false);
      setActionCommitPending(true);
      actionCommitTimers.clearAll();
      if (presentationTransaction && presentationSequence) {
        scheduleSequencePresentationSnapshots(
          presentationTransaction,
          presentationSequence,
          settleMs,
          Math.min(
            presentationSequence.commitMs,
            presentationSequence.inputUnlockMs
          ),
          actionCommitTimers,
          setPresentationSnapshot
        );
      }
      let committed = false;
      const commitTransition = () => {
        if (committed) {
          return;
        }
        committed = true;
        onCommitTransition(previousState, nextState, action);
      };
      const commitMs = presentationSequence?.commitMs ?? settleMs;
      const inputUnlockMs = presentationSequence?.inputUnlockMs ?? commitMs;
      if (
        commitMs === inputUnlockMs &&
        commitMs > 0 &&
        commitMs < settleMs
      ) {
        // Commit and unlock are one user-visible boundary. Keep them in one
        // timer so React cannot paint a settled board while input is still
        // locked because an equal-time unlock callback has not run yet.
        actionCommitTimers.schedule(commitMs, () => {
          commitTransition();
          setPresentationSnapshot(null);
          setAllowHumanActionsWhileCommitPending(false);
          setActionCommitPending(false);
        });
      } else if (commitMs <= 0) {
        commitTransition();
      } else if (commitMs < settleMs) {
        actionCommitTimers.schedule(commitMs, commitTransition);
      }
      if (commitMs === inputUnlockMs && inputUnlockMs > 0) {
        // Handled by the combined boundary above, or by the settle callback
        // when the boundary coincides with final cleanup.
      } else if (inputUnlockMs <= 0) {
        commitTransition();
        setPresentationSnapshot(null);
        setActionCommitPending(false);
      } else if (inputUnlockMs < settleMs) {
        actionCommitTimers.schedule(inputUnlockMs, () => {
          commitTransition();
          setPresentationSnapshot(null);
          setAllowHumanActionsWhileCommitPending(false);
          setActionCommitPending(false);
        });
      }
      actionCommitTimers.schedule(settleMs, () => {
        commitTransition();
        setResourceFlights([]);
        setCardFlights([]);
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
      scheduleSequenceVisuals,
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

  const presentationOverlays = presentationSnapshot?.overlays;

  return {
    enabled,
    setEnabled,
    resourceFlights,
    cardFlights,
    incomeHighlightCardIds: presentationOverlays?.incomeHighlightCardIds ?? [],
    incomeHighlightCrowns: presentationOverlays?.incomeHighlightCrowns ?? [],
    diceVisualState: presentationOverlays?.dice ?? null,
    presentationSnapshot,
    activePlayerHighlightOverride:
      presentationOverlays?.activePlayerHighlightOverride ?? null,
    actionCommitPending,
    allowHumanActionsWhileCommitPending,
    makeResourceFlightId,
    makeCardFlightId,
    clearPendingActionCommit,
    clearAllFlights,
    runTransition,
  };
}

function buildPresentationTransactionForTransition(
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  resourceFlights: readonly ResourceFlight[],
  cardFlights: readonly CardFlight[]
): GameTransaction | null {
  const events = deriveGamePresentationEvents(
    previousState,
    nextState,
    action,
    actingPlayerId
  );
  const shouldUseRuntime =
    events.some(isAnimatedPresentationEvent) ||
    resourceFlights.length > 0 ||
    cardFlights.length > 0;
  if (!shouldUseRuntime) {
    return null;
  }

  return {
    id: `${previousState.seed}:${previousState.turn}:${previousState.phase}:${action.type}`,
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
