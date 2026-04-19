import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { legalActions } from '../../engine/actionBuilders';
import { devFixtureIdFromBrowserLocation } from '../../dev/fixtures';
import {
  toDecisionPlayerView,
  turnOwnerIdForState,
} from '../../engine/decisionActor';
import { isTerminal } from '../../engine/scoring';
import type {
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
} from '../../engine/types';
import { toPlayerView } from '../../engine/view';
import type { BotProfileId } from '../../policies/catalog';
import {
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from '../../policies/catalog';
import type { SearchDecisionDiagnostics } from '../../policies/types';
import { prepareActionDispatch } from '../actionDispatcher';
import { clearAllDeedTokenLayouts } from '../components/deedTokenLayout';
import {
  activePlayerIdForState,
  botRandomForState,
  botRandomSeedForState,
  createBrowserSession,
  errorMessage,
  humanActionsAcceptingInputForState,
  incomeChoiceActionsForPlayer,
  makeBrowserSessionSeed,
  shouldScheduleBotAction,
  withSeedLogPrefix,
} from '../gameControllerModel';
import { transitionLogEntries } from '../logTimeline';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from '../turnReset';
import { useGameAnimations } from './useGameAnimations';

const DEFAULT_BOT_DELAY_MS = 450;

type UseGameControllerOptions = {
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
  startupPreloadReady: boolean;
};

function logBotSearchDiagnostics(diagnostics: SearchDecisionDiagnostics): void {
  const rootActions = diagnostics.rootActions.map((entry) => ({
    actionKey: entry.actionKey,
    visits: entry.visits,
    meanValue: roundDiagnosticNumber(entry.meanValue),
    terminalRate: roundDiagnosticNumber(entry.terminalRate),
    terminalRollouts: entry.terminalRollouts,
    prior: roundDiagnosticNumber(entry.prior),
  }));
  console.info('[Magnate bot search]', {
    heuristic: diagnostics.heuristic ?? 'v1',
    stochasticSimulation: diagnostics.stochasticSimulation ?? null,
    workers: diagnostics.parallelWorkers ?? 1,
    batches: diagnostics.parallelBatches ?? null,
    batchSize: diagnostics.parallelBatchSize ?? null,
    legalRootActions: diagnostics.legalRootActions,
    expandedRootActions: diagnostics.expandedRootActions,
    rootVisits: diagnostics.rootVisitBudget,
    simulatedActionSteps: diagnostics.simulatedActionSteps,
    maxSimulatedActionSteps: diagnostics.maxSimulatedActionSteps,
    terminalRollouts: diagnostics.terminalRollouts,
    terminalRate: diagnostics.terminalRate,
    selectedActionKey: diagnostics.selectedActionKey,
    selectedActionVisits: diagnostics.selectedActionVisits,
    selectedActionMeanValue: diagnostics.selectedActionMeanValue,
    selectedActionTerminalRate: diagnostics.selectedActionTerminalRate,
    rootActions,
  });
  if (rootActions.length > 0) {
    console.table(rootActions);
  }
}

function roundDiagnosticNumber(value: number): number {
  return Number(value.toFixed(4));
}

export function useGameController({
  humanPlayerId,
  botPlayerId,
  startupPreloadReady,
}: UseGameControllerOptions) {
  const devFixtureIdRef = useRef(devFixtureIdFromBrowserLocation());
  const [state, setState] = useState<GameState>(() =>
    createBrowserSession(
      makeBrowserSessionSeed(),
      humanPlayerId,
      devFixtureIdRef.current
    )
  );
  const [timelineLog, setTimelineLog] = useState<ReadonlyArray<GameLogEntry>>(
    () => withSeedLogPrefix(state, state.log, humanPlayerId)
  );
  const [error, setError] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);
  const stateRef = useRef(state);

  const commitTransition = useCallback(
    (previousState: GameState, nextState: GameState, action: GameAction) => {
      setTimelineLog((existing) => [
        ...existing,
        ...transitionLogEntries(previousState, nextState, action),
      ]);
      setState(nextState);
    },
    []
  );
  const {
    enabled: animationsEnabled,
    setEnabled: setAnimationsEnabled,
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
    clearAllFlights: clearAnimationFlights,
    runTransition: runAnimationTransition,
  } = useGameAnimations({
    onCommitTransition: commitTransition,
  });
  const clearAllFlights = useCallback(() => {
    clearAnimationFlights();
  }, [clearAnimationFlights]);
  const dispatchAction = useCallback(
    (
      previousState: GameState,
      action: GameAction,
      actingPlayerId: PlayerId
    ) => {
      const plan = prepareActionDispatch({
        previousState,
        action,
        actingPlayerId,
        animationsEnabled,
        makeResourceFlightId,
        makeCardFlightId,
      });
      if (!animationsEnabled) {
        clearAllFlights();
        commitTransition(previousState, plan.nextState, action);
        setError(null);
        return;
      }

      runAnimationTransition({
        previousState,
        nextState: plan.nextState,
        action,
        resourceFlights: plan.resourceFlights,
        cardFlights: plan.cardFlights,
        turnCyclePlan: plan.turnCyclePlan,
      });
      setError(null);
    },
    [
      animationsEnabled,
      clearAllFlights,
      commitTransition,
      makeCardFlightId,
      makeResourceFlightId,
      runAnimationTransition,
    ]
  );

  const terminal = isTerminal(state);
  const activePlayerId = activePlayerIdForState(state, humanPlayerId);
  const humanView = useMemo(
    () => toPlayerView(state, humanPlayerId),
    [humanPlayerId, state]
  );
  const resolvedBotProfile = useMemo(
    () => resolveBotProfile(botProfileId),
    [botProfileId]
  );
  const humanActionsAcceptingInput = useMemo(
    () =>
      humanActionsAcceptingInputForState({
        state,
        humanPlayerId,
        actionCommitPending,
        allowHumanActionsWhileCommitPending,
      }),
    [
      actionCommitPending,
      allowHumanActionsWhileCommitPending,
      humanPlayerId,
      state,
    ]
  );
  const botIncomeActions = useMemo(
    () => incomeChoiceActionsForPlayer(legalActions(state), botPlayerId),
    [botPlayerId, state]
  );
  const canResetTurn = useMemo(
    () =>
      canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor),
    [activePlayerId, humanPlayerId, state, turnResetAnchor]
  );

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    if (
      !shouldCaptureTurnResetAnchor(
        state,
        activePlayerId,
        humanPlayerId,
        turnResetAnchor
      )
    ) {
      return;
    }

    setTurnResetAnchor({
      turn: state.turn,
      playerId: humanPlayerId,
      state,
    });
    setTurnResetTimelineAnchor(timelineLog);
  }, [activePlayerId, humanPlayerId, state, timelineLog, turnResetAnchor]);

  useEffect(() => {
    if (
      !shouldScheduleBotAction({
        terminal,
        activePlayerId,
        botPlayerId,
        actionCommitPending,
        allowIncomeChoiceWhileCommitPending:
          allowHumanActionsWhileCommitPending,
        botIncomeActionCount: botIncomeActions.length,
        startupPreloadReady,
      })
    ) {
      setBotThinking(false);
      return;
    }

    let cancelled = false;
    setBotThinking(true);
    const botTurnDelayMs =
      resolvedBotProfile.selected.turnDelayMs ?? DEFAULT_BOT_DELAY_MS;
    const timerId = window.setTimeout(() => {
      void (async () => {
        const current = stateRef.current;
        const currentActive = turnOwnerIdForState(current);
        const currentLegalActions = legalActions(current);
        const currentBotIncomeActions = incomeChoiceActionsForPlayer(
          currentLegalActions,
          botPlayerId
        );
        if (
          cancelled ||
          isTerminal(current) ||
          (currentActive !== botPlayerId && currentBotIncomeActions.length === 0)
        ) {
          setBotThinking(false);
          return;
        }

        const actions =
          currentBotIncomeActions.length > 0
            ? currentBotIncomeActions
            : currentLegalActions;
        if (actions.length === 0) {
          setError('Bot has no legal actions.');
          setBotThinking(false);
          return;
        }

        try {
          const botView =
            currentBotIncomeActions.length > 0
              ? toDecisionPlayerView(current, botPlayerId)
              : toPlayerView(current, botPlayerId);
          const choice = await resolvedBotProfile.policy.selectAction({
            state: current,
            view: botView,
            legalActions: actions,
            random: botRandomForState(current, resolvedBotProfile.selected.id),
            randomSeed: botRandomSeedForState(
              current,
              resolvedBotProfile.selected.id
            ),
            onSearchDiagnostics: logBotSearchDiagnostics,
          });

          if (cancelled) {
            return;
          }
          if (!choice) {
            setError('Bot policy could not select an action.');
            return;
          }

          dispatchAction(
            current,
            choice,
            choice.type === 'choose-income-suit'
              ? choice.playerId
              : (currentActive ?? botPlayerId)
          );
        } catch (err) {
          if (!cancelled) {
            setError(`Bot action failed: ${errorMessage(err)}`);
          }
        } finally {
          if (!cancelled) {
            setBotThinking(false);
          }
        }
      })();
    }, botTurnDelayMs);

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [
    actionCommitPending,
    activePlayerId,
    botPlayerId,
    botIncomeActions.length,
    dispatchAction,
    allowHumanActionsWhileCommitPending,
    resolvedBotProfile,
    state,
    startupPreloadReady,
    terminal,
  ]);

  const performHumanAction = useCallback(
    (action: GameAction) => {
      const isHumanIncomeChoice =
        action.type === 'choose-income-suit' &&
        action.playerId === humanPlayerId;
      if (
        terminal ||
        (activePlayerId !== humanPlayerId && !isHumanIncomeChoice)
      ) {
        return;
      }
      if (actionCommitPending) {
        const canActDuringCommit =
          allowHumanActionsWhileCommitPending &&
          action.type === 'choose-income-suit';
        if (!canActDuringCommit) {
          return;
        }
      }

      try {
        dispatchAction(
          state,
          action,
          action.type === 'choose-income-suit' ? action.playerId : activePlayerId
        );
      } catch (err) {
        setError(errorMessage(err));
      }
    },
    [
      actionCommitPending,
      activePlayerId,
      allowHumanActionsWhileCommitPending,
      dispatchAction,
      humanPlayerId,
      state,
      terminal,
    ]
  );

  const resetSession = useCallback(
    (specifiedSeed?: string) => {
      const seed = specifiedSeed?.trim() || makeBrowserSessionSeed();
      setTurnResetAnchor(null);
      setTurnResetTimelineAnchor(null);
      clearPendingActionCommit();
      clearAllFlights();
      clearAllDeedTokenLayouts();

      try {
        const initialState = createBrowserSession(
          seed,
          humanPlayerId,
          devFixtureIdRef.current
        );
        setState(initialState);
        setTimelineLog(
          withSeedLogPrefix(initialState, initialState.log, humanPlayerId)
        );
        setError(null);
        setBotThinking(false);
      } catch (err) {
        setError(`Failed to start game: ${errorMessage(err)}`);
      }
    },
    [clearAllFlights, clearPendingActionCommit, humanPlayerId]
  );

  const resetTurn = useCallback(() => {
    if (!turnResetAnchor) {
      return;
    }
    if (
      !canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor)
    ) {
      return;
    }

    clearPendingActionCommit();
    setState(turnResetAnchor.state);
    setTimelineLog(
      turnResetTimelineAnchor
        ? [...turnResetTimelineAnchor]
        : withSeedLogPrefix(
            turnResetAnchor.state,
            turnResetAnchor.state.log,
            humanPlayerId
          )
    );
    setError(null);
    setBotThinking(false);
    clearAllFlights();
    clearAllDeedTokenLayouts();
  }, [
    activePlayerId,
    clearAllFlights,
    clearPendingActionCommit,
    humanPlayerId,
    state,
    turnResetAnchor,
    turnResetTimelineAnchor,
  ]);

  return {
    state,
    humanView,
    timelineLog,
    error,
    terminal,
    activePlayerId,
    botThinking,
    botProfileId,
    botStatusText: resolvedBotProfile.statusText,
    setBotProfileId,
    humanActionsAcceptingInput,
    canResetTurn,
    performHumanAction,
    resetSession,
    resetTurn,
    animations: {
      enabled: animationsEnabled,
      setEnabled: setAnimationsEnabled,
      resourceFlights,
      cardFlights,
      turnCycleOverlay,
      incomeHighlightCardIds,
      incomeHighlightCrowns,
      incomeResourcePreviewByPlayer,
      pendingDiscardHoldback,
      actionCommitPending,
      allowHumanActionsWhileCommitPending,
    },
  };
}
