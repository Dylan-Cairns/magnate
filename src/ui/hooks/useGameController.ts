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
import type { BugReportActionEntry } from '../bugReport';
import { prepareCanonicalActionDispatch } from '../canonicalActionDispatcher';
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
import {
  transitionLogUpdate,
  type DeferredIncomeLogContext,
} from '../logTimeline';
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
  const [state, setState] = useState<GameState>(() =>
    createBrowserSession(
      makeBrowserSessionSeed(),
      humanPlayerId,
      devFixtureIdFromBrowserLocation()
    )
  );
  const [timelineLog, setTimelineLog] = useState<ReadonlyArray<GameLogEntry>>(
    () => withSeedLogPrefix(state, state.log, humanPlayerId)
  );
  const [error, setError] = useState<string | null>(null);
  const [actionHistory, setActionHistory] = useState<
    ReadonlyArray<BugReportActionEntry>
  >([]);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);
  const [turnResetActionHistoryAnchor, setTurnResetActionHistoryAnchor] =
    useState<ReadonlyArray<BugReportActionEntry> | null>(null);
  const stateRef = useRef(state);
  const nextActionOrdinalRef = useRef(0);
  const canonicalDispatchInProgressRef = useRef(false);
  const deferredIncomeLogContextRef = useRef<DeferredIncomeLogContext | null>(
    null
  );
  const commitCanonicalTransition = useCallback(
    (previousState: GameState, nextState: GameState, action: GameAction) => {
      const timelineUpdate = transitionLogUpdate(
        previousState,
        nextState,
        action,
        humanPlayerId,
        deferredIncomeLogContextRef.current
      );
      deferredIncomeLogContextRef.current =
        timelineUpdate.deferredIncomeLogContext;
      setTimelineLog((existing) => [...existing, ...timelineUpdate.entries]);
      stateRef.current = nextState;
      setState(nextState);
    },
    [humanPlayerId]
  );
  const {
    enabled: animationsEnabled,
    setEnabled: setAnimationsEnabled,
    resourceFlights,
    cardFlights,
    incomeHighlightCardIds,
    incomeHighlightCrowns,
    diceVisualState,
    presentationSnapshot,
    activePlayerHighlightOverride,
    actionCommitPending,
    allowHumanActionsWhileCommitPending,
    clearPendingActionCommit,
    clearAllFlights: clearAnimationFlights,
    runTransition: runAnimationTransition,
  } = useGameAnimations();
  const clearAllFlights = useCallback(() => {
    clearAnimationFlights();
  }, [clearAnimationFlights]);
  const dispatchAction = useCallback(
    (sourceState: GameState, action: GameAction, actingPlayerId: PlayerId) => {
      if (canonicalDispatchInProgressRef.current) {
        throw new Error('A canonical action dispatch is already in progress.');
      }

      canonicalDispatchInProgressRef.current = true;
      try {
        const plan = prepareCanonicalActionDispatch({
          currentState: stateRef.current,
          sourceState,
          action,
          actingPlayerId,
          actionOrdinal: nextActionOrdinalRef.current,
        });
        nextActionOrdinalRef.current += 1;
        setActionHistory((existing) => [
          ...existing,
          {
            turn: plan.previousState.turn,
            phase: plan.previousState.phase,
            actingPlayerId,
            action,
          },
        ]);
        commitCanonicalTransition(plan.previousState, plan.nextState, action);

        if (!animationsEnabled) {
          clearAllFlights();
          setError(null);
          return;
        }

        runAnimationTransition({
          transactionId: plan.transactionId,
          previousState: plan.previousState,
          nextState: plan.nextState,
          action,
          actingPlayerId,
        });
        setError(null);
      } finally {
        canonicalDispatchInProgressRef.current = false;
      }
    },
    [
      animationsEnabled,
      clearAllFlights,
      commitCanonicalTransition,
      runAnimationTransition,
    ]
  );

  const viewState = presentationSnapshot?.viewState ?? state;
  const terminal = isTerminal(state);
  const viewTerminal = isTerminal(viewState);
  const activePlayerId = activePlayerIdForState(state, humanPlayerId);
  const humanView = useMemo(
    () => toPlayerView(viewState, humanPlayerId),
    [humanPlayerId, viewState]
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
      canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor, {
        actionCommitPending,
      }),
    [actionCommitPending, activePlayerId, humanPlayerId, state, turnResetAnchor]
  );

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const shouldCapture = shouldCaptureTurnResetAnchor(
    state,
    activePlayerId,
    humanPlayerId,
    turnResetAnchor
  );
  const [prevShouldCapture, setPrevShouldCapture] = useState(false);

  if (shouldCapture !== prevShouldCapture) {
    setPrevShouldCapture(shouldCapture);
    if (shouldCapture) {
      setTurnResetAnchor({ turn: state.turn, playerId: humanPlayerId, state });
      setTurnResetTimelineAnchor(timelineLog);
      setTurnResetActionHistoryAnchor(actionHistory);
    }
  }

  const shouldRunBot = shouldScheduleBotAction({
    terminal,
    activePlayerId,
    botPlayerId,
    isIncomeChoicePhase: state.phase === 'CollectIncome',
    actionCommitPending,
    allowIncomeChoiceWhileCommitPending: allowHumanActionsWhileCommitPending,
    botIncomeActionCount: botIncomeActions.length,
    startupPreloadReady,
  });
  const [prevShouldRunBot, setPrevShouldRunBot] = useState(false);

  if (shouldRunBot !== prevShouldRunBot) {
    setPrevShouldRunBot(shouldRunBot);
    setBotThinking(shouldRunBot);
  }

  useEffect(() => {
    if (
      !shouldScheduleBotAction({
        terminal,
        activePlayerId,
        botPlayerId,
        isIncomeChoicePhase: state.phase === 'CollectIncome',
        actionCommitPending,
        allowIncomeChoiceWhileCommitPending:
          allowHumanActionsWhileCommitPending,
        botIncomeActionCount: botIncomeActions.length,
        startupPreloadReady,
      })
    ) {
      return;
    }

    let cancelled = false;
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
        const isCurrentIncomeChoicePhase = current.phase === 'CollectIncome';
        if (
          cancelled ||
          isTerminal(current) ||
          (isCurrentIncomeChoicePhase
            ? currentBotIncomeActions.length === 0
            : currentActive !== botPlayerId)
        ) {
          setBotThinking(false);
          return;
        }

        const actions = isCurrentIncomeChoicePhase
          ? currentBotIncomeActions
          : currentLegalActions;
        if (actions.length === 0) {
          setError('Bot has no legal actions.');
          setBotThinking(false);
          return;
        }

        try {
          const botView = isCurrentIncomeChoicePhase
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
          action.type === 'choose-income-suit'
            ? action.playerId
            : activePlayerId
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
      setTurnResetActionHistoryAnchor(null);
      deferredIncomeLogContextRef.current = null;
      clearPendingActionCommit();
      clearAllFlights();
      clearAllDeedTokenLayouts();

      try {
        const initialState = createBrowserSession(
          seed,
          humanPlayerId,
          devFixtureIdFromBrowserLocation()
        );
        stateRef.current = initialState;
        nextActionOrdinalRef.current = 0;
        canonicalDispatchInProgressRef.current = false;
        setState(initialState);
        setTimelineLog(
          withSeedLogPrefix(initialState, initialState.log, humanPlayerId)
        );
        setActionHistory([]);
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
      !canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor, {
        actionCommitPending,
      })
    ) {
      return;
    }

    clearPendingActionCommit();
    deferredIncomeLogContextRef.current = null;
    stateRef.current = turnResetAnchor.state;
    canonicalDispatchInProgressRef.current = false;
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
    setActionHistory(
      turnResetActionHistoryAnchor ? [...turnResetActionHistoryAnchor] : []
    );
    setError(null);
    setBotThinking(false);
    clearAllFlights();
    clearAllDeedTokenLayouts();
  }, [
    activePlayerId,
    actionCommitPending,
    clearAllFlights,
    clearPendingActionCommit,
    humanPlayerId,
    state,
    turnResetAnchor,
    turnResetActionHistoryAnchor,
    turnResetTimelineAnchor,
  ]);

  return {
    state,
    viewState,
    humanView,
    timelineLog,
    actionHistory,
    error,
    terminal: viewTerminal,
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
      incomeHighlightCardIds,
      incomeHighlightCrowns,
      diceVisualState,
      activePlayerHighlightOverride,
      actionCommitPending,
      allowHumanActionsWhileCommitPending,
    },
  };
}
