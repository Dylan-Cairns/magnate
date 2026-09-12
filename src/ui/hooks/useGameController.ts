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
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from '../../policies/catalog';
import type { SearchDecisionDiagnostics } from '../../policies/types';
import type { BugReportActionEntry } from '../bugReport';
import { prepareCanonicalActionDispatch } from '../canonicalActionDispatcher';
import { clearAllDeedTokenLayouts } from '../components/deedTokenLayout';
import {
  activePlayerIdForState,
  botDecisionResultIsCurrent,
  botRandomForState,
  botRandomSeedForState,
  createBrowserSession,
  errorMessage,
  humanActionsAcceptingInputForState,
  incomeChoiceActionsForPlayer,
  initialBrowserTimelineLog,
  makeBrowserSessionSeed,
  shouldScheduleBotAction,
  transitionOpensHumanDecisionWindow,
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
import { loadSavedGame, writeSavedGame, type SavedGame } from '../savedGame';

const DEFAULT_BOT_DELAY_MS = 450;
const BOT_DIAGNOSTICS_QUERY_KEY = 'botDiagnostics';
const BOT_PROFILE_STORAGE_KEY = 'magnate:botProfileId';

function readBotProfilePreference(): BotProfileId {
  if (typeof window === 'undefined') return DEFAULT_BOT_PROFILE_ID;
  try {
    const stored = window.localStorage.getItem(BOT_PROFILE_STORAGE_KEY);
    return (
      BOT_PROFILES.find((profile) => profile.id === stored && profile.available)
        ?.id ?? DEFAULT_BOT_PROFILE_ID
    );
  } catch {
    return DEFAULT_BOT_PROFILE_ID;
  }
}

function persistBotProfilePreference(profileId: BotProfileId): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(BOT_PROFILE_STORAGE_KEY, profileId);
  } catch {
    // Preferences remain usable when browser storage is unavailable.
  }
}

function browserBotDiagnosticsEnabled(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return (
    new URLSearchParams(window.location.search).get(
      BOT_DIAGNOSTICS_QUERY_KEY
    ) === '1'
  );
}

function closeActionPolicy(policy: unknown): void {
  if (
    typeof policy === 'object' &&
    policy !== null &&
    'close' in policy &&
    typeof policy.close === 'function'
  ) {
    policy.close();
  }
}

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
  const [initialSave] = useState(() =>
    devFixtureIdFromBrowserLocation()
      ? { save: null, error: null }
      : loadSavedGame(humanPlayerId)
  );
  const [gameId, setGameId] = useState(
    () => initialSave.save?.gameId ?? crypto.randomUUID()
  );
  const [storageError, setStorageError] = useState<string | null>(
    initialSave.error
  );
  const storageBlockedRef = useRef(initialSave.error !== null);
  const [awaitingResumeInput, setAwaitingResumeInput] = useState(
    Boolean(initialSave.save)
  );
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    () => initialSave.save?.botProfileId ?? readBotProfilePreference()
  );
  useEffect(() => {
    persistBotProfilePreference(botProfileId);
  }, [botProfileId]);
  const [state, setState] = useState<GameState>(
    () =>
      initialSave.save?.state ??
      createBrowserSession(
        makeBrowserSessionSeed(),
        humanPlayerId,
        devFixtureIdFromBrowserLocation()
      )
  );
  const [timelineLog, setTimelineLog] = useState<ReadonlyArray<GameLogEntry>>(
    () =>
      initialSave.save?.timelineLog ??
      initialBrowserTimelineLog(
        state,
        humanPlayerId,
        resolveBotProfile(botProfileId).selected.label
      )
  );
  const [error, setError] = useState<string | null>(null);
  const [actionHistory, setActionHistory] = useState<
    ReadonlyArray<BugReportActionEntry>
  >(initialSave.save?.actionHistory ?? []);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [humanInputBarrierOrdinal, setHumanInputBarrierOrdinal] = useState<
    number | null
  >(null);
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);
  const [turnResetActionHistoryAnchor, setTurnResetActionHistoryAnchor] =
    useState<ReadonlyArray<BugReportActionEntry> | null>(null);
  const stateRef = useRef(state);
  const nextActionOrdinalRef = useRef(0);
  const canonicalDispatchInProgressRef = useRef(false);
  const humanInputBarrierOrdinalRef = useRef<number | null>(null);
  const botDecisionGenerationRef = useRef(0);
  const deferredIncomeLogContextRef = useRef<DeferredIncomeLogContext | null>(
    initialSave.save?.deferredIncomeLogContext ?? null
  );
  const timelineLogRef = useRef(timelineLog);
  const actionHistoryRef = useRef(actionHistory);
  const checkpointRef = useRef<SavedGame | null>(initialSave.save);
  const persistCheckpoint = useCallback((checkpoint: SavedGame) => {
    checkpointRef.current = checkpoint;
    if (storageBlockedRef.current || devFixtureIdFromBrowserLocation()) return;
    setStorageError(writeSavedGame(checkpoint));
  }, []);

  useEffect(() => {
    // Only session creation runs here; action checkpoints are written synchronously.
    if (checkpointRef.current) return;
    persistCheckpoint({
      version: 1,
      gameId,
      humanPlayerId,
      botProfileId,
      state: stateRef.current,
      timelineLog: timelineLogRef.current,
      actionHistory: actionHistoryRef.current,
      deferredIncomeLogContext: deferredIncomeLogContextRef.current,
    });
  }, [botProfileId, gameId, humanPlayerId, persistCheckpoint]);

  const changeBotProfile = useCallback(
    (profileId: BotProfileId) => {
      resolveBotProfile(profileId);
      persistBotProfilePreference(profileId);
      setBotProfileId(profileId);
      if (checkpointRef.current)
        persistCheckpoint({
          ...checkpointRef.current,
          botProfileId: profileId,
        });
    },
    [persistCheckpoint]
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
      timelineLogRef.current = [
        ...timelineLogRef.current,
        ...timelineUpdate.entries,
      ];
      setTimelineLog(timelineLogRef.current);
      stateRef.current = nextState;
      setState(nextState);
      if (
        transitionOpensHumanDecisionWindow(
          previousState,
          nextState,
          humanPlayerId
        ) ||
        isTerminal(nextState)
      ) {
        persistCheckpoint({
          version: 1,
          gameId,
          humanPlayerId,
          botProfileId,
          state: nextState,
          timelineLog: timelineLogRef.current,
          actionHistory: actionHistoryRef.current,
          deferredIncomeLogContext: deferredIncomeLogContextRef.current,
        });
      }
    },
    [botProfileId, gameId, humanPlayerId, persistCheckpoint]
  );
  const {
    enabled: animationsEnabled,
    animateDeedProgress,
    setEnabled: setAnimationsEnabled,
    resourceFlights,
    cardFlights,
    tradeProgress,
    incomeHighlightCardIds,
    incomeHighlightCrowns,
    diceVisualState,
    presentationSnapshot,
    presentedState,
    activePlayerHighlightOverride,
    presentationPending,
    clearPresentationQueue,
    clearAllFlights: clearAnimationFlights,
    enqueueTransition: enqueueAnimationTransition,
  } = useGameAnimations();
  const clearAllFlights = useCallback(() => {
    clearAnimationFlights();
  }, [clearAnimationFlights]);
  const clearHumanInputBarrier = useCallback(() => {
    humanInputBarrierOrdinalRef.current = null;
    setHumanInputBarrierOrdinal(null);
  }, []);
  const releaseHumanInputBarrier = useCallback((actionOrdinal: number) => {
    if (humanInputBarrierOrdinalRef.current !== actionOrdinal) {
      return;
    }
    humanInputBarrierOrdinalRef.current = null;
    setHumanInputBarrierOrdinal(null);
  }, []);
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
        const opensHumanDecisionWindow = transitionOpensHumanDecisionWindow(
          plan.previousState,
          plan.nextState,
          humanPlayerId
        );
        if (animationsEnabled && opensHumanDecisionWindow) {
          humanInputBarrierOrdinalRef.current = plan.actionOrdinal;
          setHumanInputBarrierOrdinal(plan.actionOrdinal);
        }
        actionHistoryRef.current = [
          ...actionHistoryRef.current,
          {
            turn: plan.previousState.turn,
            phase: plan.previousState.phase,
            actingPlayerId,
            action,
          },
        ];
        setActionHistory(actionHistoryRef.current);
        commitCanonicalTransition(plan.previousState, plan.nextState, action);
        if (actingPlayerId === humanPlayerId) setAwaitingResumeInput(false);

        if (!animationsEnabled) {
          clearAllFlights();
          setError(null);
          return;
        }

        try {
          enqueueAnimationTransition({
            transactionId: plan.transactionId,
            previousState: plan.previousState,
            nextState: plan.nextState,
            action,
            actingPlayerId,
            onInputUnlock: () => {
              releaseHumanInputBarrier(plan.actionOrdinal);
            },
          });
        } catch (err) {
          clearPresentationQueue();
          clearHumanInputBarrier();
          throw err;
        }
        setError(null);
      } finally {
        canonicalDispatchInProgressRef.current = false;
      }
    },
    [
      animationsEnabled,
      clearAllFlights,
      clearHumanInputBarrier,
      clearPresentationQueue,
      commitCanonicalTransition,
      enqueueAnimationTransition,
      humanPlayerId,
      releaseHumanInputBarrier,
    ]
  );

  const viewState = presentationSnapshot?.viewState ?? presentedState ?? state;
  const terminal = isTerminal(state);
  const viewTerminal = isTerminal(viewState);
  const activePlayerId = activePlayerIdForState(state, humanPlayerId);
  const viewActivePlayerId = activePlayerIdForState(viewState, humanPlayerId);
  const humanInputReady = humanInputBarrierOrdinal === null;
  const humanView = useMemo(
    () => toPlayerView(viewState, humanPlayerId),
    [humanPlayerId, viewState]
  );
  const resolvedBotProfile = useMemo(
    () => resolveBotProfile(botProfileId),
    [botProfileId]
  );
  const collectBotDiagnostics = browserBotDiagnosticsEnabled();
  const humanActionsAcceptingInput = useMemo(
    () =>
      humanActionsAcceptingInputForState({
        state,
        humanPlayerId,
        humanInputReady,
      }),
    [humanInputReady, humanPlayerId, state]
  );
  const botIncomeActions = useMemo(
    () => incomeChoiceActionsForPlayer(legalActions(state), botPlayerId),
    [botPlayerId, state]
  );
  const canResetTurn = useMemo(
    () =>
      canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor, {
        humanInputReady,
      }),
    [activePlayerId, humanInputReady, humanPlayerId, state, turnResetAnchor]
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

  const shouldRunBot =
    !awaitingResumeInput &&
    shouldScheduleBotAction({
      terminal,
      activePlayerId,
      botPlayerId,
      isIncomeChoicePhase: state.phase === 'CollectIncome',
      botIncomeActionCount: botIncomeActions.length,
      startupPreloadReady,
    });
  const [prevShouldRunBot, setPrevShouldRunBot] = useState(false);

  if (shouldRunBot !== prevShouldRunBot) {
    setPrevShouldRunBot(shouldRunBot);
    setBotThinking(shouldRunBot);
  }

  useEffect(() => {
    const decisionGeneration = botDecisionGenerationRef.current + 1;
    botDecisionGenerationRef.current = decisionGeneration;
    if (!shouldRunBot) {
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
          botDecisionGenerationRef.current !== decisionGeneration ||
          isTerminal(current) ||
          (isCurrentIncomeChoicePhase
            ? currentBotIncomeActions.length === 0
            : currentActive !== botPlayerId)
        ) {
          if (botDecisionGenerationRef.current === decisionGeneration) {
            setBotThinking(false);
          }
          return;
        }

        const actions = isCurrentIncomeChoicePhase
          ? currentBotIncomeActions
          : currentLegalActions;
        if (actions.length === 0) {
          setError('Opponent has no legal actions.');
          setBotThinking(false);
          return;
        }

        let choice: GameAction | null | undefined;
        try {
          const botView = isCurrentIncomeChoicePhase
            ? toDecisionPlayerView(current, botPlayerId)
            : toPlayerView(current, botPlayerId);
          choice = await resolvedBotProfile.policy.selectAction({
            state: current,
            view: botView,
            legalActions: actions,
            random: botRandomForState(current, resolvedBotProfile.selected.id),
            randomSeed: botRandomSeedForState(
              current,
              resolvedBotProfile.selected.id
            ),
            ...(collectBotDiagnostics
              ? { onSearchDiagnostics: logBotSearchDiagnostics }
              : {}),
          });
        } catch (err) {
          if (
            botDecisionResultIsCurrent({
              cancelled,
              decisionGeneration,
              currentGeneration: botDecisionGenerationRef.current,
              decisionState: current,
              currentState: stateRef.current,
            })
          ) {
            setError(`Opponent action failed: ${errorMessage(err)}`);
            setBotThinking(false);
          }
          return;
        }

        if (
          !botDecisionResultIsCurrent({
            cancelled,
            decisionGeneration,
            currentGeneration: botDecisionGenerationRef.current,
            decisionState: current,
            currentState: stateRef.current,
          })
        ) {
          return;
        }
        if (!choice) {
          setError('Opponent policy could not select an action.');
          setBotThinking(false);
          return;
        }

        try {
          dispatchAction(
            current,
            choice,
            choice.type === 'choose-income-suit'
              ? choice.playerId
              : (currentActive ?? botPlayerId)
          );
        } catch (err) {
          if (
            !cancelled &&
            botDecisionGenerationRef.current === decisionGeneration
          ) {
            setError(`Opponent action failed: ${errorMessage(err)}`);
          }
        } finally {
          if (
            !cancelled &&
            botDecisionGenerationRef.current === decisionGeneration &&
            stateRef.current === current
          ) {
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
    activePlayerId,
    botPlayerId,
    collectBotDiagnostics,
    botIncomeActions.length,
    dispatchAction,
    resolvedBotProfile,
    shouldRunBot,
    state,
    startupPreloadReady,
    terminal,
  ]);

  useEffect(() => {
    const policy = resolvedBotProfile.policy;
    return () => {
      closeActionPolicy(policy);
    };
  }, [resolvedBotProfile.policy]);

  const performHumanAction = useCallback(
    (action: GameAction) => {
      const isHumanIncomeChoice =
        action.type === 'choose-income-suit' &&
        action.playerId === humanPlayerId;
      if (
        terminal ||
        !humanInputReady ||
        (activePlayerId !== humanPlayerId && !isHumanIncomeChoice)
      ) {
        return;
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
      activePlayerId,
      dispatchAction,
      humanInputReady,
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
      humanInputBarrierOrdinalRef.current = null;
      setHumanInputBarrierOrdinal(null);
      botDecisionGenerationRef.current += 1;
      closeActionPolicy(resolvedBotProfile.policy);
      clearPresentationQueue();
      clearAllFlights();
      clearAllDeedTokenLayouts();

      try {
        const initialState = createBrowserSession(
          seed,
          humanPlayerId,
          devFixtureIdFromBrowserLocation()
        );
        stateRef.current = initialState;
        const nextGameId = crypto.randomUUID();
        setGameId(nextGameId);
        setAwaitingResumeInput(false);
        storageBlockedRef.current = false;
        nextActionOrdinalRef.current = 0;
        canonicalDispatchInProgressRef.current = false;
        setState(initialState);
        timelineLogRef.current = initialBrowserTimelineLog(
          initialState,
          humanPlayerId,
          resolveBotProfile(botProfileId).selected.label
        );
        setTimelineLog(timelineLogRef.current);
        actionHistoryRef.current = [];
        setActionHistory([]);
        persistCheckpoint({
          version: 1,
          gameId: nextGameId,
          humanPlayerId,
          botProfileId,
          state: initialState,
          timelineLog: timelineLogRef.current,
          actionHistory: [],
          deferredIncomeLogContext: null,
        });
        setError(null);
        setBotThinking(false);
      } catch (err) {
        setError(`Failed to start game: ${errorMessage(err)}`);
      }
    },
    [
      botProfileId,
      clearAllFlights,
      clearPresentationQueue,
      humanPlayerId,
      persistCheckpoint,
      resolvedBotProfile.policy,
    ]
  );

  const resetTurn = useCallback(() => {
    if (!turnResetAnchor) {
      return;
    }
    if (
      !canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor, {
        humanInputReady,
      })
    ) {
      return;
    }

    humanInputBarrierOrdinalRef.current = null;
    setHumanInputBarrierOrdinal(null);
    botDecisionGenerationRef.current += 1;
    closeActionPolicy(resolvedBotProfile.policy);
    clearPresentationQueue();
    deferredIncomeLogContextRef.current = null;
    stateRef.current = turnResetAnchor.state;
    canonicalDispatchInProgressRef.current = false;
    setState(turnResetAnchor.state);
    timelineLogRef.current = turnResetTimelineAnchor
      ? [...turnResetTimelineAnchor]
      : initialBrowserTimelineLog(
          turnResetAnchor.state,
          humanPlayerId,
          resolveBotProfile(botProfileId).selected.label
        );
    setTimelineLog(timelineLogRef.current);
    actionHistoryRef.current = turnResetActionHistoryAnchor
      ? [...turnResetActionHistoryAnchor]
      : [];
    setActionHistory(actionHistoryRef.current);
    setError(null);
    setBotThinking(false);
    clearAllFlights();
    clearAllDeedTokenLayouts();
  }, [
    activePlayerId,
    clearAllFlights,
    clearPresentationQueue,
    humanInputReady,
    botProfileId,
    humanPlayerId,
    resolvedBotProfile.policy,
    state,
    turnResetAnchor,
    turnResetActionHistoryAnchor,
    turnResetTimelineAnchor,
  ]);

  return {
    gameId,
    storageError,
    state,
    viewState,
    humanView,
    timelineLog,
    actionHistory,
    error,
    terminal: viewTerminal,
    activePlayerId,
    viewActivePlayerId,
    botThinking,
    botProfileId,
    botStatusText: resolvedBotProfile.statusText,
    setBotProfileId: changeBotProfile,
    humanActionsAcceptingInput,
    humanInputBlockedByPresentation: !humanInputReady,
    canResetTurn,
    performHumanAction,
    resetSession,
    resetTurn,
    animations: {
      quietIncomeRollId:
        gameId === initialSave.save?.gameId
          ? initialSave.save.state.lastIncomeRoll?.rollId
          : undefined,
      enabled: animationsEnabled,
      animateDeedProgress,
      setEnabled: setAnimationsEnabled,
      resourceFlights,
      cardFlights,
      tradeProgress,
      incomeHighlightCardIds,
      incomeHighlightCrowns,
      diceVisualState,
      activePlayerHighlightOverride,
      presentationPending,
    },
  };
}
