import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { legalActions } from '../../engine/actionBuilders';
import { isTerminal, scoreLive } from '../../engine/scoring';
import type {
  FinalScore,
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
import { prepareActionDispatch } from '../actionDispatcher';
import { clearAllDeedTokenLayouts } from '../components/deedTokenLayout';
import {
  activePlayerIdForState,
  botRandomForState,
  createBrowserSession,
  errorMessage,
  humanActionsAcceptingInputForState,
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

export function useGameController({
  humanPlayerId,
  botPlayerId,
  startupPreloadReady,
}: UseGameControllerOptions) {
  const [state, setState] = useState<GameState>(() =>
    createBrowserSession(makeBrowserSessionSeed(), humanPlayerId)
  );
  const [timelineLog, setTimelineLog] = useState<ReadonlyArray<GameLogEntry>>(
    () => withSeedLogPrefix(state, state.log, humanPlayerId)
  );
  const [error, setError] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [terminalWinnerOverlayWinner, setTerminalWinnerOverlayWinner] =
    useState<FinalScore['winner'] | null>(null);
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);
  const stateRef = useRef(state);

  const clearTerminalWinnerOverlay = useCallback(() => {
    setTerminalWinnerOverlayWinner(null);
  }, []);
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
    clearTerminalWinnerOverlay();
  }, [clearAnimationFlights, clearTerminalWinnerOverlay]);
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
        onSettle: plan.enteredTerminal
          ? () => {
              setTerminalWinnerOverlayWinner(
                (plan.nextState.finalScore ?? scoreLive(plan.nextState)).winner
              );
            }
          : undefined,
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
  const canResetTurn = useMemo(
    () =>
      canUseTurnReset(state, activePlayerId, humanPlayerId, turnResetAnchor),
    [activePlayerId, humanPlayerId, state, turnResetAnchor]
  );

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    if (!terminal) {
      clearTerminalWinnerOverlay();
    }
  }, [clearTerminalWinnerOverlay, terminal]);

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
        const currentActive = current.players[current.activePlayerIndex]?.id;
        if (cancelled || isTerminal(current) || currentActive !== botPlayerId) {
          setBotThinking(false);
          return;
        }

        const actions = legalActions(current);
        if (actions.length === 0) {
          setError('Bot has no legal actions.');
          setBotThinking(false);
          return;
        }

        try {
          const botView = toPlayerView(current, botPlayerId);
          const choice = await resolvedBotProfile.policy.selectAction({
            state: current,
            view: botView,
            legalActions: actions,
            random: botRandomForState(current, resolvedBotProfile.selected.id),
          });

          if (cancelled) {
            return;
          }
          if (!choice) {
            setError('Bot policy could not select an action.');
            return;
          }

          dispatchAction(current, choice, currentActive);
        } catch (err) {
          setError(`Bot action failed: ${errorMessage(err)}`);
        } finally {
          setBotThinking(false);
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
    dispatchAction,
    resolvedBotProfile,
    state,
    startupPreloadReady,
    terminal,
  ]);

  const performHumanAction = useCallback(
    (action: GameAction) => {
      if (terminal || activePlayerId !== humanPlayerId) {
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
        dispatchAction(state, action, activePlayerId);
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
        const initialState = createBrowserSession(seed, humanPlayerId);
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
    terminalWinnerOverlayWinner,
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
