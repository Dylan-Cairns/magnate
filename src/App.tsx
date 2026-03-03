import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { legalActions } from './engine/actionBuilders';
import type { CardId } from './engine/cards';
import { rngFromSeed } from './engine/rng';
import { createSession, stepToDecision } from './engine/session';
import {
  districtWinnersByPlayer,
  isTerminal,
  scoreLive,
} from './engine/scoring';
import type { BotProfileId } from './policies/catalog';
import { DEFAULT_BOT_PROFILE_ID, resolveBotProfile } from './policies/catalog';
import type {
  FinalScore,
  GameAction,
  GameLogEntry,
  GameState,
  PlayerId,
  Suit,
} from './engine/types';
import { toPlayerView } from './engine/view';
import {
  buildHumanActionList,
  buildTradeSourceGroups,
  pickerStillLegal,
} from './ui/actionPresentation';
import {
  developOutrightCompositePickerStillLegal,
  toPickerQuery,
  tradeCompositePickerStillLegal,
  type ActionPickerState,
} from './ui/actionPickerModel';
import { cardFlightSettleMs } from './ui/animations/timing';
import { collectTurnCycleAnimationPlan } from './ui/animations/turnCycleVisualPlan';
import {
  collectCardPlayFlights,
  collectDeedResourceFlights,
  collectIncomeChoiceResourceFlights,
  collectTerminalCleanupFlights,
} from './ui/animations/flightPlans';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from './ui/turnReset';
import {
  preloadStartupAssets,
  type StartupPreloadProgress,
} from './ui/startupPreload';
import { ActionPicker } from './ui/components/ActionPicker';
import { ActionsPanel } from './ui/components/ActionsPanel';
import { CardFlightLayer } from './ui/components/CardFlightLayer';
import { DeckPiles } from './ui/components/DeckPiles';
import {
  ResolutionWarningOverlay,
  StartupPreloadOverlay,
  TurnCycleOverlay,
} from './ui/components/GameOverlays';
import { LogPanel } from './ui/components/LogPanel';
import { OptionsBackdrop, OptionsMenu } from './ui/components/OptionsMenu';
import { ResourceFlightLayer } from './ui/components/ResourceFlightLayer';
import { clearAllDeedTokenLayouts } from './ui/components/deedTokenLayout';
import { DistrictColumn, PlayerTokenRail } from './ui/components/DistrictBoard';
import { PlayerPanel } from './ui/components/PlayerPanel';
import { RollResult } from './ui/components/RollResult';
import { useDismissableLayer } from './ui/hooks/useDismissableLayer';
import { useGameAnimations } from './ui/hooks/useGameAnimations';
import { transitionLogEntries } from './ui/logTimeline';

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const DEFAULT_BOT_DELAY_MS = 450;
const PLAYER_HAND_SLOT_COUNT = 3;
const TRADE_POPOVER_WIDTH_PX = 220;
const TRADE_POPOVER_MIN_HEIGHT_PX = 188;
const TRADE_POPOVER_GAP_PX = 8;
const VIEWPORT_PADDING_PX = 10;
const RESOLUTION_WARNING_BASE_WIDTH_PX = 1920;
const RESOLUTION_WARNING_BASE_HEIGHT_PX = 1080;
const RESOLUTION_WARNING_THRESHOLD_SCALE = 0.9;
const STARTUP_PRELOAD_INITIAL_PROGRESS: StartupPreloadProgress = {
  completed: 0,
  total: 1,
  percent: 0,
  message: 'Loading card images and bot models...',
};

function makeSeed(): string {
  return `seed-${Date.now()}`;
}

function shouldShowResolutionWarningOnLoad(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  const minimumWidthPx =
    RESOLUTION_WARNING_BASE_WIDTH_PX * RESOLUTION_WARNING_THRESHOLD_SCALE;
  const minimumHeightPx =
    RESOLUTION_WARNING_BASE_HEIGHT_PX * RESOLUTION_WARNING_THRESHOLD_SCALE;
  const resolutionWidth = window.screen?.width ?? window.innerWidth;
  const resolutionHeight = window.screen?.height ?? window.innerHeight;
  return (
    resolutionWidth <= minimumWidthPx || resolutionHeight <= minimumHeightPx
  );
}

function createInitialState(seed: string): GameState {
  return createSession(seed, HUMAN_PLAYER);
}

function withSeedLogPrefix(
  state: GameState,
  entries: readonly GameLogEntry[]
): ReadonlyArray<GameLogEntry> {
  const seedSummary = `Seed ${state.seed}`;
  if (entries[0]?.summary === seedSummary) {
    return [...entries];
  }
  const activePlayerId =
    state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  return [
    {
      turn: state.turn,
      player: activePlayerId,
      phase: state.phase,
      summary: seedSummary,
    },
    ...entries,
  ];
}

function botRandomForState(state: GameState, profileId: string): () => number {
  return rngFromSeed(
    `${state.seed}:bot:${profileId}:turn:${state.turn}:phase:${state.phase}:log:${state.log.length}:actor:${state.activePlayerIndex}`
  );
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function App() {
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [state, setState] = useState<GameState>(() =>
    createInitialState(makeSeed())
  );
  const [timelineLog, setTimelineLog] = useState<ReadonlyArray<GameLogEntry>>(
    () => withSeedLogPrefix(state, state.log)
  );
  const [error, setError] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [actionPicker, setActionPicker] = useState<ActionPickerState | null>(
    null
  );
  const [optionsMenuOpen, setOptionsMenuOpen] = useState<boolean>(false);
  const [startupPreloadReady, setStartupPreloadReady] =
    useState<boolean>(false);
  const [startupPreloadError, setStartupPreloadError] = useState<string | null>(
    null
  );
  const [startupPreloadAttempt, setStartupPreloadAttempt] = useState<number>(0);
  const [startupPreloadProgress, setStartupPreloadProgress] =
    useState<StartupPreloadProgress>(STARTUP_PRELOAD_INITIAL_PROGRESS);
  const [resolutionWarningOpen, setResolutionWarningOpen] = useState<boolean>(
    shouldShowResolutionWarningOnLoad
  );
  const [terminalWinnerOverlayWinner, setTerminalWinnerOverlayWinner] =
    useState<FinalScore['winner'] | null>(null);
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);

  const stateRef = useRef(state);
  const actionPopoverRef = useRef<HTMLElement | null>(null);
  const optionsMenuRef = useRef<HTMLElement | null>(null);
  const optionsMenuButtonRef = useRef<HTMLButtonElement | null>(null);
  const seedInputRef = useRef<HTMLInputElement | null>(null);
  const closeActionPicker = useCallback(() => setActionPicker(null), []);
  const closeOptionsMenu = useCallback(() => setOptionsMenuOpen(false), []);
  const retryStartupPreload = useCallback(
    () => setStartupPreloadAttempt((current) => current + 1),
    []
  );
  const clearTerminalWinnerOverlay = useCallback(() => {
    setTerminalWinnerOverlayWinner(null);
  }, []);
  const commitImmediateTransition = useCallback(
    (previousState: GameState, nextState: GameState, action?: GameAction) => {
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
    onCommitTransition: commitImmediateTransition,
  });
  const clearAllFlights = useCallback(() => {
    clearAnimationFlights();
    clearTerminalWinnerOverlay();
  }, [clearAnimationFlights, clearTerminalWinnerOverlay]);
  const actionPopoverLayerRefs = useMemo(() => [actionPopoverRef], []);
  const optionsMenuLayerRefs = useMemo(
    () => [optionsMenuRef, optionsMenuButtonRef],
    []
  );
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    let cancelled = false;
    setStartupPreloadReady(false);
    setStartupPreloadError(null);
    setStartupPreloadProgress(STARTUP_PRELOAD_INITIAL_PROGRESS);

    void preloadStartupAssets({
      onProgress(progress) {
        if (cancelled) {
          return;
        }
        setStartupPreloadProgress(progress);
      },
    })
      .then(() => {
        if (cancelled) {
          return;
        }
        setStartupPreloadReady(true);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setStartupPreloadError(errorMessage(err));
      });

    return () => {
      cancelled = true;
    };
  }, [startupPreloadAttempt]);

  const terminal = isTerminal(state);
  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
  const activePlayerId =
    state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
  const resolvedBotProfile = useMemo(
    () => resolveBotProfile(botProfileId),
    [botProfileId]
  );
  const score = useMemo(() => state.finalScore ?? scoreLive(state), [state]);
  const wonDistrictsByPlayer = useMemo(
    () => districtWinnersByPlayer(state),
    [state]
  );
  const humanPlayer = humanView.players.find(
    (player) => player.id === HUMAN_PLAYER
  );
  const botPlayer = humanView.players.find(
    (player) => player.id === BOT_PLAYER
  );
  const pendingIncomeChoiceCardIds = useMemo(
    () => (state.pendingIncomeChoices ?? []).map((choice) => choice.cardId),
    [state.pendingIncomeChoices]
  );
  const incomeHighlightCardIdSet = useMemo(
    () => new Set([...incomeHighlightCardIds, ...pendingIncomeChoiceCardIds]),
    [incomeHighlightCardIds, pendingIncomeChoiceCardIds]
  );
  const incomeHighlightCrownSuitsByPlayer = useMemo(() => {
    const byPlayer = new Map<PlayerId, Set<Suit>>([
      [HUMAN_PLAYER, new Set<Suit>()],
      [BOT_PLAYER, new Set<Suit>()],
    ]);
    for (const target of incomeHighlightCrowns) {
      byPlayer.get(target.playerId)?.add(target.suit);
    }
    return byPlayer;
  }, [incomeHighlightCrowns]);

  const humanActions = useMemo(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      return [] as readonly GameAction[];
    }
    return legalActions(state);
  }, [activePlayerId, state, terminal]);
  const humanActionsAcceptingInput = useMemo(() => {
    if (!actionCommitPending) {
      return humanActions;
    }
    if (!allowHumanActionsWhileCommitPending) {
      return [] as readonly GameAction[];
    }
    return humanActions.filter(
      (action) => action.type === 'choose-income-suit'
    );
  }, [actionCommitPending, allowHumanActionsWhileCommitPending, humanActions]);

  const humanActionItems = useMemo(
    () => buildHumanActionList(humanActionsAcceptingInput),
    [humanActionsAcceptingInput]
  );
  const tradeSourceGroups = useMemo(
    () => buildTradeSourceGroups(humanActionsAcceptingInput),
    [humanActionsAcceptingInput]
  );
  const hasMultipleTradeSources = tradeSourceGroups.length > 1;
  const firstTradeGroupIndex = useMemo(
    () => humanActionItems.findIndex((item) => item.kind === 'trade-group'),
    [humanActionItems]
  );
  const visibleHumanActionItems = useMemo(() => {
    if (!hasMultipleTradeSources) {
      return humanActionItems;
    }
    return humanActionItems.filter(
      (item, index) =>
        item.kind !== 'trade-group' || index === firstTradeGroupIndex
    );
  }, [firstTradeGroupIndex, hasMultipleTradeSources, humanActionItems]);

  const canResetTurn = useMemo(
    () => canUseTurnReset(state, activePlayerId, HUMAN_PLAYER, turnResetAnchor),
    [activePlayerId, state, turnResetAnchor]
  );
  const isTurnCycleAnimationLock =
    actionCommitPending && allowHumanActionsWhileCommitPending;
  const showBotThinkingDuringIncomeChoiceLock =
    isTurnCycleAnimationLock &&
    activePlayerId === BOT_PLAYER &&
    state.phase === 'CollectIncome' &&
    (state.pendingIncomeChoices?.length ?? 0) > 0;
  const hideBotWaitMessageDuringTurnCycleLock =
    isTurnCycleAnimationLock && !showBotThinkingDuringIncomeChoiceLock;
  const humanActionUiBlockedByAnimation =
    activePlayerId === HUMAN_PLAYER &&
    actionCommitPending &&
    humanActionsAcceptingInput.length === 0;
  const humanActionUiBlockedByTurnCycleAnimation =
    humanActionUiBlockedByAnimation && isTurnCycleAnimationLock;

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
        HUMAN_PLAYER,
        turnResetAnchor
      )
    ) {
      return;
    }

    setTurnResetAnchor({
      turn: state.turn,
      playerId: HUMAN_PLAYER,
      state,
    });
    setTurnResetTimelineAnchor(timelineLog);
  }, [activePlayerId, state, timelineLog, turnResetAnchor]);

  useEffect(() => {
    if (
      terminal ||
      activePlayerId !== BOT_PLAYER ||
      actionCommitPending ||
      !startupPreloadReady
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
        if (cancelled || isTerminal(current) || currentActive !== BOT_PLAYER) {
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
          const botView = toPlayerView(current, BOT_PLAYER);
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

          const actingPlayerId =
            current.players[current.activePlayerIndex]?.id ?? BOT_PLAYER;
          const queuedActionResourceFlights = [
            ...collectDeedResourceFlights(
              current,
              choice,
              actingPlayerId,
              makeResourceFlightId
            ),
            ...collectIncomeChoiceResourceFlights(choice, makeResourceFlightId),
          ];
          const next = stepToDecision(current, choice);
          if (!animationsEnabled) {
            clearAllFlights();
            commitImmediateTransition(current, next, choice);
            setError(null);
            return;
          }
          let queuedCardFlights = collectCardPlayFlights(
            current,
            next,
            choice,
            actingPlayerId,
            makeCardFlightId
          );
          const terminalCleanupPlan = collectTerminalCleanupFlights(
            current,
            next,
            makeResourceFlightId,
            makeCardFlightId
          );
          const enteredTerminal = isTerminal(next);
          const turnCyclePlan = collectTurnCycleAnimationPlan(
            current,
            next,
            choice,
            cardFlightSettleMs(queuedCardFlights)
          );
          const queuedResourceFlights = [...queuedActionResourceFlights];
          if (terminalCleanupPlan) {
            queuedResourceFlights.push(...terminalCleanupPlan.resourceFlights);
            queuedCardFlights = [
              ...queuedCardFlights,
              ...terminalCleanupPlan.cardFlights,
            ];
          }
          runAnimationTransition({
            previousState: current,
            nextState: next,
            action: choice,
            resourceFlights: queuedResourceFlights,
            cardFlights: queuedCardFlights,
            turnCyclePlan,
            onSettle: enteredTerminal
              ? () => {
                  setTerminalWinnerOverlayWinner(
                    (next.finalScore ?? scoreLive(next)).winner
                  );
                }
              : undefined,
          });
          setError(null);
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
    animationsEnabled,
    clearAllFlights,
    commitImmediateTransition,
    makeCardFlightId,
    makeResourceFlightId,
    resolvedBotProfile,
    runAnimationTransition,
    state,
    startupPreloadReady,
    terminal,
  ]);

  useEffect(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      closeActionPicker();
      return;
    }
    if (humanActionUiBlockedByAnimation) {
      closeActionPicker();
      return;
    }

    if (actionPicker) {
      if (actionPicker.kind === 'trade-combined') {
        if (
          !tradeCompositePickerStillLegal(
            actionPicker,
            humanActionsAcceptingInput
          )
        ) {
          closeActionPicker();
        }
        return;
      }
      if (actionPicker.kind === 'develop-outright-combined') {
        if (
          !developOutrightCompositePickerStillLegal(
            actionPicker,
            humanActionsAcceptingInput
          )
        ) {
          closeActionPicker();
        }
        return;
      }
      const stillLegal = pickerStillLegal(
        toPickerQuery(actionPicker),
        humanActionsAcceptingInput
      );
      if (!stillLegal) {
        closeActionPicker();
      }
    }
  }, [
    activePlayerId,
    humanActionsAcceptingInput,
    terminal,
    actionPicker,
    closeActionPicker,
    humanActionUiBlockedByAnimation,
  ]);

  useDismissableLayer({
    enabled: Boolean(actionPicker),
    onDismiss: closeActionPicker,
    insideRefs: actionPopoverLayerRefs,
    closeOnScroll: true,
  });

  useDismissableLayer({
    enabled: optionsMenuOpen,
    onDismiss: closeOptionsMenu,
    insideRefs: optionsMenuLayerRefs,
  });

  const handleHumanAction = (action: GameAction) => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
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
      const next = stepToDecision(state, action);
      if (!animationsEnabled) {
        clearAllFlights();
        commitImmediateTransition(state, next, action);
        setError(null);
        return;
      }
      const queuedActionResourceFlights = [
        ...collectDeedResourceFlights(
          state,
          action,
          activePlayerId,
          makeResourceFlightId
        ),
        ...collectIncomeChoiceResourceFlights(action, makeResourceFlightId),
      ];
      let queuedCardFlights = collectCardPlayFlights(
        state,
        next,
        action,
        activePlayerId,
        makeCardFlightId
      );
      const terminalCleanupPlan = collectTerminalCleanupFlights(
        state,
        next,
        makeResourceFlightId,
        makeCardFlightId
      );
      const enteredTerminal = isTerminal(next);
      const turnCyclePlan = collectTurnCycleAnimationPlan(
        state,
        next,
        action,
        cardFlightSettleMs(queuedCardFlights)
      );
      const queuedResourceFlights = [...queuedActionResourceFlights];
      if (terminalCleanupPlan) {
        queuedResourceFlights.push(...terminalCleanupPlan.resourceFlights);
        queuedCardFlights = [
          ...queuedCardFlights,
          ...terminalCleanupPlan.cardFlights,
        ];
      }
      runAnimationTransition({
        previousState: state,
        nextState: next,
        action,
        resourceFlights: queuedResourceFlights,
        cardFlights: queuedCardFlights,
        turnCyclePlan,
        onSettle: enteredTerminal
          ? () => {
              setTerminalWinnerOverlayWinner(
                (next.finalScore ?? scoreLive(next)).winner
              );
            }
          : undefined,
      });
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleReset = () => {
    const specifiedSeed = seedInputRef.current?.value.trim() ?? '';
    if (seedInputRef.current) {
      seedInputRef.current.value = '';
    }
    const seed = specifiedSeed || makeSeed();
    closeActionPicker();
    closeOptionsMenu();
    setTurnResetAnchor(null);
    setTurnResetTimelineAnchor(null);
    clearPendingActionCommit();
    clearAllFlights();
    clearAllDeedTokenLayouts();

    try {
      const initialState = createInitialState(seed);
      setState(initialState);
      setTimelineLog(withSeedLogPrefix(initialState, initialState.log));
      setError(null);
      setBotThinking(false);
    } catch (err) {
      setError(`Failed to start game: ${errorMessage(err)}`);
    }
  };

  const handlePickerSelection = (action: GameAction) => {
    closeActionPicker();
    handleHumanAction(action);
  };

  const handleTurnReset = () => {
    if (!turnResetAnchor) {
      return;
    }
    if (
      !canUseTurnReset(state, activePlayerId, HUMAN_PLAYER, turnResetAnchor)
    ) {
      return;
    }

    closeActionPicker();
    clearPendingActionCommit();
    setState(turnResetAnchor.state);
    setTimelineLog(
      turnResetTimelineAnchor
        ? [...turnResetTimelineAnchor]
        : withSeedLogPrefix(turnResetAnchor.state, turnResetAnchor.state.log)
    );
    setError(null);
    setBotThinking(false);
    clearAllFlights();
    clearAllDeedTokenLayouts();
  };

  const openTradePicker = (
    give: Suit,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({ kind: 'trade', give, ...position });
  };

  const openTradeCombinedPicker = (trigger: HTMLButtonElement) => {
    const position = pickerPosition(
      trigger,
      Math.max(2, tradeSourceGroups.length + 1)
    );
    setActionPicker({
      kind: 'trade-combined',
      ...position,
    });
  };

  const openDistrictPicker = (
    config: {
      actionType: 'buy-deed';
      cardId: CardId;
    },
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'district',
      actionType: config.actionType,
      cardId: config.cardId,
      ...position,
    });
  };

  const openDevelopOutrightCombinedPicker = (
    cardId: CardId,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'develop-outright-combined',
      cardId,
      ...position,
    });
  };

  const openDevelopOutrightDistrictOnlyPicker = (
    cardId: CardId,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'develop-outright-district',
      cardId,
      ...position,
    });
  };

  const openDeedPaymentPicker = (
    config: { cardId: CardId; districtId: string },
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'deed-payment',
      cardId: config.cardId,
      districtId: config.districtId,
      ...position,
    });
  };

  const pickerPosition = (
    trigger: HTMLButtonElement,
    optionCount: number
  ): { top: number; left: number } => {
    const rect = trigger.getBoundingClientRect();
    const rowCount = Math.max(1, Math.ceil(optionCount / 2));
    const estimatedHeight = Math.max(
      TRADE_POPOVER_MIN_HEIGHT_PX,
      116 + rowCount * 46
    );
    const maxLeft =
      window.innerWidth - TRADE_POPOVER_WIDTH_PX - VIEWPORT_PADDING_PX;
    const maxTop = window.innerHeight - estimatedHeight - VIEWPORT_PADDING_PX;

    const left = clamp(
      rect.right + TRADE_POPOVER_GAP_PX,
      VIEWPORT_PADDING_PX,
      maxLeft
    );
    const top = clamp(rect.top, VIEWPORT_PADDING_PX, maxTop);

    return { left, top };
  };

  if (!humanPlayer || !botPlayer) {
    return (
      <div className="app-shell">
        <section className="panel">
          <h1>Magnate</h1>
          <p>Could not load player data.</p>
        </section>
      </div>
    );
  }

  const humanPreviewResources = incomeResourcePreviewByPlayer?.[HUMAN_PLAYER];
  const botPreviewResources = incomeResourcePreviewByPlayer?.[BOT_PLAYER];
  const humanRailPlayer = humanPreviewResources
    ? {
        ...humanPlayer,
        resources: humanPreviewResources,
      }
    : humanPlayer;
  const botRailPlayer = botPreviewResources
    ? {
        ...botPlayer,
        resources: botPreviewResources,
      }
    : botPlayer;

  return (
    <div className={`app-shell${optionsMenuOpen ? ' is-options-open' : ''}`}>
      {error && (
        <section className="error-banner">
          <strong>Engine Error:</strong> {error}
        </section>
      )}

      <main className="layout">
        <aside className="actions-pane">
          <div className="brand-row">
            <section className="panel brand-panel">
              <div className="brand-header">
                <div className="brand-title-block">
                  <h1>Magnate</h1>
                  <p className="brand-subtitle">A Decktet game</p>
                  <a
                    className="brand-options-link"
                    href="http://decktet.wikidot.com/game:magnate"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Rules
                  </a>
                </div>
              </div>
            </section>
          </div>

          <ActionsPanel
            terminal={terminal}
            isLastTurn={isLastTurn}
            score={score}
            wonDistrictsByPlayer={wonDistrictsByPlayer}
            activePlayerId={activePlayerId}
            humanPlayerId={HUMAN_PLAYER}
            botPlayerId={BOT_PLAYER}
            visibleActionItems={visibleHumanActionItems}
            hasMultipleTradeSources={hasMultipleTradeSources}
            actionPicker={actionPicker}
            canResetTurn={canResetTurn}
            botThinking={botThinking}
            showBotThinkingDuringIncomeChoiceLock={
              showBotThinkingDuringIncomeChoiceLock
            }
            hideBotWaitMessageDuringTurnCycleLock={
              hideBotWaitMessageDuringTurnCycleLock
            }
            humanActionUiBlockedByAnimation={humanActionUiBlockedByAnimation}
            humanActionUiBlockedByTurnCycleAnimation={
              humanActionUiBlockedByTurnCycleAnimation
            }
            onAction={handleHumanAction}
            onResetTurn={handleTurnReset}
            onClosePicker={closeActionPicker}
            onOpenTradeCombinedPicker={openTradeCombinedPicker}
            onOpenTradePicker={openTradePicker}
            onOpenDistrictPicker={openDistrictPicker}
            onOpenDevelopOutrightCombinedPicker={
              openDevelopOutrightCombinedPicker
            }
            onOpenDevelopOutrightDistrictOnlyPicker={
              openDevelopOutrightDistrictOnlyPicker
            }
            onOpenDeedPaymentPicker={openDeedPaymentPicker}
          />

          <PlayerPanel
            title="You"
            player={humanPlayer}
            isActive={!terminal && humanView.activePlayerId === HUMAN_PLAYER}
            score={score}
            terminal={terminal}
            handSlotCount={PLAYER_HAND_SLOT_COUNT}
            botPlayerId={BOT_PLAYER}
            animateDeedProgress={animationsEnabled}
          />
        </aside>

        <section className="board-pane">
          <PlayerTokenRail
            player={botRailPlayer}
            side="bot"
            highlightedCrownSuits={incomeHighlightCrownSuitsByPlayer.get(
              BOT_PLAYER
            )}
          />
          <div className="district-strip" aria-label="District board">
            {humanView.districts.map((district) => (
              <DistrictColumn
                key={district.id}
                district={district}
                humanPlayerId={HUMAN_PLAYER}
                botPlayerId={BOT_PLAYER}
                animateDeedProgress={animationsEnabled}
                highlightedIncomeCardIds={incomeHighlightCardIdSet}
              />
            ))}
          </div>
          <PlayerTokenRail
            player={humanRailPlayer}
            side="human"
            highlightedCrownSuits={incomeHighlightCrownSuitsByPlayer.get(
              HUMAN_PLAYER
            )}
          />
        </section>

        <aside className="info-pane">
          <PlayerPanel
            title="Bot"
            player={botPlayer}
            isActive={!terminal && humanView.activePlayerId === BOT_PLAYER}
            score={score}
            terminal={terminal}
            handSlotCount={PLAYER_HAND_SLOT_COUNT}
            botPlayerId={BOT_PLAYER}
            animateDeedProgress={animationsEnabled}
          />

          <section className="panel">
            <h2>Roll Result</h2>
            <RollResult
              roll={humanView.lastIncomeRoll}
              taxSuit={humanView.lastTaxSuit}
            />
          </section>

          <DeckPiles
            drawCount={humanView.deck.drawCount}
            reshuffles={humanView.deck.reshuffles}
            discard={humanView.deck.discard}
            pendingDiscardHoldback={pendingDiscardHoldback}
            terminal={terminal}
          />

          <LogPanel timelineLog={timelineLog} humanPlayerId={HUMAN_PLAYER} />

          <OptionsMenu
            open={optionsMenuOpen}
            botProfileId={botProfileId}
            botStatusText={resolvedBotProfile.statusText}
            animationsEnabled={animationsEnabled}
            menuRef={optionsMenuRef}
            buttonRef={optionsMenuButtonRef}
            seedInputRef={seedInputRef}
            onToggle={() => setOptionsMenuOpen((open) => !open)}
            onReset={handleReset}
            onBotProfileChange={setBotProfileId}
            onAnimationsEnabledChange={setAnimationsEnabled}
          />
        </aside>
      </main>

      <OptionsBackdrop open={optionsMenuOpen} onClose={closeOptionsMenu} />

      <StartupPreloadOverlay
        ready={startupPreloadReady}
        error={startupPreloadError}
        progress={startupPreloadProgress}
        onRetry={retryStartupPreload}
      />

      <ResolutionWarningOverlay
        open={resolutionWarningOpen}
        onDismiss={() => setResolutionWarningOpen(false)}
      />

      <TurnCycleOverlay
        overlay={turnCycleOverlay}
        terminalWinner={terminalWinnerOverlayWinner}
        humanPlayerId={HUMAN_PLAYER}
      />

      <ResourceFlightLayer flights={resourceFlights} />

      <CardFlightLayer
        flights={cardFlights}
        animationsEnabled={animationsEnabled}
      />

      {actionPicker ? (
        <ActionPicker
          picker={actionPicker}
          pickerRef={actionPopoverRef}
          legalActions={humanActionsAcceptingInput}
          tradeSourceGroups={tradeSourceGroups}
          onPickerChange={setActionPicker}
          onSelectAction={handlePickerSelection}
          onClose={closeActionPicker}
        />
      ) : null}
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}
