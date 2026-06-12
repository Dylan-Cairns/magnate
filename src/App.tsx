import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { CardId } from './engine/cards';
import { PROPERTY_CARDS } from './engine/cards';
import { districtWinnersByPlayer, scoreLive } from './engine/scoring';
import type { GameAction, PlayerId, Suit } from './engine/types';
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
import { errorMessage } from './ui/gameControllerModel';
import {
  getBugReportIssueUrl,
  buildBugReport,
  downloadBugReport,
} from './ui/bugReport';
import {
  preloadStartupAssets,
  type StartupPreloadProgress,
} from './ui/startupPreload';
import { ActionPicker } from './ui/components/ActionPicker';
import { ActionsPanel } from './ui/components/ActionsPanel';
import { CardFlightLayer } from './ui/components/CardFlightLayer';
import { BotHandPanel } from './ui/components/BotHandPanel';
import { DeckPiles } from './ui/components/DeckPiles';
import {
  ResolutionWarningOverlay,
  StartupPreloadOverlay,
} from './ui/components/GameOverlays';
import { DecktetSuitDiagram } from './ui/components/DecktetSuitDiagram';
import { LogPanel } from './ui/components/LogPanel';
import { OptionsBackdrop, OptionsMenu } from './ui/components/OptionsMenu';
import { ResourceFlightLayer } from './ui/components/ResourceFlightLayer';
import { DistrictColumn, PlayerTokenRail } from './ui/components/DistrictBoard';
import { PlayerPanel } from './ui/components/PlayerPanel';
import { RollResult } from './ui/components/RollResult';
import { useDismissableLayer } from './ui/hooks/useDismissableLayer';
import { useGameController } from './ui/hooks/useGameController';

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
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

const LOG_VISIBLE_KEY = 'magnate:logVisible';
const MAP_VISIBLE_KEY = 'magnate:mapVisible';
const DECK_MAP_INTERACTIVE_KEY = 'magnate:deckMapInteractive';

const ALL_SUITS: Suit[] = ['Moons', 'Suns', 'Waves', 'Leaves', 'Wyrms', 'Knots'];
const SUIT_CARD_IDS = new Map<Suit, ReadonlyArray<CardId>>(
  ALL_SUITS.map((suit) => [
    suit,
    PROPERTY_CARDS.filter((c) => c.suits.includes(suit)).map((c) => c.id),
  ])
);

function readBooleanPreference(key: string, defaultValue: boolean): boolean {
  if (typeof window === 'undefined') return defaultValue;
  try {
    const stored = window.localStorage.getItem(key);
    if (stored === null) return defaultValue;
    return stored !== 'false';
  } catch {
    return defaultValue;
  }
}

function persistBooleanPreference(key: string, value: boolean): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, value ? 'true' : 'false');
  } catch {
    // Ignore storage failures (e.g. private browsing restrictions).
  }
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

export function App() {
  const [actionPicker, setActionPicker] = useState<ActionPickerState | null>(
    null
  );
  const [optionsMenuOpen, setOptionsMenuOpen] = useState<boolean>(false);
  const [bugReportOpen, setBugReportOpen] = useState<boolean>(false);
  const [newGameExpanded, setNewGameExpanded] = useState<boolean>(false);
  const [logVisible, setLogVisible] = useState<boolean>(() =>
    readBooleanPreference(LOG_VISIBLE_KEY, true)
  );
  const [mapVisible, setMapVisible] = useState<boolean>(() =>
    readBooleanPreference(MAP_VISIBLE_KEY, true)
  );
  const [deckMapInteractive, setDeckMapInteractive] = useState<boolean>(() =>
    readBooleanPreference(DECK_MAP_INTERACTIVE_KEY, true)
  );
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
  const {
    state,
    humanView,
    pendingNextDistricts,
    timelineLog,
    actionHistory,
    error,
    terminal,
    activePlayerId,
    botThinking,
    botProfileId,
    botStatusText,
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
      incomeResourcePreviewByPlayer,
      pendingDiscardHoldback,
      pendingDrawCardIds,
      activePlayerHighlightOverride,
      actionCommitPending,
      allowHumanActionsWhileCommitPending,
    },
  } = useGameController({
    humanPlayerId: HUMAN_PLAYER,
    botPlayerId: BOT_PLAYER,
    startupPreloadReady,
  });
  const actionPopoverRef = useRef<HTMLElement | null>(null);
  const optionsMenuRef = useRef<HTMLElement | null>(null);
  const optionsMenuButtonRef = useRef<HTMLButtonElement | null>(null);
  const newGamePanelRef = useRef<HTMLElement | null>(null);
  const newGameButtonRef = useRef<HTMLButtonElement | null>(null);
  const seedInputRef = useRef<HTMLInputElement | null>(null);
  const closeActionPicker = useCallback(() => setActionPicker(null), []);
  const closeOptionsMenu = useCallback(() => setOptionsMenuOpen(false), []);
  const closeBugReport = useCallback(() => setBugReportOpen(false), []);
  const closeNewGame = useCallback(() => setNewGameExpanded(false), []);
  const turnCyclePreludeActive = activePlayerHighlightOverride !== null;
  const visualActivePlayerId = activePlayerHighlightOverride ?? activePlayerId;
  const retryStartupPreload = useCallback(() => {
    setStartupPreloadReady(false);
    setStartupPreloadError(null);
    setStartupPreloadProgress(STARTUP_PRELOAD_INITIAL_PROGRESS);
    setStartupPreloadAttempt((current) => current + 1);
  }, []);
  const actionPopoverLayerRefs = useMemo(() => [actionPopoverRef], []);
  const optionsMenuLayerRefs = useMemo(
    () => [optionsMenuRef, optionsMenuButtonRef],
    []
  );
  const newGameLayerRefs = useMemo(
    () => [newGamePanelRef, newGameButtonRef],
    []
  );
  useEffect(() => {
    let cancelled = false;

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

  useEffect(() => {
    persistBooleanPreference(LOG_VISIBLE_KEY, logVisible);
  }, [logVisible]);

  useEffect(() => {
    persistBooleanPreference(MAP_VISIBLE_KEY, mapVisible);
  }, [mapVisible]);

  useEffect(() => {
    persistBooleanPreference(DECK_MAP_INTERACTIVE_KEY, deckMapInteractive);
  }, [deckMapInteractive]);

  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
  const score = useMemo(() => state.finalScore ?? scoreLive(state), [state]);
  const wonDistrictsByPlayer = useMemo(
    () => districtWinnersByPlayer(state),
    [state]
  );
  const { dimmedCardIds, dimmedSuits } = useMemo(() => {
    if (!deckMapInteractive) {
      return {
        dimmedCardIds: new Set<CardId>(),
        dimmedSuits: new Set<Suit>(),
      };
    }
    const inCirculation = new Set<CardId>([
      ...state.deck.draw,
      ...state.players.flatMap((p) => p.hand),
      ...(state.deck.reshuffles === 0 ? state.deck.discard : []),
    ]);
    // Cards about to leave circulation once the current animation commits
    const pendingCards = new Set<CardId>();
    if (pendingNextDistricts) {
      for (const district of pendingNextDistricts) {
        for (const stack of Object.values(district.stacks)) {
          for (const cardId of stack.developed) pendingCards.add(cardId);
          if (stack.deed) pendingCards.add(stack.deed.cardId);
        }
      }
    }
    // Edge-label dimming: 2-suit numeral property cards not in (effective) circulation
    const dimmedCardIds = new Set<CardId>();
    for (const card of PROPERTY_CARDS) {
      if (card.suits.length === 2 && !inCirculation.has(card.id)) {
        dimmedCardIds.add(card.id);
      }
    }
    for (const cardId of pendingCards) {
      dimmedCardIds.add(cardId);
    }
    // Suit-node dimming: a suit dims when every one of its property cards is gone
    const dimmedSuits = new Set<Suit>();
    for (const [suit, cardIds] of SUIT_CARD_IDS) {
      if (cardIds.every((id) => !inCirculation.has(id) || pendingCards.has(id))) {
        dimmedSuits.add(suit);
      }
    }
    return { dimmedCardIds, dimmedSuits };
  }, [deckMapInteractive, state, pendingNextDistricts]);
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
    () =>
      new Set([
        ...incomeHighlightCardIds,
        ...(actionCommitPending ? [] : pendingIncomeChoiceCardIds),
      ]),
    [actionCommitPending, incomeHighlightCardIds, pendingIncomeChoiceCardIds]
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

  const humanActionItems = useMemo(
    () => buildHumanActionList(humanActionsAcceptingInput),
    [humanActionsAcceptingInput]
  );
  const tradeSourceGroups = useMemo(
    () => buildTradeSourceGroups(humanActionsAcceptingInput),
    [humanActionsAcceptingInput]
  );
  const hasMultipleTradeSources = tradeSourceGroups.length > 1;
  const isIncomeChoicePhase =
    state.phase === 'CollectIncome' &&
    (state.pendingIncomeChoices?.length ?? 0) > 0;
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

  const isTurnCycleAnimationLock =
    actionCommitPending && allowHumanActionsWhileCommitPending;
  const showBotThinkingDuringIncomeChoiceLock =
    isTurnCycleAnimationLock &&
    activePlayerId === BOT_PLAYER &&
    isIncomeChoicePhase;
  const hideBotWaitMessageDuringTurnCycleLock =
    isTurnCycleAnimationLock && !showBotThinkingDuringIncomeChoiceLock;
  const humanActionUiBlockedByAnimation =
    activePlayerId === HUMAN_PLAYER &&
    actionCommitPending &&
    humanActionsAcceptingInput.length === 0;
  const humanActionUiBlockedByTurnCycleAnimation =
    humanActionUiBlockedByAnimation && isTurnCycleAnimationLock;

  if (
    terminal ||
    (activePlayerId !== HUMAN_PLAYER && !isIncomeChoicePhase) ||
    humanActionUiBlockedByAnimation
  ) {
    if (actionPicker) closeActionPicker();
  } else if (actionPicker) {
    const isIllegal =
      actionPicker.kind === 'trade-combined'
        ? !tradeCompositePickerStillLegal(actionPicker, humanActionsAcceptingInput)
        : actionPicker.kind === 'develop-outright-combined'
          ? !developOutrightCompositePickerStillLegal(actionPicker, humanActionsAcceptingInput)
          : !pickerStillLegal(toPickerQuery(actionPicker), humanActionsAcceptingInput);
    if (isIllegal) closeActionPicker();
  }

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

  useDismissableLayer({
    enabled: newGameExpanded,
    onDismiss: closeNewGame,
    insideRefs: newGameLayerRefs,
  });

  const handleReset = () => {
    const specifiedSeed = seedInputRef.current?.value.trim() ?? '';
    if (seedInputRef.current) {
      seedInputRef.current.value = '';
    }
    closeActionPicker();
    closeOptionsMenu();
    closeNewGame();
    resetSession(specifiedSeed || undefined);
  };

  const handleNewGameToggle = () => {
    if (newGameExpanded) {
      handleReset();
    } else {
      closeOptionsMenu();
      closeBugReport();
      setNewGameExpanded(true);
    }
  };

  const handleOpenBugReport = () => {
    closeActionPicker();
    closeOptionsMenu();
    setBugReportOpen((open) => !open);
  };

  const handleDownloadBugReport = () => {
    downloadBugReport(
      buildBugReport({
        state,
        timelineLog,
        actionHistory,
        humanPlayerId: HUMAN_PLAYER,
        botPlayerId: BOT_PLAYER,
        botProfileId,
        animationsEnabled,
        error,
      })
    );
  };

  const handlePickerSelection = (action: GameAction) => {
    closeActionPicker();
    performHumanAction(action);
  };

  const handleTurnReset = () => {
    if (!canResetTurn) {
      return;
    }

    closeActionPicker();
    resetTurn();
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

  const openIncomeChoicePicker = (
    config: { playerId: PlayerId; cardId: CardId; districtId: string },
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'income-choice',
      playerId: config.playerId,
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

  if (!startupPreloadReady) {
    return (
      <div className="app-shell">
        <StartupPreloadOverlay
          ready={false}
          error={startupPreloadError}
          progress={startupPreloadProgress}
          onRetry={retryStartupPreload}
        />
        <ResolutionWarningOverlay
          open={resolutionWarningOpen}
          onDismiss={() => setResolutionWarningOpen(false)}
        />
      </div>
    );
  }

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
  const humanPanelPlayer =
    pendingDrawCardIds.length > 0
      ? {
          ...humanPlayer,
          hand: humanPlayer.hand.filter(
            (id) => !pendingDrawCardIds.includes(id)
          ),
        }
      : humanPlayer;
  const botPendingDrawCount = pendingDrawCardIds.filter(
    (id) => !humanPlayer.hand.includes(id)
  ).length;
  const botPanelPlayer =
    botPendingDrawCount > 0
      ? { ...botPlayer, handCount: botPlayer.handCount - botPendingDrawCount }
      : botPlayer;

  return (
    <div className="app-shell">
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
            isIncomeChoicePhase={isIncomeChoicePhase}
            hasMultipleTradeSources={hasMultipleTradeSources}
            actionPicker={actionPicker}
            canResetTurn={canResetTurn}
            botThinking={botThinking}
            hideBotWaitMessageDuringTurnCycleLock={
              hideBotWaitMessageDuringTurnCycleLock
            }
            humanActionUiBlockedByAnimation={humanActionUiBlockedByAnimation}
            humanActionUiBlockedByTurnCycleAnimation={
              humanActionUiBlockedByTurnCycleAnimation
            }
            onAction={performHumanAction}
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
            onOpenIncomeChoicePicker={openIncomeChoicePicker}
          />

          <PlayerPanel
            player={humanPanelPlayer}
            isActive={!terminal && visualActivePlayerId === HUMAN_PLAYER}
            score={score}
            terminal={terminal}
            handSlotCount={PLAYER_HAND_SLOT_COUNT}
            humanPlayerId={HUMAN_PLAYER}
            botPlayerId={BOT_PLAYER}
            animateDeedProgress={animationsEnabled}
          />
        </aside>

        <section className="board-pane">
          <div className="board-top-row">
            <div className="dice-float">
              <RollResult
                roll={
                  turnCyclePreludeActive ? undefined : humanView.lastIncomeRoll
                }
                taxSuit={
                  turnCyclePreludeActive ? undefined : humanView.lastTaxSuit
                }
                gameKey={state.seed}
                holdPrevious={turnCyclePreludeActive}
                animationsEnabled={animationsEnabled}
              />
            </div>
            <PlayerTokenRail
              player={botRailPlayer}
              side="bot"
              highlightedCrownSuits={incomeHighlightCrownSuitsByPlayer.get(
                BOT_PLAYER
              )}
            />
          </div>
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
          <div className="bot-info-row">
            <BotHandPanel
              player={botPanelPlayer}
              isActive={!terminal && visualActivePlayerId === BOT_PLAYER}
              score={score}
              terminal={terminal}
              humanPlayerId={HUMAN_PLAYER}
              botPlayerId={BOT_PLAYER}
            />

            <DeckPiles
              drawCount={humanView.deck.drawCount}
              reshuffles={humanView.deck.reshuffles}
              discard={humanView.deck.discard}
              pendingDiscardHoldback={pendingDiscardHoldback}
              terminal={terminal}
            />
          </div>

          <div className="log-map-stack">
            {logVisible && <LogPanel timelineLog={timelineLog} humanPlayerId={HUMAN_PLAYER} />}
            {mapVisible && <DecktetSuitDiagram dimmedCardIds={dimmedCardIds} dimmedSuits={dimmedSuits} />}
          </div>

          <OptionsMenu
            open={optionsMenuOpen}
            botProfileId={botProfileId}
            botStatusText={botStatusText}
            animationsEnabled={animationsEnabled}
            menuRef={optionsMenuRef}
            buttonRef={optionsMenuButtonRef}
            seedInputRef={seedInputRef}
            newGameExpanded={newGameExpanded}
            newGamePanelRef={newGamePanelRef}
            newGameButtonRef={newGameButtonRef}
            onBugReport={handleOpenBugReport}
            onToggle={() => {
                setBugReportOpen(false);
                closeNewGame();
                setOptionsMenuOpen((open) => !open);
              }}
            onNewGameToggle={handleNewGameToggle}
            onBotProfileChange={setBotProfileId}
            onAnimationsEnabledChange={setAnimationsEnabled}
            bugReportOpen={bugReportOpen}
            bugReportIssueUrl={getBugReportIssueUrl()}
            onBugReportDownload={handleDownloadBugReport}
            logVisible={logVisible}
            onToggleLog={() => setLogVisible((v) => !v)}
            mapVisible={mapVisible}
            onToggleMap={() => setMapVisible((v) => !v)}
            deckMapInteractive={deckMapInteractive}
            onDeckMapInteractiveChange={setDeckMapInteractive}
          />
        </aside>
      </main>

      <OptionsBackdrop open={optionsMenuOpen} onClose={closeOptionsMenu} />
      <OptionsBackdrop open={bugReportOpen} onClose={closeBugReport} />
      <OptionsBackdrop open={newGameExpanded} onClose={closeNewGame} />

      <ResolutionWarningOverlay
        open={resolutionWarningOpen}
        onDismiss={() => setResolutionWarningOpen(false)}
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
