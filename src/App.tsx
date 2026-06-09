import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { CardId } from './engine/cards';
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
import { BugReportModal } from './ui/components/BugReportModal';
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
      turnCycleOverlay,
      incomeHighlightCardIds,
      incomeHighlightCrowns,
      incomeResourcePreviewByPlayer,
      pendingDiscardHoldback,
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
  const seedInputRef = useRef<HTMLInputElement | null>(null);
  const closeActionPicker = useCallback(() => setActionPicker(null), []);
  const closeOptionsMenu = useCallback(() => setOptionsMenuOpen(false), []);
  const closeBugReport = useCallback(() => setBugReportOpen(false), []);
  const retryStartupPreload = useCallback(
    () => setStartupPreloadAttempt((current) => current + 1),
    []
  );
  const actionPopoverLayerRefs = useMemo(() => [actionPopoverRef], []);
  const optionsMenuLayerRefs = useMemo(
    () => [optionsMenuRef, optionsMenuButtonRef],
    []
  );
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

  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
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

  useEffect(() => {
    if (terminal || (activePlayerId !== HUMAN_PLAYER && !isIncomeChoicePhase)) {
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
    isIncomeChoicePhase,
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

  const handleReset = () => {
    const specifiedSeed = seedInputRef.current?.value.trim() ?? '';
    if (seedInputRef.current) {
      seedInputRef.current.value = '';
    }
    closeActionPicker();
    closeOptionsMenu();
    resetSession(specifiedSeed || undefined);
  };

  const handleOpenBugReport = () => {
    closeActionPicker();
    closeOptionsMenu();
    setBugReportOpen(true);
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
            botStatusText={botStatusText}
            animationsEnabled={animationsEnabled}
            menuRef={optionsMenuRef}
            buttonRef={optionsMenuButtonRef}
            seedInputRef={seedInputRef}
            onBugReport={handleOpenBugReport}
            onToggle={() => setOptionsMenuOpen((open) => !open)}
            onReset={handleReset}
            onBotProfileChange={setBotProfileId}
            onAnimationsEnabledChange={setAnimationsEnabled}
          />
        </aside>
      </main>

      <OptionsBackdrop open={optionsMenuOpen} onClose={closeOptionsMenu} />

      <BugReportModal
        open={bugReportOpen}
        issueUrl={getBugReportIssueUrl()}
        onDownload={handleDownloadBugReport}
        onClose={closeBugReport}
      />

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
