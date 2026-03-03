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
  ResourcePool,
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
import {
  applySingleTaxLossToPreview,
  buildResourcePreviewByPlayer,
  CARD_DRAW_FLIGHT_DELAY_MS,
  cardFlightSettleMs,
  RESOURCE_FLIGHT_STAGGER_MS,
  resourceFlightSettleMs,
  shouldAllowHumanActionsDuringAnimationSettle,
  shouldCommitBeforeAnimationSettle,
  TERMINAL_CLEANUP_FLIGHT_DURATION_MS,
  TERMINAL_CLEANUP_VERTICAL_TRAVEL_PX,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from './ui/animations/timing';
import {
  buildTurnCycleVisualPlan,
  type TurnCycleVisualPlan,
} from './ui/animations/turnCycleVisualPlan';
import type {
  CardFlight,
  PendingResourceFlight,
  ResourceFlight,
  TurnCycleOverlayState,
} from './ui/animations/types';
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
import type { CardPerspective } from './ui/components/CardTile';
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
import {
  clearAllDeedTokenLayouts,
  layoutDeedTokensBySide,
  resetDeedTokenLayout,
  type DeedTokenSide,
} from './ui/components/deedTokenLayout';
import { DistrictColumn, PlayerTokenRail } from './ui/components/DistrictBoard';
import { PlayerPanel } from './ui/components/PlayerPanel';
import { RollResult } from './ui/components/RollResult';
import { tokenEntries } from './ui/components/TokenComponents';
import { useDismissableLayer } from './ui/hooks/useDismissableLayer';
import { transitionLogEntries } from './ui/logTimeline';
import {
  deriveTurnCycleEvents,
  type TurnCycleIncomeToken,
} from './ui/turnCycleEvents';

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const DEFAULT_BOT_DELAY_MS = 450;
const PLAYER_HAND_SLOT_COUNT = 3;
const TRADE_POPOVER_WIDTH_PX = 220;
const TRADE_POPOVER_MIN_HEIGHT_PX = 188;
const TRADE_POPOVER_GAP_PX = 8;
const VIEWPORT_PADDING_PX = 10;
const DEFAULT_TOKEN_CHIP_SIZE_PX = 22;
const DEFAULT_TOKEN_RAIL_GAP_PX = 2.56;
const RESOLUTION_WARNING_BASE_WIDTH_PX = 1920;
const RESOLUTION_WARNING_BASE_HEIGHT_PX = 1080;
const RESOLUTION_WARNING_THRESHOLD_SCALE = 0.9;
const ANIMATIONS_STORAGE_KEY = 'magnate:animationsEnabled';
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

function cssEscapeValue(value: string): string {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value);
  }
  return value.replace(/["\\]/g, '\\$&');
}

function elementCenter(element: Element): { x: number; y: number } {
  const rect = element.getBoundingClientRect();
  return {
    x: rect.left + rect.width / 2,
    y: rect.top + rect.height / 2,
  };
}

function tokenVisualCenter(element: HTMLElement): { x: number; y: number } {
  return elementCenter(element);
}

function parsePixelValue(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function deedTokenCenterInRail(
  railElement: HTMLElement,
  tokenSizePx: number,
  gapPx: number,
  index: number,
  sideCount: number
): { x: number; y: number } {
  const railCenter = elementCenter(railElement);
  if (sideCount <= 1) {
    return railCenter;
  }

  const totalHeight = sideCount * tokenSizePx + (sideCount - 1) * gapPx;
  const firstTokenCenterOffset = -totalHeight / 2 + tokenSizePx / 2;
  return {
    x: railCenter.x,
    y: railCenter.y + firstTokenCenterOffset + index * (tokenSizePx + gapPx),
  };
}

function tokenElementInRail(
  playerId: PlayerId,
  rowClassName: 'rail-resources-row' | 'crowns-rail-row',
  suit: Suit
): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  const escapedSuit = cssEscapeValue(suit);
  const selector =
    `[data-token-rail-player-id="${escapedPlayerId}"] ` +
    `.${rowClassName} .token-chip[data-token-suit="${escapedSuit}"]`;
  const searchRoot =
    document.querySelector<HTMLElement>('.board-pane') ?? document;
  const matches = [...searchRoot.querySelectorAll<HTMLElement>(selector)];
  if (matches.length === 0) {
    return null;
  }
  const visibleMatch = matches.find((element) => {
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  });
  return visibleMatch ?? matches[0];
}

function resourceTokenElementForPlayer(
  playerId: PlayerId,
  suit: Suit
): HTMLElement | null {
  return tokenElementInRail(playerId, 'rail-resources-row', suit);
}

function crownTokenElementForPlayer(
  playerId: PlayerId,
  suit: Suit
): HTMLElement | null {
  return tokenElementInRail(playerId, 'crowns-rail-row', suit);
}

function districtCardElementForPlayer(
  playerId: PlayerId,
  districtId: string,
  cardId: CardId
): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  const escapedDistrictId = cssEscapeValue(districtId);
  const escapedCardId = cssEscapeValue(cardId);
  return document.querySelector<HTMLElement>(
    `.district-column[data-district-id="${escapedDistrictId}"] .district-lane[data-lane-player-id="${escapedPlayerId}"] .card-tile[data-card-id="${escapedCardId}"]`
  );
}

function deedTokenElementInCard(
  cardElement: HTMLElement,
  suit: Suit
): HTMLElement | null {
  const escapedSuit = cssEscapeValue(suit);
  return cardElement.querySelector<HTMLElement>(
    `.card-side-token-rail .token-chip[data-token-suit="${escapedSuit}"]`
  );
}

function terminalCleanupTargetY(startY: number): number {
  const viewportCenterY = window.innerHeight / 2;
  const direction = startY < viewportCenterY ? -1 : 1;
  return startY + direction * TERMINAL_CLEANUP_VERTICAL_TRAVEL_PX;
}

function collectTerminalCleanupFlights(
  previousState: GameState,
  nextState: GameState,
  makeResourceFlightId: () => string,
  makeCardFlightId: () => string
): {
  resourceFlights: ReadonlyArray<ResourceFlight>;
  cardFlights: ReadonlyArray<CardFlight>;
} | null {
  if (typeof document === 'undefined') {
    return null;
  }
  if (isTerminal(previousState) || !isTerminal(nextState)) {
    return null;
  }

  const resourceFlights: ResourceFlight[] = [];
  const cardFlights: CardFlight[] = [];

  for (const district of previousState.districts) {
    for (const player of previousState.players) {
      const deed = district.stacks[player.id].deed;
      if (!deed) {
        continue;
      }

      const cardElement =
        districtCardElementForPlayer(player.id, district.id, deed.cardId) ??
        document.querySelector<HTMLElement>(
          `.district-strip .card-tile[data-card-id="${cssEscapeValue(deed.cardId)}"][data-in-development="true"]`
        );
      if (!cardElement) {
        continue;
      }

      const cardRect = cardElement.getBoundingClientRect();
      const cardCenter = elementCenter(cardElement);
      cardFlights.push({
        id: makeCardFlightId(),
        variant: 'terminal-clear',
        visual: 'face',
        cardId: deed.cardId,
        isDeed: true,
        perspective: player.id === BOT_PLAYER ? 'bot' : 'human',
        startX: cardCenter.x,
        startY: cardCenter.y,
        endX: cardCenter.x,
        endY: terminalCleanupTargetY(cardCenter.y),
        startWidth: cardRect.width,
        startHeight: cardRect.height,
        endWidth: cardRect.width,
        endHeight: cardRect.height,
        delayMs: 0,
        durationMs: TERMINAL_CLEANUP_FLIGHT_DURATION_MS,
      });

      for (const entry of tokenEntries(deed.tokens)) {
        const tokenElement = deedTokenElementInCard(cardElement, entry.suit);
        if (!tokenElement) {
          continue;
        }
        const tokenCenter = tokenVisualCenter(tokenElement);
        for (let index = 0; index < entry.count; index += 1) {
          resourceFlights.push({
            id: makeResourceFlightId(),
            suit: entry.suit,
            startX: tokenCenter.x,
            startY: tokenCenter.y,
            endX: tokenCenter.x,
            endY: terminalCleanupTargetY(tokenCenter.y),
            delayMs: 0,
            durationMs: TERMINAL_CLEANUP_FLIGHT_DURATION_MS,
            variant: 'terminal-clear',
          });
        }
      }
    }
  }

  if (resourceFlights.length === 0 && cardFlights.length === 0) {
    return null;
  }

  return {
    resourceFlights,
    cardFlights,
  };
}

function collectTurnCycleAnimationPlan(
  previousState: GameState,
  nextState: GameState,
  action: GameAction,
  baseDelayMs: number
): {
  visualPlan: TurnCycleVisualPlan;
  totalDurationMs: number;
  incomeAnimationEndMs: number;
} | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const cycle = deriveTurnCycleEvents(previousState, nextState, action);
  if (!cycle) {
    return null;
  }

  return buildTurnCycleVisualPlan(cycle, baseDelayMs);
}

function buildTaxLossFlightsFromDom(
  targets: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>,
  makeFlightId: () => string
): ResourceFlight[] {
  if (typeof document === 'undefined' || targets.length === 0) {
    return [];
  }

  const viewportCenterY = window.innerHeight / 2;
  const flights: ResourceFlight[] = [];
  for (const [index, target] of targets.entries()) {
    const sourceElement = resourceTokenElementForPlayer(
      target.playerId,
      target.suit
    );
    if (!sourceElement) {
      continue;
    }
    const source = tokenVisualCenter(sourceElement);
    flights.push({
      id: makeFlightId(),
      suit: target.suit,
      startX: source.x,
      startY: source.y,
      endX: source.x,
      endY: viewportCenterY,
      delayMs: index * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
      durationMs: TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
      variant: 'tax-loss',
    });
  }

  return flights;
}

function buildIncomeFlightsFromDom(
  tokens: ReadonlyArray<TurnCycleIncomeToken>,
  makeFlightId: () => string
): ResourceFlight[] {
  if (typeof document === 'undefined' || tokens.length === 0) {
    return [];
  }

  const flights: ResourceFlight[] = [];
  for (const [index, token] of tokens.entries()) {
    const sourceElement =
      token.source.kind === 'district-card'
        ? districtCardElementForPlayer(
            token.playerId,
            token.source.districtId,
            token.source.cardId
          )
        : crownTokenElementForPlayer(token.playerId, token.suit);
    const targetElement = resourceTokenElementForPlayer(
      token.playerId,
      token.suit
    );
    if (!sourceElement || !targetElement) {
      continue;
    }

    const source =
      token.source.kind === 'district-card'
        ? elementCenter(sourceElement)
        : tokenVisualCenter(sourceElement);
    const target = tokenVisualCenter(targetElement);
    flights.push({
      id: makeFlightId(),
      suit: token.suit,
      startX: source.x,
      startY: source.y,
      endX: target.x,
      endY: target.y,
      delayMs: index * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
      durationMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
      variant: 'transfer',
    });
  }

  return flights;
}

function collectDeedResourceFlights(
  state: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  makeFlightId: () => string
): PendingResourceFlight[] {
  if (action.type !== 'develop-deed') {
    return [];
  }
  if (typeof document === 'undefined') {
    return [];
  }

  const suitsToAnimate: Suit[] = [];
  for (const entry of tokenEntries(action.tokens)) {
    for (let count = 0; count < entry.count; count += 1) {
      suitsToAnimate.push(entry.suit);
    }
  }
  if (suitsToAnimate.length === 0) {
    return [];
  }

  const escapedCardId = cssEscapeValue(action.cardId);
  const cardSelector = `.district-strip .card-tile[data-card-id="${escapedCardId}"]`;
  const cardElement =
    document.querySelector<HTMLElement>(
      `${cardSelector}[data-in-development="true"]`
    ) ?? document.querySelector<HTMLElement>(cardSelector);
  if (!cardElement) {
    return [];
  }

  const district = state.districts.find(
    (candidate) => candidate.id === action.districtId
  );
  const deedBefore = district?.stacks[actingPlayerId]?.deed;
  if (!deedBefore || deedBefore.cardId !== action.cardId) {
    return [];
  }

  const nextDeedTokens: Partial<Record<Suit, number>> = {
    ...deedBefore.tokens,
  };
  for (const entry of tokenEntries(action.tokens)) {
    nextDeedTokens[entry.suit] =
      (nextDeedTokens[entry.suit] ?? 0) + entry.count;
  }

  const perspective: 'human' | 'bot' = cardElement.classList.contains(
    'perspective-bot'
  )
    ? 'bot'
    : 'human';
  const deedTokenEntries = tokenEntries(deedBefore.tokens);
  if (deedTokenEntries.length === 0) {
    resetDeedTokenLayout(action.cardId, perspective);
  }
  const nextTokenEntries = tokenEntries(nextDeedTokens);
  const nextBySide = layoutDeedTokensBySide(
    action.cardId,
    perspective,
    nextTokenEntries
  );
  const targetBySuit = new Map<
    Suit,
    { side: DeedTokenSide; index: number; sideCount: number }
  >();
  for (const [index, entry] of nextBySide.left.entries()) {
    targetBySuit.set(entry.suit, {
      side: 'left',
      index,
      sideCount: nextBySide.left.length,
    });
  }
  for (const [index, entry] of nextBySide.right.entries()) {
    targetBySuit.set(entry.suit, {
      side: 'right',
      index,
      sideCount: nextBySide.right.length,
    });
  }

  const escapedPlayerId = cssEscapeValue(actingPlayerId);
  const sourceBySuit = new Map<Suit, { x: number; y: number }>();
  const tokenSizeBySuit = new Map<Suit, number>();
  for (const suit of new Set(suitsToAnimate)) {
    const escapedSuit = cssEscapeValue(suit);
    const sourceElement = document.querySelector<HTMLElement>(
      `[data-token-rail-player-id="${escapedPlayerId}"] .rail-resources-row .token-chip[data-token-suit="${escapedSuit}"]`
    );
    if (!sourceElement) {
      continue;
    }
    sourceBySuit.set(suit, tokenVisualCenter(sourceElement));
    tokenSizeBySuit.set(
      suit,
      sourceElement.getBoundingClientRect().width || DEFAULT_TOKEN_CHIP_SIZE_PX
    );
  }

  const flights: PendingResourceFlight[] = [];
  for (const [index, suit] of suitsToAnimate.entries()) {
    const source = sourceBySuit.get(suit);
    const target = targetBySuit.get(suit);
    if (!source) {
      continue;
    }
    if (!target) {
      continue;
    }

    const escapedSuit = cssEscapeValue(suit);
    const existingTargetChip = cardElement.querySelector<HTMLElement>(
      `.card-side-token-rail-${target.side} .token-chip[data-token-suit="${escapedSuit}"]`
    );

    let targetPoint: { x: number; y: number } | null = null;
    if (existingTargetChip) {
      targetPoint = elementCenter(existingTargetChip);
    } else {
      const targetRail = cardElement.querySelector<HTMLElement>(
        `.card-side-token-rail-${target.side}`
      );
      if (targetRail) {
        const railStyle = window.getComputedStyle(targetRail);
        const gapPx = parsePixelValue(
          railStyle.rowGap || railStyle.gap,
          DEFAULT_TOKEN_RAIL_GAP_PX
        );
        targetPoint = deedTokenCenterInRail(
          targetRail,
          tokenSizeBySuit.get(suit) ?? DEFAULT_TOKEN_CHIP_SIZE_PX,
          gapPx,
          target.index,
          target.sideCount
        );
      }
    }
    if (!targetPoint) {
      continue;
    }

    flights.push({
      id: makeFlightId(),
      suit,
      startX: source.x,
      startY: source.y,
      endX: targetPoint.x,
      endY: targetPoint.y,
      delayMs: index * RESOURCE_FLIGHT_STAGGER_MS,
      variant: 'transfer',
    });
  }

  return flights;
}

function collectIncomeChoiceResourceFlights(
  action: GameAction,
  makeFlightId: () => string
): PendingResourceFlight[] {
  if (action.type !== 'choose-income-suit') {
    return [];
  }
  if (typeof document === 'undefined') {
    return [];
  }

  const sourceElement = districtCardElementForPlayer(
    action.playerId,
    action.districtId,
    action.cardId
  );
  const targetElement = resourceTokenElementForPlayer(
    action.playerId,
    action.suit
  );
  if (!sourceElement || !targetElement) {
    return [];
  }

  const source = elementCenter(sourceElement);
  const target = tokenVisualCenter(targetElement);
  return [
    {
      id: makeFlightId(),
      suit: action.suit,
      startX: source.x,
      startY: source.y,
      endX: target.x,
      endY: target.y,
      delayMs: 0,
      durationMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
      variant: 'transfer',
    },
  ];
}

function isLaneCardPlayAction(
  action: GameAction
): action is Extract<GameAction, { type: 'buy-deed' | 'develop-outright' }> {
  return action.type === 'buy-deed' || action.type === 'develop-outright';
}

function stackStepForLane(
  laneElement: HTMLElement,
  isBotLane: boolean,
  cardHeightPx: number
): number {
  const stackStepValue = window
    .getComputedStyle(laneElement)
    .getPropertyValue('--card-stack-step')
    .trim();
  if (stackStepValue.length > 0) {
    const probe = document.createElement('div');
    probe.style.position = 'absolute';
    probe.style.visibility = 'hidden';
    probe.style.pointerEvents = 'none';
    probe.style.width = '0';
    probe.style.height = stackStepValue;
    probe.style.padding = '0';
    probe.style.margin = '0';
    probe.style.border = '0';
    laneElement.appendChild(probe);
    const resolvedStepPx = probe.getBoundingClientRect().height;
    probe.remove();
    if (resolvedStepPx > 0) {
      return resolvedStepPx;
    }
  }

  const topStackCard = laneElement.querySelector<HTMLElement>(
    '.lane-stack-card:last-child'
  );
  if (!topStackCard) {
    return cardHeightPx * 0.24;
  }

  const stackPosition = Number.parseFloat(
    topStackCard.style.getPropertyValue('--stack-position')
  );
  if (!Number.isFinite(stackPosition) || stackPosition <= 0) {
    return cardHeightPx * 0.24;
  }

  const computed = window.getComputedStyle(topStackCard);
  const offsetPx = parsePixelValue(
    isBotLane ? computed.bottom : computed.top,
    0
  );
  if (offsetPx > 0) {
    return offsetPx / stackPosition;
  }
  return cardHeightPx * 0.24;
}

function laneTargetCenter(
  laneElement: HTMLElement,
  cardHeightPx: number
): { x: number; y: number } | null {
  const isBotLane = laneElement.classList.contains('is-bot');
  const topCardTile = laneElement.querySelector<HTMLElement>(
    '.lane-stack-card:last-child .card-tile'
  );
  if (topCardTile) {
    const topCenter = elementCenter(topCardTile);
    const stepPx = stackStepForLane(laneElement, isBotLane, cardHeightPx);
    return {
      x: topCenter.x,
      y: topCenter.y + (isBotLane ? -stepPx : stepPx),
    };
  }

  const laneFrame = laneElement.querySelector<HTMLElement>('.lane-stack-frame');
  if (!laneFrame) {
    return null;
  }
  const frameRect = laneFrame.getBoundingClientRect();
  return {
    x: frameRect.left + frameRect.width / 2,
    y: isBotLane
      ? frameRect.bottom - cardHeightPx / 2
      : frameRect.top + cardHeightPx / 2,
  };
}

function deckSourceElement(): HTMLElement | null {
  const deckStack = document.querySelector<HTMLElement>(
    '.deck-pile-stack.is-deck'
  );
  if (!deckStack) {
    return null;
  }
  const stackCards = deckStack.querySelectorAll<HTMLElement>(
    '.deck-pile-stack-card'
  );
  if (stackCards.length === 0) {
    return deckStack;
  }
  return stackCards[stackCards.length - 1];
}

function discardTargetElement(): HTMLElement | null {
  const discardStack = document.querySelector<HTMLElement>(
    '.deck-pile-stack.is-discard'
  );
  if (!discardStack) {
    return null;
  }
  const stackCards = discardStack.querySelectorAll<HTMLElement>(
    '.deck-pile-stack-card'
  );
  if (stackCards.length === 0) {
    return discardStack;
  }
  return stackCards[stackCards.length - 1];
}

function handSourceElement(
  playerId: PlayerId,
  cardId: CardId
): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  const escapedCardId = cssEscapeValue(cardId);
  const panelSelector = `.player-panel[data-player-id="${escapedPlayerId}"]`;
  return (
    document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-hand-card-id="${escapedCardId}"]`
    ) ??
    document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-card-id="${escapedCardId}"]`
    ) ??
    document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="hidden"]`
    )
  );
}

function handDrawTargetElement(playerId: PlayerId): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  return (
    document.querySelector<HTMLElement>(
      `.player-panel[data-player-id="${escapedPlayerId}"] .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="empty"]`
    ) ??
    document.querySelector<HTMLElement>(
      `.player-panel[data-player-id="${escapedPlayerId}"] .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="hidden"]`
    )
  );
}

function createCardFlight(
  makeFlightId: () => string,
  sourceElement: HTMLElement,
  targetElement: HTMLElement,
  visual: 'face' | 'back',
  options?: {
    cardId?: CardId;
    isDeed?: boolean;
    perspective?: CardPerspective;
    delayMs?: number;
    durationMs?: number;
    variant?: 'play' | 'draw' | 'terminal-clear';
  }
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const targetRect = targetElement.getBoundingClientRect();
  const sourceCenter = elementCenter(sourceElement);
  const targetCenter = elementCenter(targetElement);
  return {
    id: makeFlightId(),
    variant: options?.variant ?? 'play',
    visual,
    cardId: options?.cardId,
    isDeed: options?.isDeed ?? false,
    perspective: options?.perspective ?? 'human',
    startX: sourceCenter.x,
    startY: sourceCenter.y,
    endX: targetCenter.x,
    endY: targetCenter.y,
    startWidth: sourceRect.width,
    startHeight: sourceRect.height,
    endWidth: targetRect.width || sourceRect.width,
    endHeight: targetRect.height || sourceRect.height,
    delayMs: options?.delayMs ?? 0,
    durationMs: options?.durationMs,
  };
}

function createCardFlightToPoint(
  makeFlightId: () => string,
  sourceElement: HTMLElement,
  target: { x: number; y: number },
  visual: 'face' | 'back',
  options?: {
    cardId?: CardId;
    isDeed?: boolean;
    perspective?: CardPerspective;
    delayMs?: number;
    durationMs?: number;
    endWidth?: number;
    endHeight?: number;
    variant?: 'play' | 'draw' | 'terminal-clear';
  }
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const sourceCenter = elementCenter(sourceElement);
  return {
    id: makeFlightId(),
    variant: options?.variant ?? 'play',
    visual,
    cardId: options?.cardId,
    isDeed: options?.isDeed ?? false,
    perspective: options?.perspective ?? 'human',
    startX: sourceCenter.x,
    startY: sourceCenter.y,
    endX: target.x,
    endY: target.y,
    startWidth: sourceRect.width,
    startHeight: sourceRect.height,
    endWidth: options?.endWidth ?? sourceRect.width,
    endHeight: options?.endHeight ?? sourceRect.height,
    delayMs: options?.delayMs ?? 0,
    durationMs: options?.durationMs,
  };
}

function collectCardPlayFlights(
  state: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  makeFlightId: () => string
): CardFlight[] {
  if (typeof document === 'undefined') {
    return [];
  }

  const flights: CardFlight[] = [];

  if (action.type === 'sell-card') {
    const sourceElement = handSourceElement(actingPlayerId, action.cardId);
    const targetElement = discardTargetElement();
    if (sourceElement && targetElement) {
      const sourceSlotKind = sourceElement.getAttribute('data-hand-slot-kind');
      const visual: 'face' | 'back' =
        sourceSlotKind === 'occupied' ? 'face' : 'back';
      const perspective: CardPerspective = sourceElement.classList.contains(
        'perspective-bot'
      )
        ? 'bot'
        : 'human';
      flights.push(
        createCardFlight(makeFlightId, sourceElement, targetElement, visual, {
          cardId: visual === 'face' ? action.cardId : undefined,
          isDeed: false,
          perspective,
        })
      );
    }
  }

  if (isLaneCardPlayAction(action)) {
    const sourceElement = handSourceElement(actingPlayerId, action.cardId);
    if (sourceElement) {
      const escapedPlayerId = cssEscapeValue(actingPlayerId);
      const escapedDistrictId = cssEscapeValue(action.districtId);
      const laneElement = document.querySelector<HTMLElement>(
        `.district-column[data-district-id="${escapedDistrictId}"] .district-lane[data-lane-player-id="${escapedPlayerId}"]`
      );
      const districtColumn = document.querySelector<HTMLElement>(
        `.district-column[data-district-id="${escapedDistrictId}"]`
      );
      const fallbackTargetElement =
        laneElement?.querySelector<HTMLElement>('.lane-stack-frame') ??
        laneElement ??
        districtColumn;
      const targetCenter =
        (laneElement
          ? laneTargetCenter(
              laneElement,
              sourceElement.getBoundingClientRect().height
            )
          : null) ??
        (fallbackTargetElement ? elementCenter(fallbackTargetElement) : null);
      if (targetCenter) {
        const perspective: CardPerspective = laneElement
          ? laneElement.classList.contains('is-bot')
            ? 'bot'
            : 'human'
          : actingPlayerId === BOT_PLAYER
            ? 'bot'
            : 'human';
        flights.push(
          createCardFlightToPoint(
            makeFlightId,
            sourceElement,
            targetCenter,
            'face',
            {
              cardId: action.cardId,
              isDeed: action.type === 'buy-deed',
              perspective,
            }
          )
        );
      }
    }
  }

  if (action.type === 'end-turn') {
    const previousPlayer = state.players.find(
      (player) => player.id === actingPlayerId
    );
    const nextPlayer = nextState.players.find(
      (player) => player.id === actingPlayerId
    );
    const drewCard =
      previousPlayer &&
      nextPlayer &&
      nextPlayer.hand.length === previousPlayer.hand.length + 1;
    if (drewCard) {
      const sourceElement = deckSourceElement();
      const targetElement = handDrawTargetElement(actingPlayerId);
      if (sourceElement && targetElement) {
        flights.push(
          createCardFlight(makeFlightId, sourceElement, targetElement, 'back', {
            delayMs: flights.length > 0 ? CARD_DRAW_FLIGHT_DELAY_MS : 0,
            variant: 'draw',
          })
        );
      }
    }
  }

  return flights;
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
  const [animationsEnabled, setAnimationsEnabled] = useState<boolean>(() =>
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
    useState<Partial<Record<PlayerId, ResourcePool>> | null>(null);
  const [pendingDiscardHoldback, setPendingDiscardHoldback] =
    useState<number>(0);
  const [actionCommitPending, setActionCommitPending] =
    useState<boolean>(false);
  const [
    allowHumanActionsWhileCommitPending,
    setAllowHumanActionsWhileCommitPending,
  ] = useState<boolean>(false);
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
  const actionCommitTimerRef = useRef<number | null>(null);
  const terminalWinnerTimerRef = useRef<number | null>(null);
  const turnCycleVisualTimerRefs = useRef<number[]>([]);
  const taxPulseElementsRef = useRef<HTMLElement[]>([]);
  const nextResourceFlightId = useRef(0);
  const nextCardFlightId = useRef(0);
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
  const clearPendingActionCommit = useCallback(() => {
    if (actionCommitTimerRef.current !== null) {
      window.clearTimeout(actionCommitTimerRef.current);
      actionCommitTimerRef.current = null;
    }
    setActionCommitPending(false);
    setAllowHumanActionsWhileCommitPending(false);
  }, []);
  const clearTerminalWinnerOverlay = useCallback(() => {
    if (terminalWinnerTimerRef.current !== null) {
      window.clearTimeout(terminalWinnerTimerRef.current);
      terminalWinnerTimerRef.current = null;
    }
    setTerminalWinnerOverlayWinner(null);
  }, []);
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
        const element = resourceTokenElementForPlayer(
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
    for (const timerId of turnCycleVisualTimerRefs.current) {
      window.clearTimeout(timerId);
    }
    turnCycleVisualTimerRefs.current = [];
    clearTaxPulseElements();
    setTurnCycleOverlay(null);
    setIncomeHighlightCardIds([]);
    setIncomeHighlightCrowns([]);
    setIncomeResourcePreviewByPlayer(null);
  }, [clearTaxPulseElements]);
  const scheduleTurnCycleVisuals = useCallback(
    (plan: TurnCycleVisualPlan | null) => {
      clearTurnCycleVisuals();
      if (!plan) {
        return;
      }

      const queue = (delayMs: number, run: () => void) => {
        const timerId = window.setTimeout(run, Math.max(0, delayMs));
        turnCycleVisualTimerRefs.current.push(timerId);
      };

      if (plan.taxLabelAtMs !== null && plan.taxSuit) {
        const taxSuit = plan.taxSuit;
        queue(plan.taxLabelAtMs, () => {
          setTurnCycleOverlay({ kind: 'tax', suit: taxSuit });
        });
        if (plan.taxLabelHideAtMs !== null) {
          queue(plan.taxLabelHideAtMs, () => {
            setTurnCycleOverlay(null);
          });
        }
        if (
          plan.taxPulseStartAtMs !== null &&
          plan.taxPulseTargets.length > 0
        ) {
          queue(plan.taxPulseStartAtMs, () => {
            applyTaxPulseTargets(plan.taxPulseTargets);
          });
        }
        if (plan.taxPulseEndAtMs !== null) {
          queue(plan.taxPulseEndAtMs, () => {
            clearTaxPulseElements();
          });
        }
        if (
          plan.taxFlightLaunchAtMs !== null &&
          plan.taxFlightTokens.length > 0
        ) {
          queue(plan.taxFlightLaunchAtMs, () => {
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

      queue(plan.incomeLabelAtMs, () => {
        setTurnCycleOverlay({ kind: 'income', rank: plan.incomeRank });
      });
      queue(plan.incomeLabelHideAtMs, () => {
        setTurnCycleOverlay(null);
      });
      queue(plan.incomeFlightLaunchAtMs, () => {
        const incomeFlights = buildIncomeFlightsFromDom(
          plan.incomeFlightTokens,
          makeResourceFlightId
        );
        if (incomeFlights.length === 0) {
          return;
        }
        setResourceFlights((existing) => [...existing, ...incomeFlights]);
      });

      queue(plan.incomeHighlightStartAtMs, () => {
        setIncomeHighlightCardIds(plan.highlightCardIds);
        setIncomeHighlightCrowns(plan.highlightCrowns);
      });
      queue(plan.incomeHighlightEndAtMs, () => {
        setIncomeHighlightCardIds([]);
        setIncomeHighlightCrowns([]);
      });
      queue(plan.hideAllAtMs, () => {
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
    ]
  );
  const clearAllFlights = useCallback(() => {
    setResourceFlights([]);
    setCardFlights([]);
    setPendingDiscardHoldback(0);
    setAllowHumanActionsWhileCommitPending(false);
    clearTurnCycleVisuals();
    clearTerminalWinnerOverlay();
  }, [clearTerminalWinnerOverlay, clearTurnCycleVisuals]);
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
  const commitStateAfterAnimations = useCallback(
    (
      nextState: GameState,
      queuedResourceFlights: readonly ResourceFlight[],
      queuedCardFlights: readonly CardFlight[],
      options?: {
        commitBeforeSettle?: boolean;
        hideDiscardCountUntilSettle?: number;
        previousState?: GameState;
        action?: GameAction;
        extraSettleMs?: number;
        allowHumanActionsWhileCommitPending?: boolean;
        onSettle?: () => void;
      }
    ) => {
      const settleMs = Math.max(
        resourceFlightSettleMs(queuedResourceFlights),
        cardFlightSettleMs(queuedCardFlights),
        options?.extraSettleMs ?? 0
      );
      if (settleMs <= 0) {
        setPendingDiscardHoldback(0);
        setAllowHumanActionsWhileCommitPending(false);
        if (options?.previousState) {
          commitImmediateTransition(
            options.previousState,
            nextState,
            options.action
          );
        } else {
          setState(nextState);
        }
        options?.onSettle?.();
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
      setPendingDiscardHoldback(
        Math.max(0, options?.hideDiscardCountUntilSettle ?? 0)
      );
      setAllowHumanActionsWhileCommitPending(
        options?.allowHumanActionsWhileCommitPending ?? false
      );
      setActionCommitPending(true);
      if (options?.commitBeforeSettle) {
        if (options?.previousState) {
          commitImmediateTransition(
            options.previousState,
            nextState,
            options.action
          );
        } else {
          setState(nextState);
        }
      }
      if (actionCommitTimerRef.current !== null) {
        window.clearTimeout(actionCommitTimerRef.current);
      }
      actionCommitTimerRef.current = window.setTimeout(() => {
        if (!options?.commitBeforeSettle) {
          if (options?.previousState) {
            commitImmediateTransition(
              options.previousState,
              nextState,
              options.action
            );
          } else {
            setState(nextState);
          }
        }
        setResourceFlights([]);
        setCardFlights([]);
        setPendingDiscardHoldback(0);
        setAllowHumanActionsWhileCommitPending(false);
        clearTurnCycleVisuals();
        options?.onSettle?.();
        setActionCommitPending(false);
        actionCommitTimerRef.current = null;
      }, settleMs);
    },
    [clearTurnCycleVisuals, commitImmediateTransition]
  );
  const actionPopoverLayerRefs = useMemo(() => [actionPopoverRef], []);
  const optionsMenuLayerRefs = useMemo(
    () => [optionsMenuRef, optionsMenuButtonRef],
    []
  );
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    persistAnimationsEnabledPreference(animationsEnabled);
  }, [animationsEnabled]);

  useEffect(() => {
    return () => {
      if (actionCommitTimerRef.current !== null) {
        window.clearTimeout(actionCommitTimerRef.current);
        actionCommitTimerRef.current = null;
      }
      if (terminalWinnerTimerRef.current !== null) {
        window.clearTimeout(terminalWinnerTimerRef.current);
        terminalWinnerTimerRef.current = null;
      }
      clearTurnCycleVisuals();
    };
  }, [clearTurnCycleVisuals]);

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
          const shouldCommitBeforeSettle =
            shouldCommitBeforeAnimationSettle(choice);
          const allowHumanActionsDuringSettle =
            shouldAllowHumanActionsDuringAnimationSettle(choice);
          scheduleTurnCycleVisuals(turnCyclePlan?.visualPlan ?? null);
          if (choice.type === 'end-turn' && turnCyclePlan) {
            setIncomeResourcePreviewByPlayer(
              buildResourcePreviewByPlayer(current)
            );
            const visualPlan = turnCyclePlan.visualPlan;
            if (
              visualPlan.taxFlightLaunchAtMs !== null &&
              visualPlan.taxFlightTokens.length > 0
            ) {
              for (const [
                index,
                token,
              ] of visualPlan.taxFlightTokens.entries()) {
                const taxStepAtMs =
                  visualPlan.taxFlightLaunchAtMs +
                  index * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS;
                const taxTimerId = window.setTimeout(
                  () => {
                    setIncomeResourcePreviewByPlayer((existing) =>
                      applySingleTaxLossToPreview(existing, token)
                    );
                  },
                  Math.max(0, taxStepAtMs)
                );
                turnCycleVisualTimerRefs.current.push(taxTimerId);
              }
            } else if (
              visualPlan.taxResourcesApplyAtMs !== null &&
              visualPlan.taxSuit &&
              visualPlan.taxLossesByPlayer.length > 0
            ) {
              const taxTimerId = window.setTimeout(
                () => {
                  setIncomeResourcePreviewByPlayer((existing) => {
                    let preview = existing;
                    for (const loss of visualPlan.taxLossesByPlayer) {
                      for (let count = 0; count < loss.count; count += 1) {
                        preview = applySingleTaxLossToPreview(preview, {
                          playerId: loss.playerId,
                          suit: visualPlan.taxSuit!,
                        });
                      }
                    }
                    return preview;
                  });
                },
                Math.max(0, visualPlan.taxResourcesApplyAtMs)
              );
              turnCycleVisualTimerRefs.current.push(taxTimerId);
            }
            const previewResources = buildResourcePreviewByPlayer(next);
            const timerId = window.setTimeout(
              () => {
                setIncomeResourcePreviewByPlayer(previewResources);
              },
              Math.max(0, turnCyclePlan.incomeAnimationEndMs)
            );
            turnCycleVisualTimerRefs.current.push(timerId);
          }
          commitStateAfterAnimations(
            next,
            queuedResourceFlights,
            queuedCardFlights,
            {
              commitBeforeSettle: shouldCommitBeforeSettle,
              hideDiscardCountUntilSettle: choice.type === 'sell-card' ? 1 : 0,
              allowHumanActionsWhileCommitPending:
                allowHumanActionsDuringSettle,
              extraSettleMs: turnCyclePlan?.totalDurationMs ?? 0,
              onSettle: enteredTerminal
                ? () => {
                    setTerminalWinnerOverlayWinner(
                      (next.finalScore ?? scoreLive(next)).winner
                    );
                  }
                : undefined,
              previousState: current,
              action: choice,
            }
          );
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
    commitStateAfterAnimations,
    makeCardFlightId,
    makeResourceFlightId,
    resolvedBotProfile,
    scheduleTurnCycleVisuals,
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
      const shouldCommitBeforeSettle =
        shouldCommitBeforeAnimationSettle(action);
      const allowHumanActionsDuringSettle =
        shouldAllowHumanActionsDuringAnimationSettle(action);
      scheduleTurnCycleVisuals(turnCyclePlan?.visualPlan ?? null);
      if (action.type === 'end-turn' && turnCyclePlan) {
        setIncomeResourcePreviewByPlayer(buildResourcePreviewByPlayer(state));
        const visualPlan = turnCyclePlan.visualPlan;
        if (
          visualPlan.taxFlightLaunchAtMs !== null &&
          visualPlan.taxFlightTokens.length > 0
        ) {
          for (const [index, token] of visualPlan.taxFlightTokens.entries()) {
            const taxStepAtMs =
              visualPlan.taxFlightLaunchAtMs +
              index * TURN_CYCLE_TAX_FLIGHT_STAGGER_MS;
            const taxTimerId = window.setTimeout(
              () => {
                setIncomeResourcePreviewByPlayer((existing) =>
                  applySingleTaxLossToPreview(existing, token)
                );
              },
              Math.max(0, taxStepAtMs)
            );
            turnCycleVisualTimerRefs.current.push(taxTimerId);
          }
        } else if (
          visualPlan.taxResourcesApplyAtMs !== null &&
          visualPlan.taxSuit &&
          visualPlan.taxLossesByPlayer.length > 0
        ) {
          const taxTimerId = window.setTimeout(
            () => {
              setIncomeResourcePreviewByPlayer((existing) => {
                let preview = existing;
                for (const loss of visualPlan.taxLossesByPlayer) {
                  for (let count = 0; count < loss.count; count += 1) {
                    preview = applySingleTaxLossToPreview(preview, {
                      playerId: loss.playerId,
                      suit: visualPlan.taxSuit!,
                    });
                  }
                }
                return preview;
              });
            },
            Math.max(0, visualPlan.taxResourcesApplyAtMs)
          );
          turnCycleVisualTimerRefs.current.push(taxTimerId);
        }
        const previewResources = buildResourcePreviewByPlayer(next);
        const timerId = window.setTimeout(
          () => {
            setIncomeResourcePreviewByPlayer(previewResources);
          },
          Math.max(0, turnCyclePlan.incomeAnimationEndMs)
        );
        turnCycleVisualTimerRefs.current.push(timerId);
      }
      commitStateAfterAnimations(
        next,
        queuedResourceFlights,
        queuedCardFlights,
        {
          commitBeforeSettle: shouldCommitBeforeSettle,
          hideDiscardCountUntilSettle: action.type === 'sell-card' ? 1 : 0,
          allowHumanActionsWhileCommitPending: allowHumanActionsDuringSettle,
          extraSettleMs: turnCyclePlan?.totalDurationMs ?? 0,
          onSettle: enteredTerminal
            ? () => {
                setTerminalWinnerOverlayWinner(
                  (next.finalScore ?? scoreLive(next)).winner
                );
              }
            : undefined,
          previousState: state,
          action,
        }
      );
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
