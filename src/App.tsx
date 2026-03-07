import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

import { legalActions } from './engine/actionBuilders';
import { CARD_BY_ID, type CardId } from './engine/cards';
import { rngFromSeed } from './engine/rng';
import { createSession, stepToDecision } from './engine/session';
import {
  districtWinnersByPlayer,
  isTerminal,
  scoreLive,
} from './engine/scoring';
import type { BotProfileId } from './policies/catalog';
import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from './policies/catalog';
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
  actionStableKey,
  buildHumanActionList,
  buildTradeSourceGroups,
  buildPickerOptions,
  cardSummary,
  describeAction,
  formatTokens,
  paymentSignature,
  pickerStillLegal,
  pickerTitle,
  type ActionPickerQuery,
  type HumanActionListItem,
} from './ui/actionPresentation';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from './ui/turnReset';
import { getCardImage } from './ui/cardImages';
import {
  preloadStartupAssets,
  type StartupPreloadProgress,
} from './ui/startupPreload';
import { CardTile, type CardPerspective } from './ui/components/CardTile';
import {
  clearAllDeedTokenLayouts,
  layoutDeedTokensBySide,
  resetDeedTokenLayout,
  type DeedTokenSide,
} from './ui/components/deedTokenLayout';
import { DistrictColumn, PlayerTokenRail } from './ui/components/DistrictBoard';
import { PlayerPanel } from './ui/components/PlayerPanel';
import { RollResult } from './ui/components/RollResult';
import {
  SUIT_TOKEN_BG,
  TokenChip,
  tokenEntries,
} from './ui/components/TokenComponents';
import { useDismissableLayer } from './ui/hooks/useDismissableLayer';
import { transitionLogEntries } from './ui/logTimeline';
import {
  deriveTurnCycleEvents,
  type TurnCycleIncomeToken,
} from './ui/turnCycleEvents';
import {
  SUIT_TEXT_TOKEN,
  SUIT_TOKEN_REGEX,
  SUIT_TOKEN_TO_SUIT,
} from './ui/suitIcons';

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const DEFAULT_BOT_DELAY_MS = 450;
const PLAYER_HAND_SLOT_COUNT = 3;
const TRADE_POPOVER_WIDTH_PX = 220;
const TRADE_POPOVER_MIN_HEIGHT_PX = 188;
const TRADE_POPOVER_GAP_PX = 8;
const VIEWPORT_PADDING_PX = 10;
const RESOURCE_FLIGHT_DURATION_MS = 280;
const RESOURCE_FLIGHT_STAGGER_MS = 75;
const CARD_FLIGHT_DURATION_MS = 280;
const CARD_DRAW_FLIGHT_DELAY_MS = CARD_FLIGHT_DURATION_MS;
const DEFAULT_TOKEN_CHIP_SIZE_PX = 22;
const DEFAULT_TOKEN_RAIL_GAP_PX = 2.56;
const ACTION_FLIGHT_COMMIT_BUFFER_MS = 20;
const TURN_CYCLE_TEXT_ONLY_MS = 2000;
const TURN_CYCLE_INCOME_PRE_FLIGHT_MS = 3000;
const TURN_CYCLE_STAGE_GAP_MS = 220;
const TURN_CYCLE_TAX_PULSE_MS = 700;
const TURN_CYCLE_TAX_FLIGHT_DURATION_MS = 900;
const TURN_CYCLE_INCOME_FLIGHT_DURATION_MS = 560;
const TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS = 95;
const TURN_CYCLE_POST_INCOME_HOLD_MS = 220;
const ANIMATIONS_STORAGE_KEY = 'magnate:animationsEnabled';
const SUIT_LOG_CODE: Record<Suit, string> = {
  Moons: 'mo',
  Suns: 'su',
  Waves: 'wa',
  Leaves: 'le',
  Wyrms: 'wy',
  Knots: 'kn',
};
const SUIT_NAME_PATTERN = /\b(Moons|Suns|Waves|Leaves|Wyrms|Knots)\b/g;
const CARD_ACTION_PATTERN = /\b(buy deed|sell|advance|develop)\s+(\d+)\b/gi;
const INCOME_CHOICE_PATTERN = /\bincome choice\s+(\d+):([A-Za-z]+)\b/gi;
const SUIT_CODE_PATTERN = /\b(mo|su|wa|le|wy|kn)\b/g;
const STARTUP_PRELOAD_INITIAL_PROGRESS: StartupPreloadProgress = {
  completed: 0,
  total: 1,
  percent: 0,
  message: 'Loading card images and bot models...',
};

type ActionPickerState =
  | (ActionPickerQuery & {
      top: number;
      left: number;
    })
  | {
      kind: 'trade-combined';
      top: number;
      left: number;
      selectedGive?: Suit;
      selectedReceive?: Suit;
    }
  | {
      kind: 'develop-outright-combined';
      top: number;
      left: number;
      cardId: CardId;
      selectedDistrictId?: string;
      selectedPaymentKey?: string;
    };

type ResourceFlight = {
  id: string;
  suit: Suit;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  delayMs: number;
  durationMs?: number;
  variant?: 'transfer' | 'tax-loss';
};

type PendingResourceFlight = {
  id: string;
  suit: Suit;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  delayMs: number;
  durationMs?: number;
  variant?: 'transfer' | 'tax-loss';
};

type TurnCycleOverlayState =
  | {
      kind: 'tax';
      suit: Suit;
    }
  | {
      kind: 'income';
      rank: number;
    };

type TurnCycleVisualPlan = {
  taxLabelAtMs: number | null;
  taxLabelHideAtMs: number | null;
  taxPulseStartAtMs: number | null;
  taxPulseEndAtMs: number | null;
  taxFlightLaunchAtMs: number | null;
  taxResourcesApplyAtMs: number | null;
  taxPulseTargets: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>;
  taxLossesByPlayer: ReadonlyArray<{
    playerId: PlayerId;
    count: number;
  }>;
  incomeLabelAtMs: number;
  incomeLabelHideAtMs: number;
  incomeFlightLaunchAtMs: number;
  incomeFlightTokens: ReadonlyArray<TurnCycleIncomeToken>;
  incomeHighlightStartAtMs: number;
  incomeHighlightEndAtMs: number;
  hideAllAtMs: number;
  taxSuit: Suit | null;
  incomeRank: number;
  highlightCardIds: ReadonlyArray<CardId>;
};

type CardFlight = {
  id: string;
  variant: 'play' | 'draw';
  visual: 'face' | 'back';
  cardId?: CardId;
  isDeed: boolean;
  perspective: CardPerspective;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  startWidth: number;
  startHeight: number;
  endWidth: number;
  endHeight: number;
  delayMs: number;
};

function makeSeed(): string {
  return `seed-${Date.now()}`;
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
  const firstTokenCenterOffset = (-totalHeight / 2) + (tokenSizePx / 2);
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

  let cursorMs = baseDelayMs;

  let taxLabelAtMs: number | null = null;
  let taxLabelHideAtMs: number | null = null;
  let taxPulseStartAtMs: number | null = null;
  let taxPulseEndAtMs: number | null = null;
  let taxFlightLaunchAtMs: number | null = null;
  let taxResourcesApplyAtMs: number | null = null;
  const taxPulseTargets: Array<{ playerId: PlayerId; suit: Suit }> = [];
  const taxLossesByPlayer: Array<{ playerId: PlayerId; count: number }> = [];
  let taxSuit: Suit | null = null;
  if (cycle.tax) {
    taxLabelAtMs = cursorMs;
    taxLabelHideAtMs = cursorMs + TURN_CYCLE_TEXT_ONLY_MS;
    taxSuit = cycle.tax.suit;
    const taxedPlayers = cycle.tax.lossesByPlayer.filter((entry) => entry.count > 0);
    taxLossesByPlayer.push(...taxedPlayers);
    if (taxedPlayers.length > 0) {
      taxPulseStartAtMs = taxLabelHideAtMs;
      taxPulseEndAtMs = taxPulseStartAtMs + TURN_CYCLE_TAX_PULSE_MS;
      taxPulseTargets.push(
        ...taxedPlayers.map((entry) => ({
          playerId: entry.playerId,
          suit: cycle.tax!.suit,
        }))
      );
    }
    const taxAnimationStartMs =
      (taxPulseEndAtMs !== null ? taxPulseEndAtMs + 120 : taxLabelHideAtMs);
    taxFlightLaunchAtMs = taxedPlayers.length > 0 ? taxAnimationStartMs : null;
    const taxAnimationEndMs =
      taxedPlayers.length > 0
        ? taxAnimationStartMs
          + ((taxedPlayers.length - 1) * RESOURCE_FLIGHT_STAGGER_MS)
          + TURN_CYCLE_TAX_FLIGHT_DURATION_MS
        : taxAnimationStartMs;
    taxResourcesApplyAtMs = taxedPlayers.length > 0 ? taxAnimationEndMs : null;
    cursorMs = taxAnimationEndMs + TURN_CYCLE_STAGE_GAP_MS;
  }

  const incomeLabelAtMs = cursorMs;
  const incomeLabelHideAtMs = incomeLabelAtMs + TURN_CYCLE_INCOME_PRE_FLIGHT_MS;
  const incomeAnimationStartMs = incomeLabelHideAtMs;
  const incomeAnimationEndMs =
    cycle.incomeTokens.length > 0
      ? incomeAnimationStartMs
        + ((cycle.incomeTokens.length - 1) * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS)
        + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS
      : incomeAnimationStartMs;

  const highlightCardIds = [...new Set(cycle.incomeHighlights.map((entry) => entry.cardId))];
  const incomeHighlightStartAtMs = incomeLabelAtMs;
  const minimumHighlightDurationMs =
    highlightCardIds.length > 0
      ? TURN_CYCLE_INCOME_PRE_FLIGHT_MS + TURN_CYCLE_INCOME_FLIGHT_DURATION_MS
      : 0;
  const incomeHighlightEndAtMs = Math.max(
    incomeHighlightStartAtMs + minimumHighlightDurationMs,
    incomeAnimationEndMs
  );

  const hideAllAtMs =
    Math.max(incomeHighlightEndAtMs, incomeAnimationEndMs) + TURN_CYCLE_POST_INCOME_HOLD_MS;

  return {
    visualPlan: {
      taxLabelAtMs,
      taxLabelHideAtMs,
      taxPulseStartAtMs,
      taxPulseEndAtMs,
      taxFlightLaunchAtMs,
      taxResourcesApplyAtMs,
      taxPulseTargets,
      taxLossesByPlayer,
      incomeLabelAtMs,
      incomeLabelHideAtMs,
      incomeFlightLaunchAtMs: incomeAnimationStartMs,
      incomeFlightTokens: cycle.incomeTokens,
      incomeHighlightStartAtMs,
      incomeHighlightEndAtMs,
      hideAllAtMs,
      taxSuit,
      incomeRank: cycle.incomeRank,
      highlightCardIds,
    },
    totalDurationMs: hideAllAtMs + ACTION_FLIGHT_COMMIT_BUFFER_MS,
    incomeAnimationEndMs,
  };
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
    const sourceElement = resourceTokenElementForPlayer(target.playerId, target.suit);
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
      delayMs: index * RESOURCE_FLIGHT_STAGGER_MS,
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
    const targetElement = resourceTokenElementForPlayer(token.playerId, token.suit);
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

  const district = state.districts.find((candidate) => candidate.id === action.districtId);
  const deedBefore = district?.stacks[actingPlayerId]?.deed;
  if (!deedBefore || deedBefore.cardId !== action.cardId) {
    return [];
  }

  const nextDeedTokens: Partial<Record<Suit, number>> = { ...deedBefore.tokens };
  for (const entry of tokenEntries(action.tokens)) {
    nextDeedTokens[entry.suit] = (nextDeedTokens[entry.suit] ?? 0) + entry.count;
  }

  const perspective: 'human' | 'bot' = cardElement.classList.contains('perspective-bot')
    ? 'bot'
    : 'human';
  const deedTokenEntries = tokenEntries(deedBefore.tokens);
  if (deedTokenEntries.length === 0) {
    resetDeedTokenLayout(action.cardId, perspective);
  }
  const nextTokenEntries = tokenEntries(nextDeedTokens);
  const nextBySide = layoutDeedTokensBySide(action.cardId, perspective, nextTokenEntries);
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
  const targetElement = resourceTokenElementForPlayer(action.playerId, action.suit);
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

function stackStepForLane(laneElement: HTMLElement, isBotLane: boolean, cardHeightPx: number): number {
  const stackStepValue = window.getComputedStyle(laneElement).getPropertyValue('--card-stack-step').trim();
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

  const topStackCard = laneElement.querySelector<HTMLElement>('.lane-stack-card:last-child');
  if (!topStackCard) {
    return cardHeightPx * 0.24;
  }

  const stackPosition = Number.parseFloat(topStackCard.style.getPropertyValue('--stack-position'));
  if (!Number.isFinite(stackPosition) || stackPosition <= 0) {
    return cardHeightPx * 0.24;
  }

  const computed = window.getComputedStyle(topStackCard);
  const offsetPx = parsePixelValue(isBotLane ? computed.bottom : computed.top, 0);
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
  const topCardTile = laneElement.querySelector<HTMLElement>('.lane-stack-card:last-child .card-tile');
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
      ? frameRect.bottom - (cardHeightPx / 2)
      : frameRect.top + (cardHeightPx / 2),
  };
}

function deckSourceElement(): HTMLElement | null {
  const deckStack = document.querySelector<HTMLElement>('.deck-pile-stack.is-deck');
  if (!deckStack) {
    return null;
  }
  const stackCards = deckStack.querySelectorAll<HTMLElement>('.deck-pile-stack-card');
  if (stackCards.length === 0) {
    return deckStack;
  }
  return stackCards[stackCards.length - 1];
}

function discardTargetElement(): HTMLElement | null {
  const discardStack = document.querySelector<HTMLElement>('.deck-pile-stack.is-discard');
  if (!discardStack) {
    return null;
  }
  const stackCards = discardStack.querySelectorAll<HTMLElement>('.deck-pile-stack-card');
  if (stackCards.length === 0) {
    return discardStack;
  }
  return stackCards[stackCards.length - 1];
}

function handSourceElement(playerId: PlayerId, cardId: CardId): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  const escapedCardId = cssEscapeValue(cardId);
  const panelSelector = `.player-panel[data-player-id="${escapedPlayerId}"]`;
  return (
    document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-hand-card-id="${escapedCardId}"]`
    )
    ?? document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-card-id="${escapedCardId}"]`
    )
    ?? document.querySelector<HTMLElement>(
      `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="hidden"]`
    )
  );
}

function handDrawTargetElement(playerId: PlayerId): HTMLElement | null {
  const escapedPlayerId = cssEscapeValue(playerId);
  return (
    document.querySelector<HTMLElement>(
      `.player-panel[data-player-id="${escapedPlayerId}"] .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="empty"]`
    )
    ?? document.querySelector<HTMLElement>(
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
    variant?: 'play' | 'draw';
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
    endWidth?: number;
    endHeight?: number;
    variant?: 'play' | 'draw';
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
        laneElement?.querySelector<HTMLElement>('.lane-stack-frame')
        ?? laneElement
        ?? districtColumn;
      const targetCenter =
        (laneElement
          ? laneTargetCenter(
              laneElement,
              sourceElement.getBoundingClientRect().height
            )
          : null)
        ?? (fallbackTargetElement ? elementCenter(fallbackTargetElement) : null);
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
            delayMs:
              flights.length > 0 ? CARD_DRAW_FLIGHT_DELAY_MS : 0,
            variant: 'draw',
          })
        );
      }
    }
  }

  return flights;
}

function resourceFlightSettleMs(flights: readonly ResourceFlight[]): number {
  if (flights.length === 0) {
    return 0;
  }
  const latestEndMs = Math.max(
    ...flights.map(
      (flight) => flight.delayMs + (flight.durationMs ?? RESOURCE_FLIGHT_DURATION_MS)
    )
  );
  return latestEndMs + ACTION_FLIGHT_COMMIT_BUFFER_MS;
}

function hasPendingIncomeChoices(state: GameState): boolean {
  return (state.pendingIncomeChoices?.length ?? 0) > 0;
}

function shouldCommitBeforeAnimationSettle(
  action: GameAction,
  nextState: GameState
): boolean {
  return (
    action.type === 'sell-card'
    || action.type === 'choose-income-suit'
    || (action.type === 'end-turn' && hasPendingIncomeChoices(nextState))
  );
}

function shouldAllowHumanActionsDuringAnimationSettle(action: GameAction): boolean {
  return action.type === 'end-turn' || action.type === 'choose-income-suit';
}

function buildResourcePreviewByPlayer(
  state: GameState
): Partial<Record<PlayerId, ResourcePool>> {
  const preview: Partial<Record<PlayerId, ResourcePool>> = {};
  for (const player of state.players) {
    preview[player.id] = { ...player.resources };
  }
  return preview;
}

function buildTaxLossPreviewByPlayer(
  state: GameState,
  taxSuit: Suit,
  lossesByPlayer: ReadonlyArray<{ playerId: PlayerId; count: number }>
): Partial<Record<PlayerId, ResourcePool>> {
  const preview = buildResourcePreviewByPlayer(state);
  for (const loss of lossesByPlayer) {
    const resources = preview[loss.playerId];
    if (!resources || loss.count <= 0) {
      continue;
    }
    resources[taxSuit] = Math.max(0, resources[taxSuit] - loss.count);
  }
  return preview;
}

function cardFlightSettleMs(flights: readonly CardFlight[]): number {
  if (flights.length === 0) {
    return 0;
  }
  const maxDelayMs = Math.max(...flights.map((flight) => flight.delayMs));
  return maxDelayMs + CARD_FLIGHT_DURATION_MS + ACTION_FLIGHT_COMMIT_BUFFER_MS;
}

export function App() {
  const [seedInput, setSeedInput] = useState<string>(() => makeSeed());
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [state, setState] = useState<GameState>(() =>
    createInitialState(seedInput)
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
  const [resourceFlights, setResourceFlights] = useState<ReadonlyArray<ResourceFlight>>([]);
  const [cardFlights, setCardFlights] = useState<ReadonlyArray<CardFlight>>([]);
  const [turnCycleOverlay, setTurnCycleOverlay] =
    useState<TurnCycleOverlayState | null>(null);
  const [incomeHighlightCardIds, setIncomeHighlightCardIds] =
    useState<ReadonlyArray<CardId>>([]);
  const [incomeResourcePreviewByPlayer, setIncomeResourcePreviewByPlayer] =
    useState<Partial<Record<PlayerId, ResourcePool>> | null>(null);
  const [pendingDiscardHoldback, setPendingDiscardHoldback] = useState<number>(0);
  const [actionCommitPending, setActionCommitPending] = useState<boolean>(false);
  const [allowHumanActionsWhileCommitPending, setAllowHumanActionsWhileCommitPending] =
    useState<boolean>(false);
  const [startupPreloadReady, setStartupPreloadReady] = useState<boolean>(false);
  const [startupPreloadError, setStartupPreloadError] = useState<string | null>(null);
  const [startupPreloadAttempt, setStartupPreloadAttempt] = useState<number>(0);
  const [startupPreloadProgress, setStartupPreloadProgress] =
    useState<StartupPreloadProgress>(STARTUP_PRELOAD_INITIAL_PROGRESS);
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);
  const [turnResetTimelineAnchor, setTurnResetTimelineAnchor] =
    useState<ReadonlyArray<GameLogEntry> | null>(null);

  const stateRef = useRef(state);
  const actionCommitTimerRef = useRef<number | null>(null);
  const turnCycleVisualTimerRefs = useRef<number[]>([]);
  const taxPulseElementsRef = useRef<HTMLElement[]>([]);
  const nextResourceFlightId = useRef(0);
  const nextCardFlightId = useRef(0);
  const actionPopoverRef = useRef<HTMLElement | null>(null);
  const optionsMenuRef = useRef<HTMLElement | null>(null);
  const optionsMenuButtonRef = useRef<HTMLButtonElement | null>(null);
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
        const element = resourceTokenElementForPlayer(target.playerId, target.suit);
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
        if (plan.taxPulseStartAtMs !== null && plan.taxPulseTargets.length > 0) {
          queue(plan.taxPulseStartAtMs, () => {
            applyTaxPulseTargets(plan.taxPulseTargets);
          });
        }
        if (plan.taxPulseEndAtMs !== null) {
          queue(plan.taxPulseEndAtMs, () => {
            clearTaxPulseElements();
          });
        }
        if (plan.taxFlightLaunchAtMs !== null && plan.taxPulseTargets.length > 0) {
          queue(plan.taxFlightLaunchAtMs, () => {
            const taxFlights = buildTaxLossFlightsFromDom(
              plan.taxPulseTargets,
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
      });
      queue(plan.incomeHighlightEndAtMs, () => {
        setIncomeHighlightCardIds([]);
      });
      queue(plan.hideAllAtMs, () => {
        clearTaxPulseElements();
        setTurnCycleOverlay(null);
        setIncomeHighlightCardIds([]);
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
  }, [clearTurnCycleVisuals]);
  const commitImmediateTransition = useCallback(
    (
      previousState: GameState,
      nextState: GameState,
      action?: GameAction
    ) => {
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
        return;
      }

      if (queuedResourceFlights.length > 0) {
        setResourceFlights((existing) => [...existing, ...queuedResourceFlights]);
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
  const isSecondShuffle = humanView.deck.reshuffles > 0;
  const showSecondShuffleLabel =
    isSecondShuffle && !(terminal && humanView.deck.drawCount === 0);
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

  const humanActions = useMemo(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      return [] as readonly GameAction[];
    }
    return legalActions(state);
  }, [activePlayerId, state, terminal]);

  const humanActionItems = useMemo(
    () => buildHumanActionList(humanActions),
    [humanActions]
  );
  const tradeSourceGroups = useMemo(
    () => buildTradeSourceGroups(humanActions),
    [humanActions]
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

  const actionPickerOptions = useMemo(() => {
    if (
      !actionPicker
      || actionPicker.kind === 'trade-combined'
      || actionPicker.kind === 'develop-outright-combined'
    ) {
      return [];
    }
    return buildPickerOptions(
      toPickerQuery(actionPicker),
      humanActions,
      SUIT_TEXT_TOKEN
    );
  }, [actionPicker, humanActions]);

  const actionPickerTitle = useMemo((): string => {
    if (!actionPicker) {
      return '';
    }
    if (actionPicker.kind === 'trade-combined') {
      return 'Trade resources';
    }
    if (actionPicker.kind === 'develop-outright-combined') {
      return `Develop ${cardSummary(actionPicker.cardId, SUIT_TEXT_TOKEN)}`;
    }
    return pickerTitle(toPickerQuery(actionPicker), SUIT_TEXT_TOKEN);
  }, [actionPicker]);

  const canResetTurn = useMemo(
    () => canUseTurnReset(state, activePlayerId, HUMAN_PLAYER, turnResetAnchor),
    [activePlayerId, state, turnResetAnchor]
  );
  const isTurnCycleAnimationLock =
    actionCommitPending && allowHumanActionsWhileCommitPending;
  const showBotThinkingDuringIncomeChoiceLock =
    isTurnCycleAnimationLock
    && activePlayerId === BOT_PLAYER
    && state.phase === 'CollectIncome'
    && (state.pendingIncomeChoices?.length ?? 0) > 0;
  const hideBotWaitMessageDuringTurnCycleLock =
    isTurnCycleAnimationLock && !showBotThinkingDuringIncomeChoiceLock;

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
      terminal
      || activePlayerId !== BOT_PLAYER
      || actionCommitPending
      || !startupPreloadReady
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
          const queuedCardFlights = collectCardPlayFlights(
            current,
            next,
            choice,
            actingPlayerId,
            makeCardFlightId
          );
          const turnCyclePlan = collectTurnCycleAnimationPlan(
            current,
            next,
            choice,
            cardFlightSettleMs(queuedCardFlights)
          );
          const queuedResourceFlights = [...queuedActionResourceFlights];
          const shouldCommitBeforeSettle = shouldCommitBeforeAnimationSettle(
            choice,
            next
          );
          const allowHumanActionsDuringSettle =
            shouldAllowHumanActionsDuringAnimationSettle(choice);
          scheduleTurnCycleVisuals(turnCyclePlan?.visualPlan ?? null);
          if (
            choice.type === 'end-turn'
            && !shouldCommitBeforeSettle
            && turnCyclePlan
          ) {
            const visualPlan = turnCyclePlan.visualPlan;
            if (
              visualPlan.taxResourcesApplyAtMs !== null
              && visualPlan.taxSuit
              && visualPlan.taxLossesByPlayer.length > 0
            ) {
              const taxPreviewResources = buildTaxLossPreviewByPlayer(
                current,
                visualPlan.taxSuit,
                visualPlan.taxLossesByPlayer
              );
              const taxTimerId = window.setTimeout(() => {
                setIncomeResourcePreviewByPlayer(taxPreviewResources);
              }, Math.max(0, visualPlan.taxResourcesApplyAtMs));
              turnCycleVisualTimerRefs.current.push(taxTimerId);
            }
            const previewResources = buildResourcePreviewByPlayer(next);
            const timerId = window.setTimeout(() => {
              setIncomeResourcePreviewByPlayer(previewResources);
            }, Math.max(0, turnCyclePlan.incomeAnimationEndMs));
            turnCycleVisualTimerRefs.current.push(timerId);
          }
          commitStateAfterAnimations(next, queuedResourceFlights, queuedCardFlights, {
            commitBeforeSettle: shouldCommitBeforeSettle,
            hideDiscardCountUntilSettle: choice.type === 'sell-card' ? 1 : 0,
            allowHumanActionsWhileCommitPending: allowHumanActionsDuringSettle,
            extraSettleMs: turnCyclePlan?.totalDurationMs ?? 0,
            previousState: current,
            action: choice,
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

    if (actionPicker) {
      if (actionPicker.kind === 'trade-combined') {
        if (!hasMultipleTradeSources) {
          closeActionPicker();
          return;
        }
        const selectedGiveStillLegal = Boolean(
          actionPicker.selectedGive
          && tradeSourceGroups.some((group) => group.give === actionPicker.selectedGive)
        );
        if (actionPicker.selectedGive && !selectedGiveStillLegal) {
          closeActionPicker();
        }
        return;
      }
      if (actionPicker.kind === 'develop-outright-combined') {
        const outrightOptions = humanActions.filter(
          (action): action is Extract<GameAction, { type: 'develop-outright' }> =>
            action.type === 'develop-outright'
            && action.cardId === actionPicker.cardId
        );
        if (outrightOptions.length <= 1) {
          closeActionPicker();
          return;
        }
        const selectedDistrictStillLegal = Boolean(
          actionPicker.selectedDistrictId
          && outrightOptions.some(
            (option) => option.districtId === actionPicker.selectedDistrictId
          )
        );
        if (actionPicker.selectedDistrictId && !selectedDistrictStillLegal) {
          closeActionPicker();
          return;
        }
        const selectedPaymentStillLegal = Boolean(
          actionPicker.selectedPaymentKey
          && outrightOptions.some(
            (option) =>
              paymentSignature(option.payment) === actionPicker.selectedPaymentKey
          )
        );
        if (actionPicker.selectedPaymentKey && !selectedPaymentStillLegal) {
          closeActionPicker();
        }
        return;
      }
      const stillLegal = pickerStillLegal(
        toPickerQuery(actionPicker),
        humanActions
      );
      if (!stillLegal) {
        closeActionPicker();
      }
    }
  }, [
    activePlayerId,
    humanActions,
    terminal,
    actionPicker,
    closeActionPicker,
    hasMultipleTradeSources,
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
        allowHumanActionsWhileCommitPending && action.type === 'choose-income-suit';
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
      const queuedCardFlights = collectCardPlayFlights(
        state,
        next,
        action,
        activePlayerId,
        makeCardFlightId
      );
      const turnCyclePlan = collectTurnCycleAnimationPlan(
        state,
        next,
        action,
        cardFlightSettleMs(queuedCardFlights)
      );
      const queuedResourceFlights = [...queuedActionResourceFlights];
      const shouldCommitBeforeSettle = shouldCommitBeforeAnimationSettle(
        action,
        next
      );
      const allowHumanActionsDuringSettle =
        shouldAllowHumanActionsDuringAnimationSettle(action);
      scheduleTurnCycleVisuals(turnCyclePlan?.visualPlan ?? null);
      if (
        action.type === 'end-turn'
        && !shouldCommitBeforeSettle
        && turnCyclePlan
      ) {
        const visualPlan = turnCyclePlan.visualPlan;
        if (
          visualPlan.taxResourcesApplyAtMs !== null
          && visualPlan.taxSuit
          && visualPlan.taxLossesByPlayer.length > 0
        ) {
          const taxPreviewResources = buildTaxLossPreviewByPlayer(
            state,
            visualPlan.taxSuit,
            visualPlan.taxLossesByPlayer
          );
          const taxTimerId = window.setTimeout(() => {
            setIncomeResourcePreviewByPlayer(taxPreviewResources);
          }, Math.max(0, visualPlan.taxResourcesApplyAtMs));
          turnCycleVisualTimerRefs.current.push(taxTimerId);
        }
        const previewResources = buildResourcePreviewByPlayer(next);
        const timerId = window.setTimeout(() => {
          setIncomeResourcePreviewByPlayer(previewResources);
        }, Math.max(0, turnCyclePlan.incomeAnimationEndMs));
        turnCycleVisualTimerRefs.current.push(timerId);
      }
      commitStateAfterAnimations(next, queuedResourceFlights, queuedCardFlights, {
        commitBeforeSettle: shouldCommitBeforeSettle,
        hideDiscardCountUntilSettle: action.type === 'sell-card' ? 1 : 0,
        allowHumanActionsWhileCommitPending: allowHumanActionsDuringSettle,
        extraSettleMs: turnCyclePlan?.totalDurationMs ?? 0,
        previousState: state,
        action,
      });
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleReset = () => {
    const seed = seedInput.trim() || makeSeed();
    setSeedInput(seed);
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

  const recentLog = [...timelineLog].reverse();
  const recentLogGroups = groupLogEntriesByTurn(recentLog);
  const deckStackCount = Math.min(3, humanView.deck.drawCount);
  const deckOverlayShiftClass =
    deckStackCount >= 3
      ? 'overlay-shift-2'
      : deckStackCount === 2
        ? 'overlay-shift-1'
        : 'overlay-shift-0';
  const visibleDiscardCards =
    pendingDiscardHoldback > 0
      ? humanView.deck.discard.slice(pendingDiscardHoldback)
      : humanView.deck.discard;
  const discardStackCardIds = visibleDiscardCards.slice(0, 3).reverse();
  const discardCardDetails = visibleDiscardCards.map((cardId) => {
    const card = CARD_BY_ID[cardId];
    const rank =
      card.kind === 'Property' || card.kind === 'Crown'
        ? String(card.rank)
        : card.kind;
    const suitTokenText =
      card.kind === 'Excuse'
        ? ''
        : card.suits.map((suit) => SUIT_TEXT_TOKEN[suit]).join(' ');
    return {
      id: card.id,
      name: card.name,
      rank,
      suitTokenText,
    };
  });
  const startupPreloadPercent = clamp(startupPreloadProgress.percent, 0, 100);
  const startupPreloadCompleted = Math.min(
    startupPreloadProgress.completed,
    startupPreloadProgress.total
  );
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

          <section className="panel actions-panel">
            <div className="actions-heading">
              <h2>{terminal ? 'Game Over' : 'Actions'}</h2>
              {isLastTurn && <span className="last-turn-badge">Last Turn</span>}
            </div>
            <div className="actions-body">
              {terminal ? (
                <section
                  className="terminal-score-summary"
                  aria-label="Final score breakdown"
                >
                  <p className="score-result terminal-score-winner">
                    Winner: <strong>{winnerLabel(score.winner)}</strong>
                  </p>
                  <p className="score-line terminal-score-decider">
                    <span>Decided By</span>
                    <strong>{deciderLabel(score.decidedBy)}</strong>
                  </p>

                  <div className="terminal-score-players">
                    {([HUMAN_PLAYER, BOT_PLAYER] as const).map((playerId) => (
                      <article
                        key={`terminal-score-${playerId}`}
                        className="terminal-score-player"
                      >
                        <h3>{playerId === HUMAN_PLAYER ? 'You' : 'Bot'}</h3>
                        <p className="score-line">
                          <span>Districts Won</span>
                          <strong>
                            {formatDistrictList(wonDistrictsByPlayer[playerId])}
                          </strong>
                        </p>
                        <p className="score-line">
                          <span>Total Properties</span>
                          <strong>{score.rankTotals[playerId]}</strong>
                        </p>
                        <p className="score-line">
                          <span>Resources</span>
                          <strong>{score.resourceTotals[playerId]}</strong>
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ) : activePlayerId === HUMAN_PLAYER ? (
                <div className="actions-human-layout">
                  <div className="actions-human-main">
                    {visibleHumanActionItems.length === 0 ? (
                      <p className="empty-note">No legal actions.</p>
                    ) : (
                      <div className="action-list">
                        {visibleHumanActionItems.map((item, index) => {
                          const categoryKey = actionCategoryForItem(item);
                          const previousCategoryKey =
                            index > 0
                              ? actionCategoryForItem(
                                  visibleHumanActionItems[index - 1]
                                )
                              : null;
                          const showCategory =
                            previousCategoryKey !== categoryKey;
                          const categoryLabel =
                            actionCategoryLabel(categoryKey);
                          const renderCategorizedAction = (
                            key: string,
                            button: ReactNode
                          ) => (
                            <div
                              key={key}
                              className={`action-entry${showCategory ? ' has-category' : ''}`}
                            >
                              {showCategory ? (
                                <p className="action-category">
                                  {categoryLabel}
                                </p>
                              ) : null}
                              {button}
                            </div>
                          );

                          if (item.kind === 'trade-group') {
                            if (hasMultipleTradeSources) {
                              return renderCategorizedAction(
                                'trade-source-group',
                                <button
                                  type="button"
                                  className="action-button has-submenu"
                                  onClick={(event) => {
                                    const trigger = event.currentTarget;
                                    if (actionPicker?.kind === 'trade-combined') {
                                      closeActionPicker();
                                      return;
                                    }
                                    openTradeCombinedPicker(trigger);
                                  }}
                                >
                                  <span className="action-text">
                                    Trade resources
                                  </span>
                                </button>
                              );
                            }

                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `trade-direct-${item.give}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `trade-group-${item.give}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'trade' &&
                                    actionPicker.give === item.give
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openTradePicker(
                                    item.give,
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Trade ${SUIT_TEXT_TOKEN[item.give]}x3`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'buy-deed-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `buy-deed-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `buy-deed-group-${item.cardId}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'district' &&
                                    actionPicker.actionType === 'buy-deed' &&
                                    actionPicker.cardId === item.cardId
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openDistrictPicker(
                                    {
                                      actionType: 'buy-deed',
                                      cardId: item.cardId,
                                    },
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Buy deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'develop-deed-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `develop-deed-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `develop-deed-group-${item.cardId}-${item.districtId}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'deed-payment' &&
                                    actionPicker.cardId === item.cardId &&
                                    actionPicker.districtId === item.districtId
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openDeedPaymentPicker(
                                    {
                                      cardId: item.cardId,
                                      districtId: item.districtId,
                                    },
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Develop deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'develop-outright-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `develop-outright-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            const firstOptionByPayment = new Map<
                              string,
                              Extract<GameAction, { type: 'develop-outright' }>
                            >();
                            const districtIds = new Set<string>();
                            for (const option of item.options) {
                              districtIds.add(option.districtId);
                              const paymentKey = paymentSignature(option.payment);
                              if (!firstOptionByPayment.has(paymentKey)) {
                                firstOptionByPayment.set(paymentKey, option);
                              }
                            }
                            const paymentOptions = [...firstOptionByPayment.values()];
                            const hasSinglePaymentPattern = paymentOptions.length === 1;

                            return renderCategorizedAction(
                              `develop-outright-group-${item.cardId}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    (
                                      actionPicker?.kind === 'develop-outright-combined'
                                      || actionPicker?.kind === 'develop-outright-district'
                                    )
                                    && actionPicker.cardId === item.cardId
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  if (hasSinglePaymentPattern) {
                                    openDevelopOutrightDistrictOnlyPicker(
                                      item.cardId,
                                      trigger,
                                      districtIds.size
                                    );
                                    return;
                                  }
                                  openDevelopOutrightCombinedPicker(
                                    item.cardId,
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    hasSinglePaymentPattern
                                      ? `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} (${formatTokens(paymentOptions[0].payment, SUIT_TEXT_TOKEN)})`
                                      : `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          return renderCategorizedAction(
                            actionStableKey(item.action),
                            <button
                              type="button"
                              className="action-button"
                              onClick={() => handleHumanAction(item.action)}
                            >
                              <span className="action-text">
                                {renderSuitText(
                                  describeAction(item.action, SUIT_TEXT_TOKEN)
                                )}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {canResetTurn ? (
                    <div className="actions-footer">
                      <button
                        key="reset-turn"
                        type="button"
                        className="action-button reset-turn-button"
                        onClick={handleTurnReset}
                      >
                        <span className="action-text">Reset turn</span>
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : hideBotWaitMessageDuringTurnCycleLock ? null : (
                <p className="empty-note">
                  {showBotThinkingDuringIncomeChoiceLock || botThinking
                    ? 'Bot is thinking...'
                    : 'Waiting for bot...'}
                </p>
              )}
            </div>
          </section>

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
          <PlayerTokenRail player={botRailPlayer} side="bot" />
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
          <PlayerTokenRail player={humanRailPlayer} side="human" />
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

          <section className="panel">
            <h2>Deck State</h2>
            <div className="deck-piles" aria-label="Deck and discard piles">
              <div className="deck-pile">
                <div
                  className={`deck-pile-stack is-deck ${deckOverlayShiftClass}`}
                  title="Cards remaining"
                  aria-label="Cards remaining"
                >
                  {deckStackCount === 0 ? (
                    <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                  ) : (
                    Array.from({ length: deckStackCount }).map((_, index) => (
                      <div
                        key={`deck-back-${index}`}
                        className="deck-pile-card deck-pile-card-back deck-pile-stack-card"
                      />
                    ))
                  )}
                  {showSecondShuffleLabel ? (
                    <span
                      className="deck-pile-overlay-label"
                      aria-hidden="true"
                    >
                      2nd shuffle
                    </span>
                  ) : null}
                </div>
                <strong className="deck-pile-count">
                  {humanView.deck.drawCount}
                </strong>
              </div>
              <div className="deck-pile">
                <div className="player-score-wrap discard-pile-wrap">
                  <div
                    className={`deck-pile-stack is-discard${discardStackCardIds.length > 1 ? ' is-fanned' : ''}`}
                    title="Discard pile"
                    aria-label="Discard pile"
                    tabIndex={0}
                  >
                    {discardStackCardIds.length > 0 ? (
                      discardStackCardIds.map((cardId, index) => (
                        <div
                          key={`discard-${cardId}-${index}`}
                          className="deck-pile-card deck-pile-card-discard deck-pile-stack-card"
                        >
                          <img
                            className="deck-pile-image"
                            src={getCardImage(cardId)}
                            alt=""
                          />
                        </div>
                      ))
                    ) : (
                      <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                    )}
                  </div>
                  <section
                    className="player-score-popover discard-pile-popover"
                    role="tooltip"
                    aria-label="Discard pile details"
                  >
                    <p className="score-result">
                      Discarded Cards:{' '}
                      <strong>{discardCardDetails.length}</strong>
                    </p>
                    {discardCardDetails.length === 0 ? (
                      <p className="score-line">
                        <span>None yet</span>
                        <strong>-</strong>
                      </p>
                    ) : (
                      <ol className="discard-pile-list">
                        {discardCardDetails.map((card, index) => (
                          <li
                            key={`discard-detail-${card.id}-${index}`}
                            className="discard-pile-item"
                          >
                            <p className="discard-pile-card-row">
                              <strong className="discard-pile-card-rank">
                                {card.rank}
                              </strong>
                              <span className="discard-pile-card-suits">
                                {card.suitTokenText.length > 0 ? (
                                  renderSuitText(card.suitTokenText)
                                ) : (
                                  <strong>-</strong>
                                )}
                              </span>
                              <span className="discard-pile-card-name">
                                {card.name}
                              </span>
                            </p>
                          </li>
                        ))}
                      </ol>
                    )}
                  </section>
                </div>
                <strong className="deck-pile-count">
                  {visibleDiscardCards.length}
                </strong>
              </div>
            </div>
          </section>

          <section className="panel log-panel">
            <h2>Log</h2>
            {recentLog.length === 0 ? (
              <p className="empty-note">No actions yet.</p>
            ) : (
              <ol className="log-list">
                {recentLogGroups.map((group, groupIndex) => (
                  <li key={`${group.turn}-${groupIndex}`} className="log-turn-group">
                    <div className="log-turn-head">
                      <span className="log-turn">T{group.turn}</span>
                      <span className="log-player">
                        {group.player} ({group.player === HUMAN_PLAYER ? 'You' : 'Bot'})
                      </span>
                    </div>
                    <ol className="log-turn-entries">
                      {group.entries.map((entry, entryIndex) => (
                        (() => {
                          const seedValue = seedSummaryValue(entry.summary);
                          if (seedValue !== null) {
                            return (
                              <li
                                key={`${entry.turn}-${entry.phase}-${entry.summary}-${entryIndex}`}
                                className="log-turn-entry log-turn-entry-seed"
                              >
                                <div className="log-turn-head">
                                  <span className="log-turn">Seed</span>
                                  <span className="log-player">{seedValue}</span>
                                </div>
                              </li>
                            );
                          }
                          return (
                            <li
                              key={`${entry.turn}-${entry.phase}-${entry.summary}-${entryIndex}`}
                              className="log-turn-entry"
                            >
                              <span className="log-summary">
                                {entry.player !== group.player
                                  ? renderLogSummary(`[${entry.player}] ${entry.summary}`)
                                  : renderLogSummary(entry.summary)}
                              </span>
                            </li>
                          );
                        })()
                      ))}
                    </ol>
                  </li>
                ))}
              </ol>
            )}
          </section>

          <div className="corner-options-anchor">
            <button
              ref={optionsMenuButtonRef}
              type="button"
              className={`hamburger-button${optionsMenuOpen ? ' is-open' : ''}`}
              aria-label="Game options"
              aria-controls="brand-options-menu"
              aria-expanded={optionsMenuOpen}
              onClick={() => setOptionsMenuOpen((open) => !open)}
            >
              <span />
              <span />
              <span />
            </button>

            {optionsMenuOpen ? (
              <section
                id="brand-options-menu"
                ref={optionsMenuRef}
                className="brand-options-menu"
                aria-label="Game options"
              >
                <div className="brand-controls">
                  <input
                    id="seed-input"
                    aria-label="Seed"
                    className="seed-input"
                    value={seedInput}
                    onChange={(event) => setSeedInput(event.target.value)}
                  />
                  <button
                    className="reset-button"
                    type="button"
                    onClick={handleReset}
                  >
                    New Game
                  </button>
                </div>
                <div className="bot-profile-controls">
                  <label htmlFor="bot-profile-select">Bot Profile</label>
                  <select
                    id="bot-profile-select"
                    className="bot-profile-select"
                    value={botProfileId}
                    onChange={(event) =>
                      setBotProfileId(event.target.value as BotProfileId)
                    }
                  >
                    {BOT_PROFILES.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.label}
                      </option>
                    ))}
                  </select>
                  <p className="bot-profile-note">
                    {resolvedBotProfile.statusText}
                  </p>
                </div>
                <div className="bot-profile-controls animation-controls">
                  <label className="animation-toggle-row" htmlFor="animations-toggle">
                    <span>Animations</span>
                    <input
                      id="animations-toggle"
                      type="checkbox"
                      checked={animationsEnabled}
                      onChange={(event) =>
                        setAnimationsEnabled(event.target.checked)
                      }
                    />
                  </label>
                </div>
              </section>
            ) : null}
          </div>
        </aside>
      </main>

      {optionsMenuOpen ? (
        <div
          className="options-backdrop"
          aria-hidden="true"
          onClick={closeOptionsMenu}
        />
      ) : null}

      {!startupPreloadReady ? (
        <div
          className="app-bootstrap-shell startup-preload-overlay"
          role="presentation"
        >
          <section
            className="app-bootstrap-card startup-preload-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="startup-preload-title"
          >
            <h2 id="startup-preload-title" className="app-bootstrap-title">
              {startupPreloadError ? 'Loading Failed' : 'Preparing Game Assets'}
            </h2>
            <p className="app-bootstrap-copy startup-preload-message">
              {startupPreloadError
                ? `Could not preload assets: ${startupPreloadError}`
                : startupPreloadProgress.message}
            </p>
            <div
              className="app-bootstrap-bar startup-preload-progress"
              role="progressbar"
              aria-label="Startup asset preload progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={startupPreloadPercent}
            >
              <span
                className="startup-preload-progress-fill"
                style={{ width: `${startupPreloadPercent}%` }}
              />
            </div>
            <p className="startup-preload-progress-text">
              {startupPreloadCompleted} / {startupPreloadProgress.total}
            </p>
            {startupPreloadError ? (
              <button
                type="button"
                className="reset-button startup-preload-retry"
                onClick={retryStartupPreload}
              >
                Retry
              </button>
            ) : null}
          </section>
        </div>
      ) : null}

      {turnCycleOverlay ? (
        <div className="turn-cycle-overlay" aria-live="polite">
          <section className="turn-cycle-banner">
            {turnCycleOverlay.kind === 'tax' ? (
              <>
                <span className="turn-cycle-label">Taxes:</span>
                <TokenChip
                  suit={turnCycleOverlay.suit}
                  count={1}
                  compact
                  className="turn-cycle-banner-chip"
                />
              </>
            ) : (
              <>
                <span className="turn-cycle-label">Income:</span>
                <strong className="turn-cycle-income-rank">
                  {turnCycleOverlay.rank}
                </strong>
              </>
            )}
          </section>
        </div>
      ) : null}

      {resourceFlights.length > 0 ? (
        <div className="resource-flight-layer" aria-hidden="true">
          {resourceFlights.map((flight) => {
            const dx = flight.endX - flight.startX;
            const dy = flight.endY - flight.startY;
            return (
              <div
                key={flight.id}
                className={`resource-flight${flight.variant === 'tax-loss' ? ' is-tax-loss' : ' is-transfer'}`}
                style={
                  {
                    '--resource-flight-start-x': `${flight.startX}px`,
                    '--resource-flight-start-y': `${flight.startY}px`,
                    '--resource-flight-dx': `${dx}px`,
                    '--resource-flight-dy': `${dy}px`,
                    '--resource-flight-delay-ms': `${flight.delayMs}ms`,
                    '--resource-flight-duration-ms': `${flight.durationMs ?? RESOURCE_FLIGHT_DURATION_MS}ms`,
                  } as CSSProperties
                }
              >
                <TokenChip
                  suit={flight.suit}
                  count={1}
                  compact
                  className="resource-flight-chip"
                />
              </div>
            );
          })}
        </div>
      ) : null}

      {cardFlights.length > 0 ? (
        <div className="card-flight-layer" aria-hidden="true">
          {cardFlights.map((flight) => {
            const dx = flight.endX - flight.startX;
            const dy = flight.endY - flight.startY;
            const scaleX =
              flight.startWidth > 0 && Number.isFinite(flight.endWidth)
                ? flight.endWidth / flight.startWidth
                : 1;
            const scaleY =
              flight.startHeight > 0 && Number.isFinite(flight.endHeight)
                ? flight.endHeight / flight.startHeight
                : 1;
            return (
              <div
                key={flight.id}
                className={`card-flight${flight.variant === 'draw' ? ' is-draw' : ''}`}
                style={
                  {
                    '--card-flight-start-x': `${flight.startX}px`,
                    '--card-flight-start-y': `${flight.startY}px`,
                    '--card-flight-dx': `${dx}px`,
                    '--card-flight-dy': `${dy}px`,
                    '--card-flight-delay-ms': `${flight.delayMs}ms`,
                    '--card-flight-duration-ms': `${CARD_FLIGHT_DURATION_MS}ms`,
                    '--card-flight-scale-x': `${Number.isFinite(scaleX) ? scaleX : 1}`,
                    '--card-flight-scale-y': `${Number.isFinite(scaleY) ? scaleY : 1}`,
                    width: `${flight.startWidth}px`,
                    height: `${flight.startHeight}px`,
                  } as CSSProperties
                }
              >
                {flight.visual === 'face' && flight.cardId ? (
                  <CardTile
                    cardId={flight.cardId}
                    perspective={flight.perspective}
                    inDevelopment={flight.isDeed}
                    animateDeedProgress={animationsEnabled}
                  />
                ) : (
                  <CardTile hidden />
                )}
              </div>
            );
          })}
        </div>
      ) : null}

      {actionPicker && (
        <section
          ref={actionPopoverRef}
          className="panel trade-popover"
          role="dialog"
          aria-label="Choose follow-up action option"
          style={{
            top: `${actionPicker.top}px`,
            left: `${actionPicker.left}px`,
          }}
        >
          <h2>{renderSuitText(actionPickerTitle)}</h2>

          {actionPicker.kind === 'trade-combined' ? (
            <>
              {(() => {
                const tradeActions = humanActions.filter(
                  (action): action is Extract<GameAction, { type: 'trade' }> =>
                    action.type === 'trade'
                );
                const receiveOptions = [...new Set(tradeActions.map((action) => action.receive))];
                return (
                  <>
              <div className="composite-picker-group">
                <p className="composite-picker-label">Give x3</p>
                <div className="trade-choice-list">
                  {tradeSourceGroups.map((group) => (
                    <button
                      key={`trade-combined-source-${group.give}`}
                      type="button"
                      className={`trade-choice-button${actionPicker.selectedGive === group.give ? ' is-selected' : ''}`}
                      onClick={() => {
                        const nextGive = group.give;
                        const nextReceive = actionPicker.selectedReceive;
                        if (nextReceive) {
                          const selectedAction = tradeActions.find(
                            (action) =>
                              action.give === nextGive
                              && action.receive === nextReceive
                          );
                          if (selectedAction) {
                            handlePickerSelection(selectedAction);
                            return;
                          }
                        }
                        setActionPicker((current) => {
                          if (!current || current.kind !== 'trade-combined') {
                            return current;
                          }
                          return {
                            ...current,
                            selectedGive: nextGive,
                          };
                        });
                      }}
                    >
                      {renderSuitText(`${SUIT_TEXT_TOKEN[group.give]} x3`)}
                    </button>
                  ))}
                </div>
              </div>

              <div className="composite-picker-group">
                <p className="composite-picker-label">Receive x1</p>
                <div className="trade-choice-list">
                  {receiveOptions.map((receiveSuit) => (
                    <button
                      key={`trade-combined-receive-${receiveSuit}`}
                      type="button"
                      className={`trade-choice-button${actionPicker.selectedReceive === receiveSuit ? ' is-selected' : ''}`}
                      onClick={() => {
                        const nextReceive = receiveSuit;
                        const nextGive = actionPicker.selectedGive;
                        if (nextGive) {
                          const selectedAction = tradeActions.find(
                            (action) =>
                              action.give === nextGive
                              && action.receive === nextReceive
                          );
                          if (selectedAction) {
                            handlePickerSelection(selectedAction);
                            return;
                          }
                        }
                        setActionPicker((current) => {
                          if (!current || current.kind !== 'trade-combined') {
                            return current;
                          }
                          return {
                            ...current,
                            selectedReceive: nextReceive,
                          };
                        });
                      }}
                    >
                      {renderSuitText(`${SUIT_TEXT_TOKEN[receiveSuit]} x1`)}
                    </button>
                  ))}
                </div>
              </div>
                  </>
                );
              })()}
            </>
          ) : actionPicker.kind === 'develop-outright-combined' ? (
            <>
              {(() => {
                const outrightOptions = humanActions.filter(
                  (action): action is Extract<GameAction, { type: 'develop-outright' }> =>
                    action.type === 'develop-outright'
                    && action.cardId === actionPicker.cardId
                );
                const firstByDistrict = new Map<
                  string,
                  Extract<GameAction, { type: 'develop-outright' }>
                >();
                for (const option of outrightOptions) {
                  if (!firstByDistrict.has(option.districtId)) {
                    firstByDistrict.set(option.districtId, option);
                  }
                }
                const districtOptions = [...firstByDistrict.values()];
                const firstByPayment = new Map<
                  string,
                  Extract<GameAction, { type: 'develop-outright' }>
                >();
                for (const option of outrightOptions) {
                  const key = paymentSignature(option.payment);
                  if (!firstByPayment.has(key)) {
                    firstByPayment.set(key, option);
                  }
                }
                const paymentOptions = [...firstByPayment.entries()];
                return (
                  <>
                    <div className="composite-picker-group">
                      <p className="composite-picker-label">District</p>
                      <div className="trade-choice-list">
                        {districtOptions.map((option) => (
                          <button
                            key={`develop-outright-district-${option.districtId}`}
                            type="button"
                            className={`trade-choice-button${actionPicker.selectedDistrictId === option.districtId ? ' is-selected' : ''}`}
                            onClick={() => {
                              const nextDistrictId = option.districtId;
                              const nextPaymentKey = actionPicker.selectedPaymentKey;
                              if (nextPaymentKey) {
                                const selectedAction = outrightOptions.find(
                                  (candidate) =>
                                    candidate.districtId === nextDistrictId
                                    && paymentSignature(candidate.payment) === nextPaymentKey
                                );
                                if (selectedAction) {
                                  handlePickerSelection(selectedAction);
                                  return;
                                }
                              }
                              setActionPicker((current) => {
                                if (
                                  !current
                                  || current.kind !== 'develop-outright-combined'
                                ) {
                                  return current;
                                }
                                return {
                                  ...current,
                                  selectedDistrictId: nextDistrictId,
                                };
                              });
                            }}
                          >
                            {option.districtId}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="composite-picker-group">
                      <p className="composite-picker-label">Payment</p>
                      <div className="trade-choice-list single-column">
                        {paymentOptions.map(([paymentKey, option]) => (
                          <button
                            key={`develop-outright-payment-${paymentKey}`}
                            type="button"
                            className={`trade-choice-button${actionPicker.selectedPaymentKey === paymentKey ? ' is-selected' : ''}`}
                            onClick={() => {
                              const nextPaymentKey = paymentKey;
                              const nextDistrictId = actionPicker.selectedDistrictId;
                              if (nextDistrictId) {
                                const selectedAction = outrightOptions.find(
                                  (candidate) =>
                                    candidate.districtId === nextDistrictId
                                    && paymentSignature(candidate.payment) === nextPaymentKey
                                );
                                if (selectedAction) {
                                  handlePickerSelection(selectedAction);
                                  return;
                                }
                              }
                              setActionPicker((current) => {
                                if (
                                  !current
                                  || current.kind !== 'develop-outright-combined'
                                ) {
                                  return current;
                                }
                                return {
                                  ...current,
                                  selectedPaymentKey: nextPaymentKey,
                                };
                              });
                            }}
                          >
                            {renderSuitText(
                              formatTokens(option.payment, SUIT_TEXT_TOKEN)
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  </>
                );
              })()}
            </>
          ) : actionPickerOptions.length === 0 ? (
            <p className="empty-note">No options available.</p>
          ) : (
            <div className="trade-choice-list">
              {actionPickerOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className="trade-choice-button"
                  onClick={() => handlePickerSelection(option.action)}
                >
                  {renderSuitText(option.label)}
                </button>
              ))}
            </div>
          )}

          <button
            type="button"
            className="trade-cancel-button"
            onClick={closeActionPicker}
          >
            Cancel
          </button>
        </section>
      )}
    </div>
  );
}

function actionCategoryForItem(item: HumanActionListItem): string {
  switch (item.kind) {
    case 'trade-group':
      return 'trade';
    case 'buy-deed-group':
      return 'buy-deed';
    case 'develop-deed-group':
      return 'develop-deed';
    case 'develop-outright-group':
      return 'develop-outright';
    case 'action':
      return item.action.type;
  }
}

function actionCategoryLabel(category: string): string {
  switch (category) {
    case 'trade':
      return 'Trade';
    case 'buy-deed':
      return 'Buy Deed';
    case 'develop-deed':
      return 'Develop Deed';
    case 'develop-outright':
      return 'Develop Outright';
    case 'sell-card':
      return 'Sell Card';
    case 'choose-income-suit':
      return 'Choose Income';
    case 'end-turn':
      return 'End Turn';
    default:
      return category;
  }
}

function winnerLabel(winner: FinalScore['winner']): string {
  if (winner === 'Draw') {
    return 'Tie';
  }
  return winner === HUMAN_PLAYER ? 'You' : 'Bot';
}

function deciderLabel(decidedBy: FinalScore['decidedBy']): string {
  switch (decidedBy) {
    case 'districts':
      return 'Districts';
    case 'rank-total':
      return 'Total Properties';
    case 'resources':
      return 'Resources';
    case 'draw':
      return 'Tie';
  }
}

function formatDistrictList(districtIds: readonly string[]): string {
  if (districtIds.length === 0) {
    return 'None';
  }
  return districtIds.join(', ');
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}

function renderSuitText(text: string): ReactNode {
  if (!text) {
    return text;
  }

  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(SUIT_TOKEN_REGEX)) {
    const index = match.index ?? 0;
    const token = match[0];
    const suit = SUIT_TOKEN_TO_SUIT[token];

    if (index > cursor) {
      nodes.push(text.slice(cursor, index));
    }

    if (suit) {
      nodes.push(
        <TokenChip
          key={`suit-${index}-${suit}`}
          suit={suit}
          count={1}
          compact
          className="inline-token-chip"
        />
      );
    } else {
      nodes.push(token);
    }

    cursor = index + token.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? <>{nodes}</> : text;
}

function groupLogEntriesByTurn(
  entries: ReadonlyArray<GameLogEntry>
): ReadonlyArray<{
  turn: number;
  player: PlayerId;
  entries: ReadonlyArray<GameLogEntry>;
}> {
  const groups: Array<{
    turn: number;
    player: PlayerId;
    entries: GameLogEntry[];
  }> = [];

  for (const entry of entries) {
    const current = groups[groups.length - 1];
    if (!current || current.turn !== entry.turn) {
      groups.push({
        turn: entry.turn,
        player: entry.player,
        entries: [entry],
      });
      continue;
    }
    current.entries.push(entry);
  }

  return groups;
}

function sentenceCaseSummary(summary: string): string {
  if (!summary) {
    return summary;
  }
  const prefixMatch = summary.match(/^(\[[^\]]+\]\s*)/);
  const prefix = prefixMatch?.[1] ?? '';
  const rest = summary.slice(prefix.length);
  if (rest.length === 0) {
    return summary;
  }
  return `${prefix}${rest.slice(0, 1).toUpperCase()}${rest.slice(1)}`;
}

function formatLogSummary(summary: string): string {
  let next = summary;

  next = next.replace(
    INCOME_CHOICE_PATTERN,
    (_match, rawCardId: string, rawSuit: string) => {
      const cardLabel = formatCardIdForLog(rawCardId);
      const suitCode = suitNameToCode(rawSuit);
      return `income choice ${cardLabel}:${suitCode ?? rawSuit}`;
    }
  );

  next = next.replace(
    CARD_ACTION_PATTERN,
    (_match, verb: string, rawCardId: string) =>
      `${verb} ${formatCardIdForLog(rawCardId)}`
  );

  next = next.replace(SUIT_NAME_PATTERN, (_match, suitName: string) => {
    const suit = suitName as Suit;
    return SUIT_LOG_CODE[suit] ?? suitName;
  });

  return sentenceCaseSummary(next);
}

function renderLogSummary(summary: string): ReactNode {
  const text = formatLogSummary(summary);
  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(SUIT_CODE_PATTERN)) {
    const index = match.index ?? 0;
    if (index > cursor) {
      nodes.push(text.slice(cursor, index));
    }

    const suitCode = match[0] as 'mo' | 'su' | 'wa' | 'le' | 'wy' | 'kn';
    const suit = suitCodeToSuit(suitCode);
    nodes.push(
      <span
        key={`log-suit-${index}-${suitCode}`}
        className="log-suit-code"
        style={suit ? { color: SUIT_TOKEN_BG[suit] } : undefined}
      >
        {suitCode}
      </span>
    );
    cursor = index + suitCode.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? <>{nodes}</> : text;
}

function formatCardIdForLog(rawCardId: string): string {
  const card = CARD_BY_ID[rawCardId as CardId];
  if (!card) {
    return rawCardId;
  }
  if (card.kind !== 'Property' && card.kind !== 'Crown') {
    return rawCardId;
  }
  const suitCodes = card.suits.map((suit) => SUIT_LOG_CODE[suit]).join(' ');
  return `${card.rank} ${suitCodes} (${rawCardId})`;
}

function suitNameToCode(value: string): string | null {
  if (
    value !== 'Moons' &&
    value !== 'Suns' &&
    value !== 'Waves' &&
    value !== 'Leaves' &&
    value !== 'Wyrms' &&
    value !== 'Knots'
  ) {
    return null;
  }
  return SUIT_LOG_CODE[value];
}

function suitCodeToSuit(
  value: 'mo' | 'su' | 'wa' | 'le' | 'wy' | 'kn'
): Suit | null {
  switch (value) {
    case 'mo':
      return 'Moons';
    case 'su':
      return 'Suns';
    case 'wa':
      return 'Waves';
    case 'le':
      return 'Leaves';
    case 'wy':
      return 'Wyrms';
    case 'kn':
      return 'Knots';
    default:
      return null;
  }
}

function seedSummaryValue(summary: string): string | null {
  const prefix = 'Seed ';
  if (!summary.startsWith(prefix)) {
    return null;
  }
  return summary.slice(prefix.length);
}

function toPickerQuery(
  picker: Exclude<
    ActionPickerState,
    | { kind: 'trade-combined' }
    | { kind: 'develop-outright-combined' }
  >
): ActionPickerQuery {
  if (picker.kind === 'trade') {
    return { kind: 'trade', give: picker.give };
  }
  if (picker.kind === 'deed-payment') {
    return {
      kind: 'deed-payment',
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  if (picker.kind === 'develop-outright-district') {
    return {
      kind: 'develop-outright-district',
      cardId: picker.cardId,
    };
  }
  if (picker.kind === 'develop-outright-payment') {
    return {
      kind: 'develop-outright-payment',
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  return {
    kind: 'district',
    actionType: picker.actionType,
    cardId: picker.cardId,
  };
}
