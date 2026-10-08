import {
  IncomeTokenSourceKind,
  GamePresentationEventType,
} from '../runtime/values';
import type { CardId } from '../../engine/cards';
import { PlayerId, type Suit } from '../../engine/types';
import type { CardPerspective } from '../components/CardTile';
import {
  layoutDeedTokensBySide,
  resetDeedTokenLayout,
  type DeedTokenSide,
} from '../components/deedTokenLayout';
import { tokenEntries } from '../components/TokenComponents';
import type { TurnCycleIncomeToken } from '../turnCycleEvents';
import {
  browserAnimationDomTargets,
  deedTokenCenterInRail,
  parsePixelValue,
  type AnimationDomTargets,
  type Point,
} from './domTargets';
import {
  PAYMENT_FLIGHT_DURATION_MS,
  PAYMENT_FLIGHT_STAGGER_MS,
  RESOURCE_FLIGHT_STAGGER_MS,
  SELL_FLIGHT_DURATION_MS,
  SELL_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from './timing';
import type { GamePresentationEvent } from '../runtime/types';
import type {
  CardFlight,
  PendingResourceFlight,
  ResourceFlight,
} from './types';

const BOT_PLAYER: PlayerId = PlayerId.PlayerB;
const DEFAULT_TOKEN_CHIP_SIZE_PX = 22;
const DEFAULT_TOKEN_RAIL_GAP_PX = 2.56;

export type IncomeFlightToken = {
  playerId: PlayerId;
  suit: Suit;
  source:
    | TurnCycleIncomeToken['source']
    | {
        kind: typeof IncomeTokenSourceKind.IncomeChoice;
        cardId: CardId;
        districtId: string;
      };
};

export type IncomeFlightTiming = {
  durationMs: number;
  staggerMs: number;
};

export type PaymentFlightTiming = {
  durationMs: number;
  staggerMs: number;
};

export type SellFlightTiming = {
  durationMs: number;
  staggerMs: number;
};

const DEFAULT_PAYMENT_FLIGHT_TIMING: PaymentFlightTiming = {
  durationMs: PAYMENT_FLIGHT_DURATION_MS,
  staggerMs: PAYMENT_FLIGHT_STAGGER_MS,
};

const DEFAULT_SELL_FLIGHT_TIMING: SellFlightTiming = {
  durationMs: SELL_FLIGHT_DURATION_MS,
  staggerMs: SELL_FLIGHT_STAGGER_MS,
};

const DEFAULT_INCOME_FLIGHT_TIMING: IncomeFlightTiming = {
  durationMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  staggerMs: TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
};

export function buildTaxLossFlightsFromDom(
  targets: ReadonlyArray<{
    playerId: PlayerId;
    suit: Suit;
  }>,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): ResourceFlight[] {
  if (!domTargets.isAvailable() || targets.length === 0) {
    return [];
  }

  const viewportCenterY = domTargets.viewportCenterY();
  const flights: ResourceFlight[] = [];
  for (const [index, target] of targets.entries()) {
    const sourceElement = domTargets.resourceToken(
      target.playerId,
      target.suit
    );
    if (!sourceElement) {
      continue;
    }
    const source = domTargets.tokenVisualCenter(sourceElement);
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

export function buildIncomeFlightsFromDom(
  tokens: ReadonlyArray<IncomeFlightToken>,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets,
  timing: IncomeFlightTiming = DEFAULT_INCOME_FLIGHT_TIMING
): ResourceFlight[] {
  if (!domTargets.isAvailable() || tokens.length === 0) {
    return [];
  }

  const flights: ResourceFlight[] = [];
  for (const [index, token] of tokens.entries()) {
    const sourceElement =
      token.source.kind === IncomeTokenSourceKind.Crown
        ? domTargets.crownToken(token.playerId, token.suit)
        : domTargets.districtCard(
            token.playerId,
            token.source.districtId,
            token.source.cardId
          );
    const targetElement = domTargets.resourceToken(token.playerId, token.suit);
    if (!sourceElement || !targetElement) {
      continue;
    }

    const source =
      token.source.kind === IncomeTokenSourceKind.Crown
        ? domTargets.tokenVisualCenter(sourceElement)
        : domTargets.elementCenter(sourceElement);
    const target = domTargets.tokenVisualCenter(targetElement);
    flights.push({
      id: makeFlightId(),
      suit: token.suit,
      startX: source.x,
      startY: source.y,
      endX: target.x,
      endY: target.y,
      delayMs: index * timing.staggerMs,
      durationMs: timing.durationMs,
      variant: 'transfer',
    });
  }

  return flights;
}

export function buildSellTokenFlightsFromDom(
  gains: readonly Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.SellResourceGained }
  >[],
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets,
  timing: SellFlightTiming = DEFAULT_SELL_FLIGHT_TIMING
): ResourceFlight[] {
  if (!domTargets.isAvailable() || gains.length === 0) {
    return [];
  }

  const flights: ResourceFlight[] = [];
  for (const [index, gain] of gains.entries()) {
    const sourceElement = domTargets.handSource(gain.playerId, gain.cardId);
    const targetElement = domTargets.resourceToken(gain.playerId, gain.suit);
    if (!sourceElement || !targetElement) {
      continue;
    }
    const source = domTargets.elementCenter(sourceElement);
    const target = domTargets.tokenVisualCenter(targetElement);
    flights.push({
      id: makeFlightId(),
      suit: gain.suit,
      startX: source.x,
      startY: source.y,
      endX: target.x,
      endY: target.y,
      delayMs: index * timing.staggerMs,
      durationMs: timing.durationMs,
      variant: 'transfer',
    });
  }

  return flights;
}

export function buildPaymentFlightsFromDom(
  event: Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.ResourcePaymentStarted }
  >,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets,
  timing: PaymentFlightTiming = DEFAULT_PAYMENT_FLIGHT_TIMING
): ResourceFlight[] {
  const suitsToAnimate: Suit[] = [];
  for (const entry of tokenEntries(event.payment)) {
    for (let count = 0; count < entry.count; count += 1) {
      suitsToAnimate.push(entry.suit);
    }
  }

  return buildRemovalFlightsFromDom(
    event.playerId,
    suitsToAnimate,
    makeFlightId,
    domTargets,
    timing
  );
}

export function buildTradeFlightsFromDom(
  event: Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.TradeResourcesApplied }
  >,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets,
  timing: PaymentFlightTiming = DEFAULT_PAYMENT_FLIGHT_TIMING
): ResourceFlight[] {
  if (!domTargets.isAvailable()) return [];
  const sourceElement = domTargets.resourceToken(event.playerId, event.give);
  const targetElement = domTargets.resourceToken(event.playerId, event.receive);
  if (!sourceElement || !targetElement) return [];
  const source = domTargets.tokenVisualCenter(sourceElement);
  const target = domTargets.tokenVisualCenter(targetElement);
  return Array.from({ length: event.giveCount }, (_, index) => ({
    id: makeFlightId(),
    suit: event.give,
    startX: source.x,
    startY: source.y,
    endX: target.x,
    endY: target.y,
    delayMs: index * timing.staggerMs,
    durationMs: timing.durationMs,
    variant: 'transfer' as const,
  }));
}

function buildRemovalFlightsFromDom(
  playerId: PlayerId,
  suits: readonly Suit[],
  makeFlightId: () => string,
  domTargets: AnimationDomTargets,
  timing: PaymentFlightTiming
): ResourceFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const viewportCenterY = domTargets.viewportCenterY();
  const flights: ResourceFlight[] = [];
  for (const [index, suit] of suits.entries()) {
    const sourceElement =
      domTargets.resourceTokenForDeedTransfer(playerId, suit) ??
      domTargets.resourceToken(playerId, suit);
    if (!sourceElement) {
      continue;
    }
    const source = domTargets.tokenVisualCenter(sourceElement);
    flights.push({
      id: makeFlightId(),
      suit,
      startX: source.x,
      startY: source.y,
      endX: source.x,
      endY: viewportCenterY,
      delayMs: index * timing.staggerMs,
      durationMs: timing.durationMs,
      variant: 'payment',
    });
  }

  return flights;
}

export function buildDeedResourceFlightsFromDom(
  deedTokens: readonly Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.DeedTokenPaid }
  >[],
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): PendingResourceFlight[] {
  if (deedTokens.length === 0 || !domTargets.isAvailable()) {
    return [];
  }

  const firstToken = deedTokens[0];
  const suitsToAnimate = deedTokens.map((token) => token.suit);

  const cardElement = domTargets.developingCard(firstToken.cardId);
  if (!cardElement) {
    return [];
  }

  const perspective: 'human' | 'bot' = cardElement.classList.contains(
    'perspective-bot'
  )
    ? 'bot'
    : 'human';
  const deedTokenEntries = tokenEntries(firstToken.previousTokens);
  if (deedTokenEntries.length === 0) {
    resetDeedTokenLayout(firstToken.cardId, perspective);
  }
  const nextTokenEntries = tokenEntries(firstToken.nextTokens);
  const nextBySide = layoutDeedTokensBySide(
    firstToken.cardId,
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

  const sourceBySuit = new Map<Suit, Point>();
  const tokenSizeBySuit = new Map<Suit, number>();
  for (const suit of new Set(suitsToAnimate)) {
    const sourceElement = domTargets.resourceTokenForDeedTransfer(
      firstToken.playerId,
      suit
    );
    if (!sourceElement) {
      continue;
    }
    sourceBySuit.set(suit, domTargets.tokenVisualCenter(sourceElement));
    tokenSizeBySuit.set(
      suit,
      sourceElement.getBoundingClientRect().width || DEFAULT_TOKEN_CHIP_SIZE_PX
    );
  }

  const flights: PendingResourceFlight[] = [];
  for (const [index, suit] of suitsToAnimate.entries()) {
    const source = sourceBySuit.get(suit);
    const target = targetBySuit.get(suit);
    if (!source || !target) {
      continue;
    }

    const existingTargetChip = domTargets.deedTokenOnSide(
      cardElement,
      target.side,
      suit
    );
    let targetPoint: Point | null = null;
    if (existingTargetChip) {
      targetPoint = domTargets.elementCenter(existingTargetChip);
    } else {
      const targetRail = domTargets.deedTokenRail(cardElement, target.side);
      if (targetRail) {
        const railStyle = domTargets.computedStyle(targetRail);
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

function layoutSize(
  element: HTMLElement,
  rect: DOMRect
): { width: number; height: number } {
  // A rotated card reports a larger axis-aligned rect; the untransformed
  // layout box keeps flights sized to the card itself.
  return {
    width: element.offsetWidth > 0 ? element.offsetWidth : rect.width,
    height: element.offsetHeight > 0 ? element.offsetHeight : rect.height,
  };
}

export function createCardFlight(
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
    variant?: 'play' | 'draw';
    /**
     * Render the flight at the destination's box and animate translate only, so
     * the card travels at a constant apparent size instead of shrinking. This
     * is the shared convention for card flights: the transient element is laid
     * out at its landing size (like a container transform), and scale is
     * reserved for the source-to-destination fit. Without it, a destination
     * with a real box makes the card shrink over the flight while a zero-size
     * destination leaves it at source size — two different motions for the same
     * nominal timing.
     */
    renderAtDestination?: boolean;
    /**
     * The landing destination is the discard pile, which renders its cards with
     * the deck-pile card scope. CardFlightLayer remaps the flight card's scope
     * to match the landed discard card instead of the board card.
     */
    discardDestination?: boolean;
  },
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const targetRect = targetElement.getBoundingClientRect();
  const sourceSize = layoutSize(sourceElement, sourceRect);
  const sourceCenter = domTargets.elementCenter(sourceElement);
  const targetCenter = domTargets.elementCenter(targetElement);
  // Destination-layout mode needs a real destination box; otherwise fall back
  // to source sizing so the flight never collapses to nothing.
  const destinationWidth = targetRect.width > 0 ? targetRect.width : undefined;
  const destinationHeight =
    targetRect.height > 0 ? targetRect.height : undefined;
  const renderWidth = options?.renderAtDestination
    ? (destinationWidth ?? sourceSize.width)
    : undefined;
  const renderHeight = options?.renderAtDestination
    ? (destinationHeight ?? sourceSize.height)
    : undefined;
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
    startWidth: sourceSize.width,
    startHeight: sourceSize.height,
    endWidth: destinationWidth ?? sourceSize.width,
    endHeight: destinationHeight ?? sourceSize.height,
    renderWidth,
    renderHeight,
    discardDestination: options?.discardDestination,
    delayMs: options?.delayMs ?? 0,
    durationMs: options?.durationMs,
  };
}

export function createCardFlightToPoint(
  makeFlightId: () => string,
  sourceElement: HTMLElement,
  target: Point,
  visual: 'face' | 'back',
  options?: {
    cardId?: CardId;
    isDeed?: boolean;
    perspective?: CardPerspective;
    delayMs?: number;
    durationMs?: number;
    endWidth?: number;
    endHeight?: number;
    endImageAreaWidth?: number;
    endImageAreaHeight?: number;
    variant?: 'play' | 'draw';
    stacked?: boolean;
  },
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const sourceSize = layoutSize(sourceElement, sourceRect);
  const sourceCenter = domTargets.elementCenter(sourceElement);
  return {
    id: makeFlightId(),
    variant: options?.variant ?? 'play',
    visual,
    cardId: options?.cardId,
    isDeed: options?.isDeed ?? false,
    perspective: options?.perspective ?? 'human',
    stacked: options?.stacked ?? false,
    startX: sourceCenter.x,
    startY: sourceCenter.y,
    endX: target.x,
    endY: target.y,
    startWidth: sourceSize.width,
    startHeight: sourceSize.height,
    endWidth: options?.endWidth ?? sourceSize.width,
    endHeight: options?.endHeight ?? sourceSize.height,
    renderWidth: options?.endWidth ?? sourceSize.width,
    renderHeight: options?.endHeight ?? sourceSize.height,
    endImageAreaWidth: options?.endImageAreaWidth,
    endImageAreaHeight: options?.endImageAreaHeight,
    delayMs: options?.delayMs ?? 0,
    durationMs: options?.durationMs,
  };
}

export function buildSoldCardFlightFromDom(
  playerId: PlayerId,
  cardId: CardId,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const sourceElement = domTargets.handSource(playerId, cardId);
  const targetElement = domTargets.discardTarget();
  if (!sourceElement || !targetElement) {
    return [];
  }

  const sourceSlotKind = sourceElement.getAttribute('data-hand-slot-kind');
  const visual: 'face' | 'back' =
    sourceSlotKind === 'occupied' ? 'face' : 'back';
  const perspective: CardPerspective = sourceElement.classList.contains(
    'perspective-bot'
  )
    ? 'bot'
    : 'human';

  return [
    createCardFlight(
      makeFlightId,
      sourceElement,
      targetElement,
      visual,
      {
        cardId: visual === 'face' ? cardId : undefined,
        isDeed: false,
        perspective,
        renderAtDestination: true,
        discardDestination: true,
      },
      domTargets
    ),
  ];
}

export function buildCardToDistrictFlightFromDom(
  event: Extract<
    GamePresentationEvent,
    { type: typeof GamePresentationEventType.CardPlayedToDistrict }
  >,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const sourceElement = domTargets.handSource(event.playerId, event.cardId);
  if (!sourceElement) {
    return [];
  }

  const laneElement = domTargets.lane(event.playerId, event.districtId);
  const targetCardMetrics = laneElement
    ? domTargets.laneCardMetrics(laneElement, sourceElement)
    : null;
  const stacked = laneElement
    ? domTargets.laneCardCount(laneElement) > 0
    : false;
  const districtColumn = domTargets.districtColumn(event.districtId);
  const fallbackTargetElement =
    (laneElement ? domTargets.laneFrame(laneElement) : null) ??
    laneElement ??
    districtColumn;
  const targetCenter =
    (laneElement
      ? domTargets.laneTargetCenter(
          laneElement,
          targetCardMetrics?.height ??
            sourceElement.getBoundingClientRect().height
        )
      : null) ??
    (fallbackTargetElement
      ? domTargets.elementCenter(fallbackTargetElement)
      : null);
  if (!targetCenter) {
    return [];
  }

  const perspective: CardPerspective = laneElement
    ? laneElement.classList.contains('is-bot')
      ? 'bot'
      : 'human'
    : event.playerId === BOT_PLAYER
      ? 'bot'
      : 'human';

  return [
    createCardFlightToPoint(
      makeFlightId,
      sourceElement,
      targetCenter,
      'face',
      {
        cardId: event.cardId,
        isDeed: event.placement === 'deed',
        perspective,
        stacked,
        endWidth: targetCardMetrics?.width,
        endHeight: targetCardMetrics?.height,
        endImageAreaWidth:
          targetCardMetrics && targetCardMetrics.imageAreaWidth > 0
            ? targetCardMetrics.imageAreaWidth
            : undefined,
        endImageAreaHeight:
          targetCardMetrics && targetCardMetrics.imageAreaHeight > 0
            ? targetCardMetrics.imageAreaHeight
            : undefined,
      },
      domTargets
    ),
  ];
}

export function buildDrawCardFlightFromDom(
  playerId: PlayerId,
  cardId: CardId,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const sourceElement = domTargets.deckSource();
  const targetElement = domTargets.handDrawTarget(playerId);
  if (!sourceElement || !targetElement) {
    return [];
  }

  return [
    createCardFlight(
      makeFlightId,
      sourceElement,
      targetElement,
      'back',
      {
        cardId,
        variant: 'draw',
        renderAtDestination: true,
      },
      domTargets
    ),
  ];
}
