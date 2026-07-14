import type { CardId } from '../../engine/cards';
import type { PlayerId, Suit } from '../../engine/types';
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
  RESOURCE_FLIGHT_DURATION_MS,
  RESOURCE_FLIGHT_STAGGER_MS,
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

const BOT_PLAYER: PlayerId = 'PlayerB';
const DEFAULT_TOKEN_CHIP_SIZE_PX = 22;
const DEFAULT_TOKEN_RAIL_GAP_PX = 2.56;

export type IncomeFlightToken = {
  playerId: PlayerId;
  suit: Suit;
  source:
    | TurnCycleIncomeToken['source']
    | {
        kind: 'income-choice';
        cardId: CardId;
        districtId: string;
      };
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
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): ResourceFlight[] {
  if (!domTargets.isAvailable() || tokens.length === 0) {
    return [];
  }

  const flights: ResourceFlight[] = [];
  for (const [index, token] of tokens.entries()) {
    const sourceElement =
      token.source.kind === 'crown'
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
      token.source.kind === 'crown'
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
      delayMs: index * TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
      durationMs: TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
      variant: 'transfer',
    });
  }

  return flights;
}

export function buildPaymentFlightsFromDom(
  event: Extract<GamePresentationEvent, { type: 'resource-payment-started' }>,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): ResourceFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const viewportCenterY = domTargets.viewportCenterY();
  const flights: ResourceFlight[] = [];
  const suitsToAnimate: Suit[] = [];
  for (const entry of tokenEntries(event.payment)) {
    for (let count = 0; count < entry.count; count += 1) {
      suitsToAnimate.push(entry.suit);
    }
  }

  for (const [index, suit] of suitsToAnimate.entries()) {
    const sourceElement =
      domTargets.resourceTokenForDeedTransfer(event.playerId, suit) ??
      domTargets.resourceToken(event.playerId, suit);
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
      delayMs: index * RESOURCE_FLIGHT_STAGGER_MS,
      durationMs: RESOURCE_FLIGHT_DURATION_MS,
      variant: 'payment',
    });
  }

  return flights;
}

export function buildDeedResourceFlightsFromDom(
  deedTokens: readonly Extract<
    GamePresentationEvent,
    { type: 'deed-token-paid' }
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
  },
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const targetRect = targetElement.getBoundingClientRect();
  const sourceCenter = domTargets.elementCenter(sourceElement);
  const targetCenter = domTargets.elementCenter(targetElement);
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
    variant?: 'play' | 'draw';
  },
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight {
  const sourceRect = sourceElement.getBoundingClientRect();
  const sourceCenter = domTargets.elementCenter(sourceElement);
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
      },
      domTargets
    ),
  ];
}

export function buildCardToDistrictFlightFromDom(
  event: Extract<GamePresentationEvent, { type: 'card-played-to-district' }>,
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
  const targetCardSize = laneElement
    ? domTargets.laneCardSize(laneElement, sourceElement)
    : null;
  const districtColumn = domTargets.districtColumn(event.districtId);
  const fallbackTargetElement =
    (laneElement ? domTargets.laneFrame(laneElement) : null) ??
    laneElement ??
    districtColumn;
  const targetCenter =
    (laneElement
      ? domTargets.laneTargetCenter(
          laneElement,
          targetCardSize?.height ?? sourceElement.getBoundingClientRect().height
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
        endWidth: targetCardSize?.width,
        endHeight: targetCardSize?.height,
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
      },
      domTargets
    ),
  ];
}
