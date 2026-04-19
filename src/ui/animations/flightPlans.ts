import type { CardId } from '../../engine/cards';
import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
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
  CARD_DRAW_FLIGHT_DELAY_MS,
  RESOURCE_FLIGHT_STAGGER_MS,
  TURN_CYCLE_INCOME_FLIGHT_DURATION_MS,
  TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS,
  TURN_CYCLE_TAX_FLIGHT_DURATION_MS,
  TURN_CYCLE_TAX_FLIGHT_STAGGER_MS,
} from './timing';
import type {
  CardFlight,
  PendingResourceFlight,
  ResourceFlight,
} from './types';

const BOT_PLAYER: PlayerId = 'PlayerB';
const DEFAULT_TOKEN_CHIP_SIZE_PX = 22;
const DEFAULT_TOKEN_RAIL_GAP_PX = 2.56;

export function collectTerminalCleanupFlights(): {
  resourceFlights: ReadonlyArray<ResourceFlight>;
  cardFlights: ReadonlyArray<CardFlight>;
} | null {
  // Keep terminal board state visible for post-game review.
  return null;
}

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
  tokens: ReadonlyArray<TurnCycleIncomeToken>,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): ResourceFlight[] {
  if (!domTargets.isAvailable() || tokens.length === 0) {
    return [];
  }

  const flights: ResourceFlight[] = [];
  for (const [index, token] of tokens.entries()) {
    const sourceElement =
      token.source.kind === 'district-card'
        ? domTargets.districtCard(
            token.playerId,
            token.source.districtId,
            token.source.cardId
          )
        : domTargets.crownToken(token.playerId, token.suit);
    const targetElement = domTargets.resourceToken(token.playerId, token.suit);
    if (!sourceElement || !targetElement) {
      continue;
    }

    const source =
      token.source.kind === 'district-card'
        ? domTargets.elementCenter(sourceElement)
        : domTargets.tokenVisualCenter(sourceElement);
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

export function collectDeedResourceFlights(
  state: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): PendingResourceFlight[] {
  if (action.type !== 'develop-deed' || !domTargets.isAvailable()) {
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

  const cardElement = domTargets.developingCard(action.cardId);
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

  const sourceBySuit = new Map<Suit, Point>();
  const tokenSizeBySuit = new Map<Suit, number>();
  for (const suit of new Set(suitsToAnimate)) {
    const sourceElement = domTargets.resourceTokenForDeedTransfer(
      actingPlayerId,
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

export function collectIncomeChoiceResourceFlights(
  action: GameAction,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): PendingResourceFlight[] {
  if (action.type !== 'choose-income-suit' || !domTargets.isAvailable()) {
    return [];
  }

  const sourceElement = domTargets.districtCard(
    action.playerId,
    action.districtId,
    action.cardId
  );
  const targetElement = domTargets.resourceToken(action.playerId, action.suit);
  if (!sourceElement || !targetElement) {
    return [];
  }

  const source = domTargets.elementCenter(sourceElement);
  const target = domTargets.tokenVisualCenter(targetElement);
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
    variant?: 'play' | 'draw' | 'terminal-clear';
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
    variant?: 'play' | 'draw' | 'terminal-clear';
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

export function collectCardPlayFlights(
  state: GameState,
  nextState: GameState,
  action: GameAction,
  actingPlayerId: PlayerId,
  makeFlightId: () => string,
  domTargets: AnimationDomTargets = browserAnimationDomTargets
): CardFlight[] {
  if (!domTargets.isAvailable()) {
    return [];
  }

  const flights: CardFlight[] = [];

  if (action.type === 'sell-card') {
    const sourceElement = domTargets.handSource(actingPlayerId, action.cardId);
    const targetElement = domTargets.discardTarget();
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
        createCardFlight(
          makeFlightId,
          sourceElement,
          targetElement,
          visual,
          {
            cardId: visual === 'face' ? action.cardId : undefined,
            isDeed: false,
            perspective,
          },
          domTargets
        )
      );
    }
  }

  if (isLaneCardPlayAction(action)) {
    const sourceElement = domTargets.handSource(actingPlayerId, action.cardId);
    if (sourceElement) {
      const laneElement = domTargets.lane(actingPlayerId, action.districtId);
      const districtColumn = domTargets.districtColumn(action.districtId);
      const fallbackTargetElement =
        (laneElement ? domTargets.laneFrame(laneElement) : null) ??
        laneElement ??
        districtColumn;
      const targetCenter =
        (laneElement
          ? domTargets.laneTargetCenter(
              laneElement,
              sourceElement.getBoundingClientRect().height
            )
          : null) ??
        (fallbackTargetElement
          ? domTargets.elementCenter(fallbackTargetElement)
          : null);
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
            },
            domTargets
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
      const sourceElement = domTargets.deckSource();
      const targetElement = domTargets.handDrawTarget(actingPlayerId);
      if (sourceElement && targetElement) {
        flights.push(
          createCardFlight(
            makeFlightId,
            sourceElement,
            targetElement,
            'back',
            {
              delayMs: flights.length > 0 ? CARD_DRAW_FLIGHT_DELAY_MS : 0,
              variant: 'draw',
            },
            domTargets
          )
        );
      }
    }
  }

  return flights;
}
