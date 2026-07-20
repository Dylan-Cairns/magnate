import type { CardId } from '../../engine/cards';
import type { PlayerId, Suit } from '../../engine/types';
import type { DeedTokenSide } from '../components/deedTokenLayout';

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface AnimationDomEnvironment {
  isAvailable(): boolean;
  querySelector<T extends Element = HTMLElement>(selectors: string): T | null;
  querySelectorAll<T extends Element = HTMLElement>(
    selectors: string
  ): readonly T[];
  createElement(tagName: string): HTMLElement;
  getComputedStyle(element: Element): CSSStyleDeclaration;
  viewportHeight(): number;
}

export interface AnimationDomTargets {
  isAvailable(): boolean;
  viewportCenterY(): number;
  elementCenter(element: Element): Point;
  tokenVisualCenter(element: HTMLElement): Point;
  resourceToken(playerId: PlayerId, suit: Suit): HTMLElement | null;
  resourceTokenForDeedTransfer(
    playerId: PlayerId,
    suit: Suit
  ): HTMLElement | null;
  crownToken(playerId: PlayerId, suit: Suit): HTMLElement | null;
  districtCard(
    playerId: PlayerId,
    districtId: string,
    cardId: CardId
  ): HTMLElement | null;
  terminalDevelopingCard(cardId: CardId): HTMLElement | null;
  developingCard(cardId: CardId): HTMLElement | null;
  deedToken(cardElement: HTMLElement, suit: Suit): HTMLElement | null;
  deedTokenOnSide(
    cardElement: HTMLElement,
    side: DeedTokenSide,
    suit: Suit
  ): HTMLElement | null;
  deedTokenRail(
    cardElement: HTMLElement,
    side: DeedTokenSide
  ): HTMLElement | null;
  computedStyle(element: Element): CSSStyleDeclaration;
  lane(playerId: PlayerId, districtId: string): HTMLElement | null;
  districtColumn(districtId: string): HTMLElement | null;
  laneFrame(laneElement: HTMLElement): HTMLElement | null;
  laneTargetCenter(
    laneElement: HTMLElement,
    cardHeightPx: number
  ): Point | null;
  laneCardSize(
    laneElement: HTMLElement,
    fallbackElement?: HTMLElement
  ): Size | null;
  deckSource(): HTMLElement | null;
  discardTarget(): HTMLElement | null;
  handSource(playerId: PlayerId, cardId: CardId): HTMLElement | null;
  handDrawTarget(playerId: PlayerId): HTMLElement | null;
}

export function cssEscapeValue(value: string): string {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value);
  }
  return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

export function elementCenter(element: Element): Point {
  const rect = element.getBoundingClientRect();
  return {
    x: rect.left + rect.width / 2,
    y: rect.top + rect.height / 2,
  };
}

export function tokenVisualCenter(element: HTMLElement): Point {
  return elementCenter(element);
}

export function parsePixelValue(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function deedTokenCenterInRail(
  railElement: HTMLElement,
  tokenSizePx: number,
  gapPx: number,
  index: number,
  sideCount: number
): Point {
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

function lastMatch<T extends Element>(matches: readonly T[]): T | null {
  return matches.length > 0 ? matches[matches.length - 1] : null;
}

function middleMatch<T extends Element>(matches: readonly T[]): T | null {
  return matches.length > 0 ? matches[Math.floor(matches.length / 2)] : null;
}

function measureCssLength(
  environment: AnimationDomEnvironment,
  container: HTMLElement,
  propertyName: string,
  cssValue: string,
  fallbackPx: number
): number {
  if (cssValue.length === 0) {
    return fallbackPx;
  }

  const probe = environment.createElement('div');
  probe.style.position = 'absolute';
  probe.style.visibility = 'hidden';
  probe.style.pointerEvents = 'none';
  probe.style.padding = '0';
  probe.style.margin = '0';
  probe.style.border = '0';
  probe.style.width = propertyName === 'width' ? cssValue : '0';
  probe.style.height = propertyName === 'height' ? cssValue : '0';
  container.appendChild(probe);
  const rect = probe.getBoundingClientRect();
  probe.remove();

  const measured = propertyName === 'width' ? rect.width : rect.height;
  return measured > 0 ? measured : fallbackPx;
}

function laneCardSize(
  environment: AnimationDomEnvironment,
  laneElement: HTMLElement,
  fallbackElement?: HTMLElement
): Size | null {
  const explicitTarget = laneElement.querySelector<HTMLElement>(
    '.lane-card-animation-target'
  );
  const explicitRect = explicitTarget?.getBoundingClientRect();
  if (explicitRect && explicitRect.width > 0 && explicitRect.height > 0) {
    return { width: explicitRect.width, height: explicitRect.height };
  }

  const fallbackRect = fallbackElement?.getBoundingClientRect();
  const fallbackWidth = fallbackRect?.width ?? 0;
  const fallbackHeight = fallbackRect?.height ?? 0;
  const laneStyle = environment.getComputedStyle(laneElement);
  const width = measureCssLength(
    environment,
    laneElement,
    'width',
    laneStyle.getPropertyValue('--card-width').trim(),
    fallbackWidth
  );
  const height = measureCssLength(
    environment,
    laneElement,
    'height',
    laneStyle.getPropertyValue('--card-height').trim(),
    fallbackHeight
  );

  return width > 0 && height > 0 ? { width, height } : null;
}

function stackStepForLane(
  environment: AnimationDomEnvironment,
  laneElement: HTMLElement,
  isBotLane: boolean,
  cardHeightPx: number
): number {
  const laneStyle = environment.getComputedStyle(laneElement);
  const stackStepValue = laneStyle.getPropertyValue('--card-stack-step').trim();
  if (stackStepValue.length > 0) {
    const probe = environment.createElement('div');
    probe.style.position = 'absolute';
    probe.style.visibility = 'hidden';
    probe.style.pointerEvents = 'none';
    probe.style.width = '0';
    probe.style.height = stackStepValue;
    probe.style.padding = '0';
    probe.style.margin = '0';
    probe.style.border = '0';
    laneElement.appendChild(probe);
    const measuredStep = probe.getBoundingClientRect().height;
    probe.remove();
    if (measuredStep > 0) {
      return measuredStep;
    }
  }

  const topStackCard = laneElement.querySelector<HTMLElement>(
    '.lane-stack-card:last-child'
  );
  if (topStackCard) {
    const stackPosition = Number.parseFloat(
      topStackCard.style.getPropertyValue('--stack-position')
    );
    if (Number.isFinite(stackPosition) && stackPosition > 0) {
      const computed = environment.getComputedStyle(topStackCard);
      const offsetPx = parsePixelValue(
        isBotLane ? computed.bottom : computed.top,
        0
      );
      if (offsetPx > 0) {
        return offsetPx / stackPosition;
      }
    }
  }

  return cardHeightPx * 0.24;
}

function laneTargetCenter(
  environment: AnimationDomEnvironment,
  laneElement: HTMLElement,
  cardHeightPx: number
): Point | null {
  const explicitTarget = laneElement.querySelector<HTMLElement>(
    '.lane-card-animation-target'
  );
  if (explicitTarget) {
    const targetRect = explicitTarget.getBoundingClientRect();
    if (targetRect.width > 0 && targetRect.height > 0) {
      return elementCenter(explicitTarget);
    }
  }

  const isBotLane = laneElement.classList.contains('is-bot');
  const topCard = laneElement.querySelector<HTMLElement>(
    '.lane-stack-card:last-child .card-tile'
  );
  if (topCard) {
    const center = elementCenter(topCard);
    const stackStep = stackStepForLane(
      environment,
      laneElement,
      isBotLane,
      cardHeightPx
    );
    return {
      x: center.x,
      y: center.y + (isBotLane ? -stackStep : stackStep),
    };
  }

  const frame = laneElement.querySelector<HTMLElement>('.lane-stack-frame');
  if (!frame) {
    return null;
  }
  const frameRect = frame.getBoundingClientRect();
  return {
    x: frameRect.left + frameRect.width / 2,
    y: isBotLane
      ? frameRect.bottom - cardHeightPx / 2
      : frameRect.top + cardHeightPx / 2,
  };
}

export function createAnimationDomTargets(
  environment: AnimationDomEnvironment
): AnimationDomTargets {
  const queryVisibleTokenInRail = (
    playerId: PlayerId,
    rowClassName: string,
    suit: Suit
  ): HTMLElement | null => {
    const selector = `[data-token-rail-player-id="${cssEscapeValue(
      playerId
    )}"] .${rowClassName} .token-chip[data-token-suit="${cssEscapeValue(
      suit
    )}"]`;
    const boardPane = environment.querySelector<HTMLElement>('.board-pane');
    const matches = boardPane
      ? Array.from(boardPane.querySelectorAll<HTMLElement>(selector))
      : environment.querySelectorAll<HTMLElement>(selector);
    return (
      Array.from(matches).find((match) => {
        const rect = match.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      }) ??
      matches[0] ??
      null
    );
  };

  const nestedStackTarget = (
    stackSelector: string,
    anchorSelector?: string
  ): HTMLElement | null => {
    const stack = environment.querySelector<HTMLElement>(stackSelector);
    if (!stack) {
      return null;
    }
    return (
      (anchorSelector
        ? stack.querySelector<HTMLElement>(anchorSelector)
        : null) ??
      lastMatch(
        Array.from(stack.querySelectorAll<HTMLElement>('.deck-pile-stack-card'))
      ) ?? stack
    );
  };

  const hiddenHandTarget = (escapedPlayerId: string): HTMLElement | null => {
    const panelSelector = `.player-panel[data-player-id="${escapedPlayerId}"]`;
    const hiddenSelector = `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="hidden"]`;
    return (
      environment.querySelector<HTMLElement>(
        `${panelSelector} [data-hand-owner-id="${escapedPlayerId}"][data-hand-animation-anchor="true"]`
      ) ??
      middleMatch(environment.querySelectorAll<HTMLElement>(hiddenSelector)) ??
      environment.querySelector<HTMLElement>(hiddenSelector)
    );
  };

  return {
    isAvailable: () => environment.isAvailable(),
    viewportCenterY: () => environment.viewportHeight() / 2,
    elementCenter,
    tokenVisualCenter,
    resourceToken: (playerId, suit) =>
      queryVisibleTokenInRail(playerId, 'rail-resources-row', suit),
    resourceTokenForDeedTransfer: (playerId, suit) =>
      environment.querySelector<HTMLElement>(
        `[data-token-rail-player-id="${cssEscapeValue(
          playerId
        )}"] .rail-resources-row .token-chip[data-token-suit="${cssEscapeValue(
          suit
        )}"]`
      ),
    crownToken: (playerId, suit) =>
      queryVisibleTokenInRail(playerId, 'crowns-rail-row', suit),
    districtCard: (playerId, districtId, cardId) =>
      environment.querySelector<HTMLElement>(
        `.district-column[data-district-id="${cssEscapeValue(
          districtId
        )}"] .district-lane[data-lane-player-id="${cssEscapeValue(
          playerId
        )}"] .card-tile[data-card-id="${cssEscapeValue(cardId)}"]`
      ),
    terminalDevelopingCard: (cardId) =>
      environment.querySelector<HTMLElement>(
        `.district-strip .card-tile[data-card-id="${cssEscapeValue(
          cardId
        )}"][data-in-development="true"]`
      ),
    developingCard: (cardId) =>
      environment.querySelector<HTMLElement>(
        `.district-strip .card-tile[data-card-id="${cssEscapeValue(
          cardId
        )}"][data-in-development="true"]`
      ) ??
      environment.querySelector<HTMLElement>(
        `.district-strip .card-tile[data-card-id="${cssEscapeValue(cardId)}"]`
      ),
    deedToken: (cardElement, suit) =>
      cardElement.querySelector<HTMLElement>(
        `.card-side-token-rail .token-chip[data-token-suit="${cssEscapeValue(
          suit
        )}"]`
      ),
    deedTokenOnSide: (cardElement, side, suit) =>
      cardElement.querySelector<HTMLElement>(
        `.card-side-token-rail-${side} .token-chip[data-token-suit="${cssEscapeValue(
          suit
        )}"]`
      ),
    deedTokenRail: (cardElement, side) =>
      cardElement.querySelector<HTMLElement>(`.card-side-token-rail-${side}`),
    computedStyle: (element) => environment.getComputedStyle(element),
    lane: (playerId, districtId) =>
      environment.querySelector<HTMLElement>(
        `.district-column[data-district-id="${cssEscapeValue(
          districtId
        )}"] .district-lane[data-lane-player-id="${cssEscapeValue(playerId)}"]`
      ),
    districtColumn: (districtId) =>
      environment.querySelector<HTMLElement>(
        `.district-column[data-district-id="${cssEscapeValue(districtId)}"]`
      ),
    laneFrame: (laneElement) =>
      laneElement.querySelector<HTMLElement>('.lane-stack-frame'),
    laneTargetCenter: (laneElement, cardHeightPx) =>
      laneTargetCenter(environment, laneElement, cardHeightPx),
    laneCardSize: (laneElement, fallbackElement) =>
      laneCardSize(environment, laneElement, fallbackElement),
    deckSource: () =>
      nestedStackTarget(
        '.deck-pile-stack.is-deck',
        '.deck-pile-animation-anchor'
      ),
    discardTarget: () =>
      nestedStackTarget(
        '.deck-pile-stack.is-discard',
        '.discard-pile-animation-anchor'
      ),
    handSource: (playerId, cardId) => {
      const escapedPlayerId = cssEscapeValue(playerId);
      const escapedCardId = cssEscapeValue(cardId);
      const panelSelector = `.player-panel[data-player-id="${escapedPlayerId}"]`;
      return (
        environment.querySelector<HTMLElement>(
          `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-hand-card-id="${escapedCardId}"]`
        ) ??
        environment.querySelector<HTMLElement>(
          `${panelSelector} .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="occupied"][data-card-id="${escapedCardId}"]`
        ) ??
        hiddenHandTarget(escapedPlayerId)
      );
    },
    handDrawTarget: (playerId) => {
      const escapedPlayerId = cssEscapeValue(playerId);
      return (
        environment.querySelector<HTMLElement>(
          `.player-panel[data-player-id="${escapedPlayerId}"] .card-tile[data-hand-owner-id="${escapedPlayerId}"][data-hand-slot-kind="empty"]`
        ) ??
        hiddenHandTarget(escapedPlayerId)
      );
    },
  };
}

const browserEnvironment: AnimationDomEnvironment = {
  isAvailable: () => typeof document !== 'undefined',
  querySelector: (selectors) =>
    typeof document === 'undefined' ? null : document.querySelector(selectors),
  querySelectorAll: (selectors) =>
    typeof document === 'undefined'
      ? []
      : Array.from(document.querySelectorAll(selectors)),
  createElement: (tagName) => document.createElement(tagName),
  getComputedStyle: (element) => window.getComputedStyle(element),
  viewportHeight: () => window.innerHeight,
};

export const browserAnimationDomTargets =
  createAnimationDomTargets(browserEnvironment);
